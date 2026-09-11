package route

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"slices"
	"strings"
	"time"

	"github.com/tamnd/llm"
)

// The classification lives in the root package, because it is a fact about a
// provider response rather than about routing. These aliases keep route's own
// API readable without a second definition to drift from the first.
type (
	State  = llm.State
	Signal = llm.Signal
)

const (
	StateLive         = llm.StateLive
	StateQuota        = llm.StateQuota
	StateUnauthorized = llm.StateUnauthorized
	StateUnreachable  = llm.StateUnreachable
	StateBroken       = llm.StateBroken
	StateGone         = llm.StateGone
	StateUnknown      = llm.StateUnknown
)

// Health is one probe result.
type Health struct {
	Route     string        `json:"route"`
	State     State         `json:"state"`
	Latency   time.Duration `json:"latency"`
	Detail    string        `json:"detail"`
	ResetsAt  time.Time     `json:"resets_at,omitzero"`
	Model     string        `json:"model,omitempty"`
	CheckedAt time.Time     `json:"checked_at"`

	// Transport is what a session pool says it is fronting.
	Transport string `json:"transport,omitempty"`
	// Verified is how many sessions the pool has logged in. It is the number
	// that silently drops when a session expires, and a host at zero answers
	// every call with a refusal that looks like a model failure.
	Verified int `json:"verified,omitempty"`
	// Declared is the concurrency the host announces, which is not always the
	// one the route file assumes.
	Declared int `json:"declared,omitempty"`
	// Catalogue is the model list the host advertises.
	Catalogue []string `json:"catalogue,omitempty"`
	// Drift is set when the configured model is absent from the catalogue.
	Drift string `json:"drift,omitempty"`
	// Answered is the model that came back on a deep probe, which is not
	// always the one that was asked for. An account can be moved down between
	// two runs and neither the route file nor the catalogue can see it: both
	// say what is on offer, and this says what arrived.
	Answered string `json:"answered,omitempty"`
}

// Downgraded reports that the route answered on a model other than the one it
// was asked for.
//
// Only a deep probe can know, since it is the only one that asks a question. A
// shallow probe leaves Answered empty and this is false, which is the honest
// answer: not knowing is not the same as knowing it is fine.
func (h Health) Downgraded() bool {
	return h.Answered != "" && h.Model != "" && h.Answered != h.Model
}

// Prober runs health checks.
//
// The check is deliberately not a model call. A measured round trip to one of
// these boxes for nine tokens is 151 seconds, so probing three hosts that way
// costs eight minutes and would make a doctor command useless as a cron guard.
// GET /health answers in milliseconds and says the thing that actually goes
// wrong, which is that the pool's sessions have logged themselves out. Deep
// asks for the model call as well, for when the question is whether the whole
// path works.
type Prober struct {
	HTTPClient *http.Client
	Timeout    time.Duration
	Deep       bool
	Now        func() time.Time
	// Look says where a program is, and is exec.LookPath unless a test says
	// otherwise. It is the whole of what an exec route can be asked without
	// running the thing.
	Look func(name string) (string, error)
}

// ProbePrompt is trivial on purpose. A deep probe of a fleet should cost a few
// hundred tokens, not a page.
const ProbePrompt = "Reply with the single word ok."

// Probe checks one route: the host's own health first, then its catalogue,
// then optionally a real completion.
func (p Prober) Probe(ctx context.Context, value Route) Health {
	started := p.now()
	result := Health{Route: value.Name, Model: value.Model, CheckedAt: started}
	finish := func(signal Signal) Health {
		result.State = signal.State
		result.Detail = signal.Detail
		result.ResetsAt = signal.ResetsAt
		result.Latency = p.now().Sub(started)
		return result
	}
	if err := value.Validate(); err != nil {
		return finish(Signal{State: StateBroken, Detail: err.Error()})
	}

	switch value.Kind {
	case KindExec:
		// There is no /health to ask, no catalogue to list and no key to be
		// rejected, so the whole of the shallow probe is whether the program
		// is on this machine. Everything below speaks HTTP and would report a
		// working route as broken.
		return finish(p.command(value))
	case KindReader:
		// A reader is reached over ssh and its model server listens on a
		// loopback address on the far side of it, which means nothing here.
		// Readiness for one is llm/fleet's question, not this package's.
		result.Transport = "reader"
		return finish(Signal{State: StateUnknown,
			Detail: fmt.Sprintf("%s on %s, probed by fleet rather than here", value.Reader, value.Host)})
	}

	var body healthBody
	if value.Kind.HasHealth() {
		var signal Signal
		body, signal = p.health(ctx, value)
		if signal.State != StateLive {
			return finish(signal)
		}
		result.Transport = body.Transport
		result.Verified = body.Pool.Verified
		result.Declared = body.Pool.Concurrency
		if body.Pool.Verified == 0 {
			// The host is up and has nothing to answer with. Sending it work
			// would produce refusals that read like model failures on every
			// page.
			return finish(Signal{State: StateBroken, Detail: "no verified sessions in the pool"})
		}
	}

	catalogue, signal := p.catalogue(ctx, value)
	result.Catalogue = catalogue
	if signal.State == StateUnreachable || signal.State == StateUnauthorized {
		return finish(signal)
	}
	// For a route with no /health this is the only question that was asked, so
	// any answer short of live is the state of the route and there is nothing
	// to fall through to.
	if !value.Kind.HasHealth() && signal.State != StateLive {
		return finish(signal)
	}
	// A catalogue that answers and does not name the model is the clearest
	// possible statement that the route file is stale.
	wire := value.Wire()
	if len(catalogue) > 0 && !slices.Contains(catalogue, wire) {
		result.Drift = fmt.Sprintf("model %s is not in the %s catalogue, which lists %d: %s",
			wire, value.Name, len(catalogue), strings.Join(catalogue, ", "))
		return finish(Signal{State: StateGone, Detail: result.Drift})
	}

	detail := fmt.Sprintf("%s, %d verified", body.Transport, body.Pool.Verified)
	if !value.Kind.HasHealth() {
		result.Transport = string(value.Kind)
		detail = fmt.Sprintf("%s, %d models", value.Kind, len(catalogue))
	}
	if !p.Deep {
		return finish(Signal{State: StateLive, Detail: detail})
	}

	client, err := value.Client(p.completionTimeout(), 0)
	if err != nil {
		return finish(Signal{State: StateBroken, Detail: err.Error()})
	}
	response, err := client.Complete(ctx, llm.Request{
		Model:        wire,
		Instructions: "Answer in one word.",
		Input:        ProbePrompt,
	})
	if err != nil {
		return finish(llm.ClassifyError(err))
	}
	if strings.TrimSpace(response.Text) == "" {
		return finish(Signal{State: StateBroken, Detail: "completed with empty text"})
	}
	result.Answered = response.Model
	detail = fmt.Sprintf("%s, answered in %s", detail, response.Elapsed.Round(time.Second))
	if result.Downgraded() {
		// The route still works, so it is not broken and not gone: work sent
		// to it will come back, on a lesser model. That is a thing a person
		// decides about, and the only way anybody can decide is if it is said.
		detail = fmt.Sprintf("%s on %s, not %s", detail, response.Model, value.Model)
	}
	return finish(Signal{State: StateLive, Detail: detail})
}

// command is the probe for a route that is a program on this machine.
//
// Being on PATH is the whole of the shallow check. Whether the subscription
// behind it is signed in and has turns left is not something to ask without
// spending one, so a deep probe asks the question the same way every other
// route's deep probe does, by putting a real question, and that is done
// through llm/exec rather than here.
func (p Prober) command(value Route) Signal {
	look := p.Look
	if look == nil {
		look = exec.LookPath
	}
	path, err := look(value.Command)
	if err != nil {
		return Signal{State: StateUnreachable,
			Detail: fmt.Sprintf("%s is not on PATH on this machine: %v", value.Command, err)}
	}
	return Signal{State: StateLive, Detail: fmt.Sprintf("command, %s", path)}
}

func (p Prober) now() time.Time {
	if p.Now != nil {
		return p.Now()
	}
	return time.Now()
}

func (p Prober) httpClient() *http.Client {
	if p.HTTPClient != nil {
		return p.HTTPClient
	}
	timeout := p.Timeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return &http.Client{Timeout: timeout}
}

func (p Prober) completionTimeout() time.Duration {
	if p.Timeout > 0 {
		return p.Timeout
	}
	// The known baseline is 151 s for nine tokens. Five minutes is slack over
	// that without waiting out a host that has stopped answering entirely.
	return 5 * time.Minute
}

// healthBody is the shape a session proxy returns. Its profile list carries
// account email addresses, so it is read and discarded: nothing here keeps
// them, and no report can write what it never received.
type healthBody struct {
	Status    string `json:"status"`
	Transport string `json:"transport"`
	Pool      struct {
		Verified    int `json:"verified"`
		Concurrency int `json:"concurrency"`
	} `json:"pool"`
}

func (p Prober) health(ctx context.Context, value Route) (healthBody, Signal) {
	raw, status, header, err := p.get(ctx, value, "/health")
	if err != nil {
		return healthBody{}, llm.ClassifyError(err)
	}
	if status < 200 || status >= 300 {
		return healthBody{}, llm.Classify(status, header, raw)
	}
	var body healthBody
	if err := json.Unmarshal(raw, &body); err != nil {
		return healthBody{}, Signal{State: StateBroken, Detail: "health is not valid JSON: " + err.Error()}
	}
	if body.Status != "" && body.Status != "ok" {
		return body, Signal{State: StateBroken, Detail: "health says " + body.Status}
	}
	return body, Signal{State: StateLive, Detail: "ok"}
}

// catalogue calls GET /v1/models. A route whose catalogue call fails is not
// condemned for that alone, because not every compatible server implements it.
func (p Prober) catalogue(ctx context.Context, value Route) ([]string, Signal) {
	raw, status, header, err := p.get(ctx, value, "/models")
	if err != nil {
		return nil, llm.ClassifyError(err)
	}
	if status < 200 || status >= 300 {
		return nil, llm.Classify(status, header, raw)
	}
	var envelope struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, Signal{State: StateBroken, Detail: "catalogue is not valid JSON: " + err.Error()}
	}
	models := make([]string, 0, len(envelope.Data))
	for _, item := range envelope.Data {
		if item.ID != "" {
			models = append(models, item.ID)
		}
	}
	return models, Signal{State: StateLive, Detail: "ok"}
}

func (p Prober) get(ctx context.Context, value Route, path string) ([]byte, int, http.Header, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, value.Endpoint(path), nil)
	if err != nil {
		return nil, 0, nil, err
	}
	if key := value.Key(); key != "" {
		request.Header.Set("Authorization", "Bearer "+key)
	}
	request.Header.Set("User-Agent", llm.UserAgent())
	response, err := p.httpClient().Do(request)
	if err != nil {
		return nil, 0, nil, err
	}
	defer func() { _ = response.Body.Close() }()
	raw, err := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	if err != nil {
		return nil, response.StatusCode, response.Header, err
	}
	return raw, response.StatusCode, response.Header, nil
}
