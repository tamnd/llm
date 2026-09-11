package route

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// proxy stands in for a session proxy: a /health with a pool count on it and a
// /v1/models catalogue.
func proxy(t *testing.T, health, models string, status int) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/health"):
			if status != 0 {
				w.WriteHeader(status)
			}
			_, _ = w.Write([]byte(health))
		case strings.HasSuffix(r.URL.Path, "/models"):
			_, _ = w.Write([]byte(models))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

const catalogue = `{"data":[{"id":"test-model"},{"id":"other-model"}]}`

func poolRoute(url string) Route {
	return Route{Name: "box", Kind: KindPool, Host: "box", BaseURL: url + "/v1", Model: "test-model"}
}

func TestProbeLive(t *testing.T) {
	server := proxy(t, `{"status":"ok","transport":"browser","pool":{"verified":3,"concurrency":2}}`, catalogue, 0)
	health := Prober{HTTPClient: server.Client()}.Probe(context.Background(), poolRoute(server.URL))
	if health.State != StateLive {
		t.Fatalf("state = %q: %s", health.State, health.Detail)
	}
	if health.Verified != 3 || health.Declared != 2 {
		t.Errorf("pool = %d verified, %d declared", health.Verified, health.Declared)
	}
	if len(health.Catalogue) != 2 {
		t.Errorf("catalogue = %v", health.Catalogue)
	}
}

// A host that is up with no session in its pool answers every call with a
// refusal that reads like a model failure. Saying so here is the difference
// between one clear line and a batch of unexplained rejections.
func TestProbeEmptyPoolIsBroken(t *testing.T) {
	server := proxy(t, `{"status":"ok","transport":"browser","pool":{"verified":0,"concurrency":2}}`, catalogue, 0)
	health := Prober{HTTPClient: server.Client()}.Probe(context.Background(), poolRoute(server.URL))
	if health.State != StateBroken {
		t.Fatalf("state = %q, want broken", health.State)
	}
	if !strings.Contains(health.Detail, "verified") {
		t.Errorf("detail = %q, want it to name the empty pool", health.Detail)
	}
}

// A catalogue that answers and does not name the model is the clearest
// possible statement that the route file is stale.
func TestProbeCatalogueDrift(t *testing.T) {
	server := proxy(t, `{"status":"ok","pool":{"verified":1}}`, `{"data":[{"id":"something-else"}]}`, 0)
	health := Prober{HTTPClient: server.Client()}.Probe(context.Background(), poolRoute(server.URL))
	if health.State != StateGone {
		t.Fatalf("state = %q, want gone", health.State)
	}
	if health.Drift == "" || !strings.Contains(health.Drift, "test-model") {
		t.Errorf("drift = %q", health.Drift)
	}
}

func TestProbeQuotaCarriesTheReset(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "600")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"out of turns"}}`))
	}))
	t.Cleanup(server.Close)

	health := Prober{HTTPClient: server.Client()}.Probe(context.Background(), poolRoute(server.URL))
	if health.State != StateQuota {
		t.Fatalf("state = %q, want quota: %s", health.State, health.Detail)
	}
	if left := time.Until(health.ResetsAt); left < 9*time.Minute || left > 11*time.Minute {
		t.Errorf("resets in %s, want the ten minutes the header asked for", left)
	}
}

// An exec route has no endpoint to get and no key to be rejected. The whole of
// the shallow check is whether the program is on this machine.
func TestProbeExecChecksThePath(t *testing.T) {
	route := Route{Name: "cli", Kind: KindExec, Command: "codex", Model: "m"}
	found := Prober{Look: func(string) (string, error) { return "/usr/local/bin/codex", nil }}
	if health := found.Probe(context.Background(), route); health.State != StateLive {
		t.Errorf("state = %q, want live: %s", health.State, health.Detail)
	}
	missing := Prober{Look: func(string) (string, error) { return "", errors.New("not found") }}
	health := missing.Probe(context.Background(), route)
	if health.State != StateUnreachable {
		t.Errorf("state = %q, want unreachable", health.State)
	}
	if !strings.Contains(health.Detail, "PATH") {
		t.Errorf("detail = %q, want it to say where it looked", health.Detail)
	}
}

// A reader's model server listens on a loopback address on the far side of an
// ssh hop, which means nothing from here. Saying unknown is honest; saying
// unreachable would be a lie about a working host.
func TestProbeReaderDefersToTheFleet(t *testing.T) {
	route := Route{Name: "gpu", Kind: KindReader, Host: "box", Reader: "local-ocr", Model: "m"}
	health := Prober{}.Probe(context.Background(), route)
	if health.State != StateUnknown {
		t.Errorf("state = %q, want unknown", health.State)
	}
	if !strings.Contains(health.Detail, "fleet") {
		t.Errorf("detail = %q, want it to name who does ask", health.Detail)
	}
}

func TestProbeRefusesABrokenRoute(t *testing.T) {
	health := Prober{}.Probe(context.Background(), Route{Name: "x", Kind: KindGateway, Model: "m"})
	if health.State != StateBroken {
		t.Errorf("state = %q, want broken", health.State)
	}
}

// A gateway has no /health. Asking for one and reporting its absence would
// condemn every route that is merely a plain compatible endpoint.
func TestProbeGatewaySkipsHealth(t *testing.T) {
	var asked []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked = append(asked, r.URL.Path)
		_, _ = w.Write([]byte(catalogue))
	}))
	t.Cleanup(server.Close)

	route := Route{Name: "g", Kind: KindGateway, BaseURL: server.URL + "/v1", Model: "test-model"}
	health := Prober{HTTPClient: server.Client()}.Probe(context.Background(), route)
	if health.State != StateLive {
		t.Fatalf("state = %q: %s", health.State, health.Detail)
	}
	for _, path := range asked {
		if strings.HasSuffix(path, "/health") {
			t.Errorf("asked a gateway for %s", path)
		}
	}
}

func TestDeepProbeNoticesADowngrade(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/models") {
			_, _ = w.Write([]byte(catalogue))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		// The account was moved down between two runs. Neither the route file
		// nor the catalogue can see it: both say what is on offer, and only
		// this says what arrived.
		_, _ = w.Write([]byte("data: {\"id\":\"1\",\"model\":\"test-model-mini\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n"))
	}))
	t.Cleanup(server.Close)

	route := Route{Name: "g", Kind: KindGateway, BaseURL: server.URL + "/v1", Model: "test-model"}
	health := Prober{HTTPClient: server.Client(), Deep: true, Timeout: 5 * time.Second}.
		Probe(context.Background(), route)
	if health.State != StateLive {
		t.Fatalf("state = %q: %s", health.State, health.Detail)
	}
	if !health.Downgraded() {
		t.Errorf("answered on %q against a configured %q and was not called a downgrade",
			health.Answered, health.Model)
	}
	if !strings.Contains(health.Detail, "not test-model") {
		t.Errorf("detail = %q, want it to say what arrived instead", health.Detail)
	}
}

// A shallow probe asks no question, so it knows nothing about what would
// arrive. Not knowing is not the same as knowing it is fine.
func TestShallowProbeDoesNotClaimADowngrade(t *testing.T) {
	health := Health{Model: "big"}
	if health.Downgraded() {
		t.Error("a probe that asked nothing reported on the answer")
	}
}
