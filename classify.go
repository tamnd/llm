package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// State is what a call found. The distinctions exist because the remedies
// differ and nothing else about them matters: out of quota is a wait, a
// rejected key is a login, a missing model is an edit to the route file, a
// dead tunnel is bringing the tunnel back up, Broken is a bug, and Gone is a
// route to retire for the life of the process.
type State string

const (
	StateLive         State = "live"
	StateQuota        State = "quota"
	StateUnauthorized State = "unauthorized"
	StateUnreachable  State = "unreachable"
	StateBroken       State = "broken"
	StateGone         State = "gone"
	StateUnknown      State = "unknown"
)

// Usable reports whether a route in this state is worth sending work to. An
// unprobed route counts, because the first call is itself a probe.
func (s State) Usable() bool { return s == StateLive || s == StateUnknown }

// Signal is a classified provider response.
//
// This lives in the root package rather than in route, which is where it was
// written, because the classification is a fact about a provider response and
// the router is only one of the three things that needs it: the client decides
// whether to retry from it, the router decides how long to cool a route down,
// and the ledger records it.
type Signal struct {
	State  State
	Detail string
	// RetryAfter is what the provider asked for, from the header or from a
	// body field. Zero means it said nothing.
	RetryAfter time.Duration
	// ResetsAt is an absolute instant if the provider gave one. Zero means
	// unknown and must not be read as the epoch.
	ResetsAt time.Time
}

// quotaMarkers are copied from live bodies rather than guessed. A browser
// session announces its daily limit in more than one way depending on which
// surface refused.
var quotaMarkers = []string{
	"usage_limit_reached",
	"rate_limit_exceeded",
	"too many requests",
	"quota",
}

var goneMarkers = []string{
	"is not supported",
	"model_not_found",
	"unknown model",
	"no endpoints found",
}

// Classify maps a provider response to a state.
//
// Order matters: a body naming a missing model is a route file problem even
// when it arrives with a status that would otherwise read as something else. A
// free gateway answers an image request with 404 and "No endpoints found that
// support image input", and treating that as a transport fault costs an hour
// of retries against a route that will never take the call.
func Classify(status int, header http.Header, body []byte) Signal {
	text := string(body)
	lower := strings.ToLower(text)
	after := ParseRetryAfter(header.Get("Retry-After"))
	switch {
	case containsAny(lower, goneMarkers):
		return Signal{State: StateGone, Detail: message(text, "model is not served here")}
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		return Signal{State: StateUnauthorized, Detail: message(text, "credential rejected")}
	case status == http.StatusTooManyRequests || containsAny(lower, quotaMarkers):
		resets := resetsFrom(text)
		if resets.IsZero() && after > 0 {
			resets = time.Now().UTC().Add(after)
		}
		if after == 0 {
			after = time.Until(resets)
		}
		return Signal{State: StateQuota, Detail: message(text, "out of quota"),
			RetryAfter: max(0, after), ResetsAt: resets}
	case strings.Contains(lower, "no chatgpt response found"):
		// The browser transport reached the page and found nothing on it. It is
		// a real failure of this host, not of the request, and it clears.
		return Signal{State: StateBroken, Detail: "no ChatGPT response found", RetryAfter: after}
	case status >= 400:
		return Signal{State: StateBroken,
			Detail: message(text, fmt.Sprintf("host returned %d", status)), RetryAfter: after}
	case strings.TrimSpace(text) == "":
		// A 200 with nothing in it is worse than an error, because it looks
		// like an answer. It must never reach the caller as a candidate.
		return Signal{State: StateBroken, Detail: "empty response"}
	}
	return Signal{State: StateLive, Detail: "ok"}
}

// ClassifyError reads a Go-side error. The client folds the upstream status and
// body into its error text, so this covers both a dead tunnel and a relayed
// provider message.
func ClassifyError(err error) Signal {
	if err == nil {
		return Signal{State: StateLive, Detail: "ok"}
	}
	if errors.Is(err, context.Canceled) {
		return Signal{State: StateUnknown, Detail: err.Error()}
	}
	text := Condense(err.Error())
	lower := strings.ToLower(text)
	switch {
	case containsAny(lower, goneMarkers):
		return Signal{State: StateGone, Detail: text}
	case containsAny(lower, []string{"401", "403", "unauthorized", "forbidden", "invalid_api_key"}):
		return Signal{State: StateUnauthorized, Detail: text}
	case containsAny(lower, quotaMarkers) || strings.Contains(lower, "429"):
		resets := resetsFrom(text)
		return Signal{State: StateQuota, Detail: text, ResetsAt: resets,
			RetryAfter: max(0, time.Until(resets))}
	case errors.Is(err, context.DeadlineExceeded) ||
		containsAny(lower, []string{"no such host", "connection refused", "dial tcp", "tls:", "i/o timeout", "timeout", "eof"}):
		// connection refused on a loopback port is the ordinary shape of a
		// tunnel that died, which is why it is unreachable and not broken.
		return Signal{State: StateUnreachable, Detail: text}
	}
	return Signal{State: StateBroken, Detail: text}
}

func containsAny(lower string, markers []string) bool {
	for _, marker := range markers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

var (
	// The quotes are optional because this also runs over an error string that
	// folded a JSON body into a sentence.
	resetsAtPattern   = regexp.MustCompile(`"?resets_at"?\s*[:= ]\s*(\d+)`)
	retryDelayPattern = regexp.MustCompile(`(?i)"?retry(?:_delay|_after|-after|afterseconds)"?\s*[:=]\s*"?(\d+)`)
)

// resetsFrom digs the reset instant out of whatever shape the host used. A host
// that offers nothing leaves it zero, and a caller must read a zero instant as
// unknown rather than as the epoch.
func resetsFrom(body string) time.Time {
	if match := resetsAtPattern.FindStringSubmatch(body); match != nil {
		if seconds, err := strconv.ParseInt(match[1], 10, 64); err == nil && seconds > 0 {
			return time.Unix(seconds, 0).UTC()
		}
	}
	if match := retryDelayPattern.FindStringSubmatch(body); match != nil {
		if seconds, err := strconv.Atoi(match[1]); err == nil && seconds > 0 {
			return time.Now().UTC().Add(time.Duration(seconds) * time.Second)
		}
	}
	return time.Time{}
}

// message pulls a human sentence out of a body, falling back to a description
// when the body is not the shape we expected.
func message(body, fallback string) string {
	if text := ErrorMessage([]byte(body)); text != "" {
		return text
	}
	return fallback
}

// ErrorMessage pulls the human sentence out of an error body. Falling back to
// the raw body matters: a proxy relays some upstream failures as plain text,
// and reporting an empty detail for those would hide the only clue.
func ErrorMessage(raw []byte) string {
	var envelope struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
		Message string `json:"message"`
	}
	if json.Unmarshal(raw, &envelope) == nil {
		if text := strings.TrimSpace(envelope.Error.Message); text != "" {
			return Condense(text)
		}
		if text := strings.TrimSpace(envelope.Message); text != "" {
			return Condense(text)
		}
	}
	return Condense(string(raw))
}

// SmallModel reports whether a slug names a cut-down model, from the slug
// alone. It is here rather than in route because the audit needs it about a
// recorded model name long after the route is gone.
func SmallModel(name string) bool {
	lower := strings.ToLower(name)
	for _, marker := range []string{"mini", "flash", "lightning", "nano", "small", "lite", "tiny"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}
