package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestClassify(t *testing.T) {
	for _, c := range []struct {
		name   string
		status int
		header http.Header
		body   string
		want   State
	}{
		// A free gateway answers an image request this way. Reading it as a
		// transport fault costs an hour of retries against a route that will
		// never take the call, which is why it is Gone and not Broken.
		{"no endpoints for images", 404, nil,
			`{"error":{"message":"No endpoints found that support image input"}}`, StateGone},
		{"model not found", 400, nil, `{"error":{"code":"model_not_found"}}`, StateGone},
		{"rejected key", 401, nil, `{"error":{"message":"invalid key"}}`, StateUnauthorized},
		{"forbidden", 403, nil, `{"error":{"message":"no"}}`, StateUnauthorized},
		{"too many requests", 429, nil, `{"error":{"message":"slow down"}}`, StateQuota},
		{"quota named in a 200 body", 200, nil, `{"error":{"message":"usage_limit_reached"}}`, StateQuota},
		{"the page had nothing on it", 200, nil, `No ChatGPT response found`, StateBroken},
		{"a 500 from the host", 500, nil, `upstream exploded`, StateBroken},
		{"an empty 200", 200, nil, "  ", StateBroken},
		{"an answer", 200, nil, `{"choices":[]}`, StateLive},
	} {
		t.Run(c.name, func(t *testing.T) {
			signal := Classify(c.status, c.header, []byte(c.body))
			if signal.State != c.want {
				t.Errorf("state = %q, want %q (detail %q)", signal.State, c.want, signal.Detail)
			}
			if signal.Detail == "" {
				t.Error("no detail, and the detail is the only thing a person reads")
			}
		})
	}
}

func TestClassifyReadsRetryAfter(t *testing.T) {
	header := http.Header{"Retry-After": []string{"53931"}}
	signal := Classify(429, header, []byte(`{"error":{"message":"out of turns"}}`))
	if signal.State != StateQuota {
		t.Fatalf("state = %q, want quota", signal.State)
	}
	if signal.RetryAfter != 53931*time.Second {
		t.Errorf("retry after = %s, want the header's fifteen hours", signal.RetryAfter)
	}
	// An hour and a half of cooldown is a fact about the route, and a router
	// that did not know when it ends would either wait it out or hammer it.
	if left := time.Until(signal.ResetsAt); left < 53000*time.Second {
		t.Errorf("resets at %s, which is %s away", signal.ResetsAt, left)
	}
}

func TestClassifyReadsAResetInstantOutOfTheBody(t *testing.T) {
	at := time.Now().UTC().Add(90 * time.Minute).Truncate(time.Second)
	body := fmt.Sprintf(`{"error":{"message":"out of turns","resets_at":%d}}`, at.Unix())
	signal := Classify(429, nil, []byte(body))
	if !signal.ResetsAt.Equal(at) {
		t.Errorf("resets at %s, want %s", signal.ResetsAt, at)
	}
}

// A host that says nothing about when it comes back must leave the instant
// zero. Reading an unset instant as the epoch would make every such route
// look ready the moment it failed.
func TestClassifyLeavesAnUnknownResetZero(t *testing.T) {
	signal := Classify(429, nil, []byte(`{"error":{"message":"out of turns"}}`))
	if !signal.ResetsAt.IsZero() {
		t.Errorf("resets at %s, want the zero instant", signal.ResetsAt)
	}
	if signal.RetryAfter != 0 {
		t.Errorf("retry after = %s, want nothing", signal.RetryAfter)
	}
}

func TestClassifyError(t *testing.T) {
	for _, c := range []struct {
		name string
		err  error
		want State
	}{
		{"nothing went wrong", nil, StateLive},
		{"cancelled by the caller", context.Canceled, StateUnknown},
		// connection refused on a loopback port is the ordinary shape of a
		// tunnel that died, and the remedy is to bring it back up rather than
		// to go looking for a bug.
		{"dead tunnel", errors.New("dial tcp 127.0.0.1:8081: connect: connection refused"), StateUnreachable},
		{"deadline", context.DeadlineExceeded, StateUnreachable},
		{"relayed 401", errors.New("chat completions returned 401 Unauthorized: bad key"), StateUnauthorized},
		{"relayed 429", errors.New("chat completions returned 429 Too Many Requests: slow down"), StateQuota},
		{"relayed missing model", errors.New("unknown model: nope"), StateGone},
		{"anything else", errors.New("something odd"), StateBroken},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := ClassifyError(c.err).State; got != c.want {
				t.Errorf("state = %q, want %q", got, c.want)
			}
		})
	}
}

func TestStateUsable(t *testing.T) {
	// Unknown is usable because the first call is itself a probe; everything
	// else has been measured and found wanting.
	for state, want := range map[State]bool{
		StateLive: true, StateUnknown: true,
		StateQuota: false, StateUnauthorized: false, StateUnreachable: false,
		StateBroken: false, StateGone: false,
	} {
		if got := state.Usable(); got != want {
			t.Errorf("%q usable = %t, want %t", state, got, want)
		}
	}
}

func TestErrorMessageFallsBackToTheRawBody(t *testing.T) {
	// A proxy relays some upstream failures as plain text. Reporting an empty
	// detail for those hides the only clue there is.
	if got := ErrorMessage([]byte("upstream  closed\nthe   connection")); got != "upstream closed the connection" {
		t.Errorf("message = %q", got)
	}
	if got := ErrorMessage([]byte(`{"message":"said so at the top level"}`)); got != "said so at the top level" {
		t.Errorf("message = %q", got)
	}
}

func TestParseRetryAfter(t *testing.T) {
	if got := ParseRetryAfter("120"); got != 2*time.Minute {
		t.Errorf("seconds: %s", got)
	}
	when := time.Now().UTC().Add(time.Hour).Round(time.Second)
	if got := ParseRetryAfter(when.Format(http.TimeFormat)); got < 59*time.Minute || got > 61*time.Minute {
		t.Errorf("http date: %s", got)
	}
	if got := ParseRetryAfter("not a number"); got != 0 {
		t.Errorf("nonsense: %s", got)
	}
	// A date in the past is a header written by a clock that is not ours. It
	// must not come back as a negative wait.
	if got := ParseRetryAfter(time.Now().Add(-time.Hour).Format(http.TimeFormat)); got < 0 {
		t.Errorf("a stale date gave %s", got)
	}
}

func TestSmallModel(t *testing.T) {
	for _, name := range []string{"gpt-5.4-mini", "gemini-flash", "something-nano"} {
		if !SmallModel(name) {
			t.Errorf("%q read as a full model", name)
		}
	}
	if SmallModel("gpt-5.4") {
		t.Error("a full model read as a cut down one")
	}
}

func TestCondense(t *testing.T) {
	got := Condense("  a  line\nand\tanother  ")
	if got != "a line and another" {
		t.Errorf("condensed = %q", got)
	}
	long := Condense(strings.Repeat("x", 500))
	if len(long) > 210 {
		t.Errorf("condensed to %d characters, want a table cell's worth", len(long))
	}
}

func TestUsage(t *testing.T) {
	// A proxy that reports cached tokens above the input count, or leaves the
	// total out, must not produce a negative or missing number downstream.
	got := Usage{InputTokens: 100, CachedInputTokens: 500, OutputTokens: 10}.Normalized()
	if got.CachedInputTokens != 100 {
		t.Errorf("cached = %d, want it clamped to the input", got.CachedInputTokens)
	}
	if got.TotalTokens != 110 {
		t.Errorf("total = %d, want it filled in", got.TotalTokens)
	}
	sum := got.Add(got)
	if sum.InputTokens != 200 || sum.TotalTokens != 220 {
		t.Errorf("sum = %+v", sum)
	}
}

func TestBackoffGrowsAndIsBounded(t *testing.T) {
	first, second := Backoff(0), Backoff(1)
	if second <= first {
		t.Errorf("backoff did not grow: %s then %s", first, second)
	}
	if late := Backoff(20); late > DefaultMaxRetryDelay*10 {
		t.Errorf("backoff reached %s, which is a sleep nobody asked for", late)
	}
}
