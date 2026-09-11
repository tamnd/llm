package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// The recorded bodies here are shaped like the ones the real transports send
// and carry nothing from any document. Where a case came from a real failure
// the comment says which fault it stands for.

const recordedStream = `data: {"id":"chatcmpl-1","model":"test-model","choices":[{"delta":{"content":"one "}}]}

data: {"id":"chatcmpl-1","model":"test-model","choices":[{"delta":{"content":"two"}}]}

data: {"id":"chatcmpl-1","model":"test-model","choices":[],"usage":{"prompt_tokens":120,"completion_tokens":8,"total_tokens":128,"prompt_tokens_details":{"cached_tokens":100}}}

data: [DONE]
`

func serve(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return &Client{URL: server.URL, HTTPClient: server.Client(),
		Sleep: func(context.Context, time.Duration) error { return nil }}
}

func ask(t *testing.T, client *Client) (Response, error) {
	t.Helper()
	return client.Complete(context.Background(), Request{
		Model: "test-model", Instructions: "Answer in one word.", Input: "question"})
}

func TestCompleteReadsStream(t *testing.T) {
	client := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(recordedStream))
	})
	response, err := ask(t, client)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.Text != "one two" {
		t.Errorf("text = %q, want %q", response.Text, "one two")
	}
	if response.ID != "chatcmpl-1" || response.Model != "test-model" {
		t.Errorf("id/model = %q/%q", response.ID, response.Model)
	}
	want := Usage{InputTokens: 120, CachedInputTokens: 100, OutputTokens: 8, TotalTokens: 128}
	if response.Usage != want {
		t.Errorf("usage = %+v, want %+v", response.Usage, want)
	}
	if response.Elapsed <= 0 {
		t.Error("elapsed was not measured")
	}
}

// A stream cut off partway through a chunk must be an error and not a short
// answer. A truncated page that came back as a successful half page would be
// written to disk and never looked at again.
func TestCompleteStreamStopsMidChunk(t *testing.T) {
	client := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"one "` + "\n"))
	})
	_, err := ask(t, client)
	if err == nil {
		t.Fatal("a half written chunk was accepted as an answer")
	}
	if !strings.Contains(err.Error(), "decode chat stream") {
		t.Errorf("error = %v, want a decode failure", err)
	}
}

// A provider answering 429 with fifteen hours on it is telling us to go
// elsewhere. Sitting out the wait inside one call is what put both lanes of a
// run to sleep with a log line ninety four minutes stale.
func TestCompleteDoesNotSitOutALongRetryAfter(t *testing.T) {
	var calls int
	slept := time.Duration(-1)
	client := serve(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Retry-After", "53931")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate_limit_exceeded"}}`))
	})
	client.MaxRetries = 3
	client.Sleep = func(_ context.Context, d time.Duration) error { slept = d; return nil }

	_, err := ask(t, client)
	if err == nil {
		t.Fatal("a 429 was reported as success")
	}
	if calls != 1 {
		t.Errorf("called the provider %d times, want 1", calls)
	}
	if slept >= 0 {
		t.Errorf("slept %s inside the call, want no sleep at all", slept)
	}
	if !strings.Contains(err.Error(), "429") {
		t.Errorf("error = %v, want the status in it", err)
	}
}

// A short Retry-After is worth waiting out, and the wait is the one the
// provider asked for rather than the backoff.
func TestCompleteHonoursAShortRetryAfter(t *testing.T) {
	var calls int
	var slept []time.Duration
	client := serve(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Retry-After", "2")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"slow down"}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(recordedStream))
	})
	client.MaxRetries = 2
	client.Sleep = func(_ context.Context, d time.Duration) error { slept = append(slept, d); return nil }

	response, err := ask(t, client)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.Text != "one two" {
		t.Errorf("text = %q", response.Text)
	}
	if len(slept) != 1 || slept[0] != 2*time.Second {
		t.Errorf("slept %v, want one wait of 2s", slept)
	}
}

// Not every compatible server honours stream: true. A plain JSON body is an
// answer, not a fault.
func TestCompleteReadsPlainJSON(t *testing.T) {
	client := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"cmpl-2","model":"test-model",
			"choices":[{"message":{"content":" plain "}}],
			"usage":{"prompt_tokens":10,"completion_tokens":2}}`))
	})
	response, err := ask(t, client)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.Text != "plain" {
		t.Errorf("text = %q, want %q", response.Text, "plain")
	}
	if response.Usage.TotalTokens != 12 {
		t.Errorf("total tokens = %d, want the sum filled in", response.Usage.TotalTokens)
	}
}

// A dead gateway answers with a proxy's HTML rather than with JSON. The error
// must carry the page's words condensed, not a blank detail and not the whole
// page.
func TestCompleteReportsAnHTMLErrorPage(t *testing.T) {
	page := "<html>\n<head><title>502 Bad Gateway</title></head>\n<body>\n<center><h1>502 Bad Gateway</h1></center>\n</body>\n</html>\n"
	client := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(page))
	})
	_, err := ask(t, client)
	if err == nil {
		t.Fatal("an HTML error page was accepted as an answer")
	}
	if !strings.Contains(err.Error(), "502 Bad Gateway") {
		t.Errorf("error = %v, want the page's own words", err)
	}
	if strings.Contains(err.Error(), "\n") {
		t.Errorf("error spans lines, want one condensed line: %q", err)
	}
}

// A text ask must go out with a bare string content. A proxy fronting a
// browser session understands that and nothing else, and a one-element list of
// text parts reads to it as an empty prompt.
func TestTextAskSerialisesAsAString(t *testing.T) {
	var body map[string]any
	client := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(recordedStream))
	})
	if _, err := ask(t, client); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	messages, _ := body["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("sent %d messages, want a system and a user", len(messages))
	}
	for _, message := range messages {
		content := message.(map[string]any)["content"]
		if _, ok := content.(string); !ok {
			t.Errorf("content is %T, want a string", content)
		}
	}
	if body["stream"] != true {
		t.Error("stream was not requested")
	}
}

// An image ask must go out as parts, with the bytes inline as a data URL.
func TestImageAskSerialisesAsParts(t *testing.T) {
	var body map[string]any
	client := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(recordedStream))
	})
	_, err := client.Complete(context.Background(), Request{
		Model: "test-model", Input: "read this",
		Images: []Image{{MediaType: "image/png", Data: []byte{1, 2, 3}, Detail: "high"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	messages, _ := body["messages"].([]any)
	if len(messages) != 1 {
		t.Fatalf("sent %d messages, want one", len(messages))
	}
	parts, ok := messages[0].(map[string]any)["content"].([]any)
	if !ok {
		t.Fatalf("content is not a list of parts: %#v", messages[0])
	}
	if len(parts) != 2 {
		t.Fatalf("sent %d parts, want the text and the image", len(parts))
	}
	image := parts[1].(map[string]any)["image_url"].(map[string]any)
	if url, _ := image["url"].(string); !strings.HasPrefix(url, "data:image/png;base64,") {
		t.Errorf("image url = %q, want an inline data url", url)
	}
}

// An image sent with no bytes is a caller bug and must fail before the call
// rather than as an unreadable refusal from the far end.
func TestImageWithNoBytesIsRefused(t *testing.T) {
	client := serve(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("the call went out")
	})
	_, err := client.Complete(context.Background(), Request{
		Model: "test-model", Input: "read this", Images: []Image{{MediaType: "image/png"}}})
	if err == nil {
		t.Fatal("an empty image was sent")
	}
}

func TestCacheKeyIsNamespacedByApp(t *testing.T) {
	Configure(Config{App: "papers"})
	t.Cleanup(func() { Configure(Config{}) })

	client := &Client{}
	key := client.cacheKey("a system prompt")
	if !strings.HasPrefix(key, "papers-") {
		t.Errorf("cache key = %q, want the app name on the front", key)
	}
	if client.cacheKey("") != "" {
		t.Error("a call with no system prompt got a cache key")
	}
	client.CacheKeyPrefix = "other"
	if !strings.HasPrefix(client.cacheKey("a system prompt"), "other-") {
		t.Error("an explicit prefix was ignored")
	}
}

func TestCompleteRefusesAnEmptyAsk(t *testing.T) {
	for _, c := range []struct {
		name    string
		client  Client
		request Request
	}{
		{"no url", Client{}, Request{Model: "m", Input: "hello"}},
		{"no model", Client{URL: "http://x"}, Request{Input: "hello"}},
		{"no input", Client{URL: "http://x"}, Request{Model: "m"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := c.client.Complete(context.Background(), c.request); err == nil {
				t.Error("want an error before anything is sent")
			}
		})
	}
}
