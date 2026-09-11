package llm

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultMaxRetryDelay is how long a call will sit out a provider's requested
// wait before giving up and letting the router try somewhere else.
//
// A provider answering 429 with retry-after: 53931 is telling you it is done
// for the day, and the answer to that is the next route, not a fifteen hour
// sleep inside one call. Both lanes of a translation run went to sleep that
// way once and the log's last line was ninety four minutes stale before
// anybody noticed.
const DefaultMaxRetryDelay = time.Minute

// Client speaks POST /v1/chat/completions, streaming.
//
// Streaming is not decoration. A page takes minutes to come back through a
// browser session, and a non-streaming request holds one connection open with
// nothing on it for that whole time, which is exactly the shape an idle
// timeout somewhere in the middle kills. The stream also arrives with usage on
// the last chunk.
type Client struct {
	URL        string
	APIKey     string
	HTTPClient *http.Client
	// MaxRetries is retries inside one call, for a failure that looks like it
	// will pass. Failing over to another host is the router's job, not this.
	MaxRetries    int
	MaxRetryDelay time.Duration
	UserAgent     string
	// CacheKeyPrefix namespaces this caller's prompt cache on a shared proxy.
	// Empty means the configured app name.
	CacheKeyPrefix string
	// Sleep is the wait between attempts, replaceable so a test does not.
	Sleep func(context.Context, time.Duration) error
}

type chatRequest struct {
	Model          string        `json:"model"`
	Messages       []chatMessage `json:"messages"`
	Stream         bool          `json:"stream"`
	StreamOptions  streamOptions `json:"stream_options"`
	PromptCacheKey string        `json:"prompt_cache_key,omitempty"`
}

// chatMessage carries either a plain string content or a list of parts. The
// OpenAI wire allows both and a proxy fronting a browser session only
// understands the first, so a text ask must serialise as a bare string and not
// as a one-element list of text parts.
type chatMessage struct {
	Role  string
	Text  string
	Parts []contentPart
}

func (m chatMessage) MarshalJSON() ([]byte, error) {
	if len(m.Parts) == 0 {
		return json.Marshal(struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{m.Role, m.Text})
	}
	return json.Marshal(struct {
		Role    string        `json:"role"`
		Content []contentPart `json:"content"`
	}{m.Role, m.Parts})
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatChunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
		CacheReadTokens  int `json:"cache_read_input_tokens"`
		PromptDetails    struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Complete puts one question and returns the answer, retrying inside the call
// only for a failure that looks like it will pass.
func (c *Client) Complete(ctx context.Context, request Request) (Response, error) {
	if strings.TrimSpace(c.URL) == "" {
		return Response{}, errors.New("chat completions URL is empty")
	}
	if strings.TrimSpace(request.Model) == "" {
		return Response{}, errors.New("model is empty")
	}
	if strings.TrimSpace(request.Input) == "" && len(request.Images) == 0 {
		return Response{}, errors.New("input is empty")
	}
	messages := make([]chatMessage, 0, 2)
	if strings.TrimSpace(request.Instructions) != "" {
		messages = append(messages, chatMessage{Role: "system", Text: request.Instructions})
	}
	user, err := userMessage(request)
	if err != nil {
		return Response{}, err
	}
	messages = append(messages, user)
	payload, err := json.Marshal(chatRequest{
		Model:          request.Model,
		Messages:       messages,
		Stream:         true,
		StreamOptions:  streamOptions{IncludeUsage: true},
		PromptCacheKey: c.cacheKey(request.Instructions),
	})
	if err != nil {
		return Response{}, fmt.Errorf("encode chat request: %w", err)
	}

	maxRetryDelay := c.MaxRetryDelay
	if maxRetryDelay == 0 {
		maxRetryDelay = DefaultMaxRetryDelay
	}
	started := time.Now()
	var last error
	for attempt := 0; attempt <= max(0, c.MaxRetries); attempt++ {
		response, retryAfter, retry, err := c.do(ctx, payload)
		if err == nil {
			response.Elapsed = time.Since(started)
			return response, nil
		}
		last = err
		if !retry || attempt == max(0, c.MaxRetries) {
			break
		}
		// A provider that names a wait longer than we are willing to sit out is
		// telling us to go elsewhere. Give the router the chance.
		if maxRetryDelay > 0 && retryAfter > maxRetryDelay {
			break
		}
		if retryAfter <= 0 {
			retryAfter = Backoff(attempt)
		}
		if err := c.sleep(ctx, retryAfter); err != nil {
			return Response{}, err
		}
	}
	return Response{}, last
}

func userMessage(request Request) (chatMessage, error) {
	if len(request.Images) == 0 {
		return chatMessage{Role: "user", Text: request.Input}, nil
	}
	parts := make([]contentPart, 0, len(request.Images)+1)
	if strings.TrimSpace(request.Input) != "" {
		parts = append(parts, contentPart{Type: "text", Text: request.Input})
	}
	for i, image := range request.Images {
		if len(image.Data) == 0 {
			return chatMessage{}, fmt.Errorf("image %d has no bytes", i)
		}
		mediaType := image.MediaType
		if mediaType == "" {
			mediaType = "image/png"
		}
		parts = append(parts, contentPart{
			Type: "image_url",
			ImageURL: &imageURL{
				URL:    "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(image.Data),
				Detail: image.Detail,
			},
		})
	}
	return chatMessage{Role: "user", Parts: parts}, nil
}

func (c *Client) do(ctx context.Context, payload []byte) (Response, time.Duration, bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL, bytes.NewReader(payload))
	if err != nil {
		return Response{}, 0, false, fmt.Errorf("create chat request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	agent := c.UserAgent
	if agent == "" {
		agent = UserAgent()
	}
	req.Header.Set("User-Agent", agent)
	client := c.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Minute}
	}
	resp, err := client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return Response{}, 0, false, ctx.Err()
		}
		return Response{}, 0, true, fmt.Errorf("call chat completions: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		signal := Classify(resp.StatusCode, resp.Header, raw)
		// Gone is never worth another attempt on this route: the model is not
		// served here and will not be a second later.
		retry := signal.State != StateGone &&
			(resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500)
		after := signal.RetryAfter
		return Response{}, after, retry, fmt.Errorf("chat completions returned %s: %s%s",
			resp.Status, signal.Detail, retryAfterSuffix(after))
	}
	if strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		response, err := parseStream(resp.Body)
		if err != nil {
			return Response{}, 0, true, err
		}
		return response, 0, false, nil
	}
	// Not every compatible server honours stream: true. Reading the plain JSON
	// body is cheaper than arguing with it.
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return Response{}, 0, true, fmt.Errorf("read chat response: %w", err)
	}
	response, err := parseJSON(raw)
	if err != nil {
		return Response{}, 0, false, err
	}
	return response, 0, false, nil
}

func parseStream(reader io.Reader) (Response, error) {
	scanner := bufio.NewScanner(reader)
	// A transcribed page is tens of kilobytes and arrives in small deltas, but
	// a server that batches can put the lot in one data: line, so the ceiling
	// is generous.
	scanner.Buffer(make([]byte, 64*1024), 32*1024*1024)
	var text strings.Builder
	var response Response
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		data := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		var chunk chatChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			return Response{}, fmt.Errorf("decode chat stream: %w", err)
		}
		if chunk.Error != nil {
			return Response{}, fmt.Errorf("chat stream error: %s", chunk.Error.Message)
		}
		if response.ID == "" {
			response.ID = chunk.ID
		}
		if chunk.Model != "" {
			response.Model = chunk.Model
		}
		for _, choice := range chunk.Choices {
			text.WriteString(choice.Delta.Content)
			text.WriteString(choice.Message.Content)
		}
		setUsage(&response, chunk)
	}
	if err := scanner.Err(); err != nil {
		return Response{}, fmt.Errorf("read chat stream: %w", err)
	}
	response.Text = strings.TrimSpace(text.String())
	if response.Text == "" {
		return Response{}, errors.New("chat completions returned no text")
	}
	return response, nil
}

func parseJSON(raw []byte) (Response, error) {
	var chunk chatChunk
	if err := json.Unmarshal(raw, &chunk); err != nil {
		return Response{}, fmt.Errorf("decode chat response: %w", err)
	}
	if chunk.Error != nil {
		return Response{}, fmt.Errorf("chat completions error: %s", chunk.Error.Message)
	}
	response := Response{ID: chunk.ID, Model: chunk.Model}
	var text strings.Builder
	for _, choice := range chunk.Choices {
		text.WriteString(choice.Message.Content)
		text.WriteString(choice.Delta.Content)
	}
	setUsage(&response, chunk)
	response.Text = strings.TrimSpace(text.String())
	if response.Text == "" {
		return Response{}, errors.New("chat completions returned no text")
	}
	return response, nil
}

func setUsage(response *Response, chunk chatChunk) {
	if chunk.Usage == nil {
		return
	}
	cached := chunk.Usage.PromptDetails.CachedTokens
	if cached == 0 {
		cached = chunk.Usage.CacheReadTokens
	}
	response.Usage = Usage{
		InputTokens:       chunk.Usage.PromptTokens,
		CachedInputTokens: cached,
		OutputTokens:      chunk.Usage.CompletionTokens,
		ReasoningTokens:   chunk.Usage.CompletionDetails.ReasoningTokens,
		TotalTokens:       chunk.Usage.TotalTokens,
	}.Normalized()
}

func (c *Client) sleep(ctx context.Context, duration time.Duration) error {
	if c.Sleep != nil {
		return c.Sleep(ctx, duration)
	}
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// cacheKey groups calls that share a system prompt. A page-reading prompt is a
// couple of thousand tokens repeated on every page of a long document, so it
// is worth naming. The prefix is per application so two projects sharing one
// proxy do not collide in its cache.
func (c *Client) cacheKey(instructions string) string {
	if instructions == "" {
		return ""
	}
	prefix := c.CacheKeyPrefix
	if prefix == "" {
		prefix = App()
	}
	sum := sha256.Sum256([]byte(instructions))
	return prefix + "-" + hex.EncodeToString(sum[:8])
}
