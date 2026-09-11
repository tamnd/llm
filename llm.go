// Package llm is one model call: a request, a response, token accounting, and
// the classification of a failure into something worth retrying and something
// worth moving on from.
//
// It is deliberately small. The interesting problems in talking to a model are
// not in the wire, they are in deciding which of several hosts is answering
// today (package route), in surviving a run that takes longer than a process
// lives (package queue), and in knowing exactly what was asked (package
// prompt). This package is what those three agree on.
//
// The wire is the OpenAI chat completions format, streaming, because every
// transport this was written for speaks it: a proxy fronting browser sessions,
// a free public gateway, and a local vLLM. The one transport that does not is
// a CLI on this machine, and package exec makes that look the same.
package llm

import (
	"context"
	"fmt"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Completer is one model call. It is an interface so a test can answer without
// a server and so the router can wrap it.
type Completer interface {
	Complete(ctx context.Context, request Request) (Response, error)
}

// Request is one call to a model.
type Request struct {
	Model string
	// Instructions is the system message. It is the part that repeats across a
	// run, so it also becomes the prompt cache key.
	Instructions string
	Input        string
	// Images are sent alongside Input as a multi-part user message. Empty for
	// a text ask, which is nearly all of them.
	//
	// The field exists because whether a route can take an image is a property
	// of the route and not of this package: a proxy fronting browser sessions
	// takes none, a free gateway answers "no endpoints found that support
	// image input", and a local vLLM takes them happily. A caller with a route
	// that can should not have to leave the library to use it.
	Images []Image
}

// Image is one picture in a request.
type Image struct {
	MediaType string // "image/png", "image/jpeg"
	Data      []byte // raw bytes; the client base64s them
	Detail    string // "", "low", "high"
}

// Response is what came back, plus enough about where it came from to write an
// honest report.
type Response struct {
	ID    string `json:"id"`
	Model string `json:"model"`
	// Route names the host that served the call. A document assembled from two
	// hosts has to be able to say which one answered each page.
	Route   string        `json:"route,omitempty"`
	Text    string        `json:"text"`
	Usage   Usage         `json:"usage"`
	Elapsed time.Duration `json:"elapsed"`
}

// Usage is the token accounting the provider returned. InputTokens counts
// cached reads and cache writes as well. ReasoningTokens is a part of
// OutputTokens and must not be added to it.
//
// A subscription or a free gateway bills nobody, so these numbers buy no cost
// estimate. They are here because they are the only measure of how much work
// an ask took, and an answer that comes back with a tenth of the usual output
// tokens is an answer that got truncated.
type Usage struct {
	InputTokens       int `json:"input_tokens"`
	CachedInputTokens int `json:"cached_input_tokens"`
	OutputTokens      int `json:"output_tokens"`
	ReasoningTokens   int `json:"reasoning_tokens"`
	TotalTokens       int `json:"total_tokens"`
}

// Normalized fills in totals a compatible proxy left out and stops malformed
// cache detail from producing a negative count of uncached input.
func (u Usage) Normalized() Usage {
	u.InputTokens = max(0, u.InputTokens)
	u.OutputTokens = max(0, u.OutputTokens)
	u.CachedInputTokens = min(max(0, u.CachedInputTokens), u.InputTokens)
	u.ReasoningTokens = min(max(0, u.ReasoningTokens), u.OutputTokens)
	if u.TotalTokens <= 0 {
		u.TotalTokens = u.InputTokens + u.OutputTokens
	}
	return u
}

// Add sums two accountings, for a report that covers a whole run.
func (u Usage) Add(other Usage) Usage {
	u.InputTokens += other.InputTokens
	u.CachedInputTokens += other.CachedInputTokens
	u.OutputTokens += other.OutputTokens
	u.ReasoningTokens += other.ReasoningTokens
	u.TotalTokens += other.TotalTokens
	return u
}

// ParseRetryAfter reads the header, which a provider may write as a number of
// seconds or as an HTTP date.
func ParseRetryAfter(value string) time.Duration {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}
	if seconds, err := strconv.Atoi(value); err == nil && seconds >= 0 {
		return time.Duration(seconds) * time.Second
	}
	if when, err := http.ParseTime(value); err == nil {
		return max(0, time.Until(when))
	}
	return 0
}

func retryAfterSuffix(delay time.Duration) string {
	if delay <= 0 {
		return ""
	}
	return fmt.Sprintf(" (retry after %s)", delay)
}

// Backoff doubles from a second to thirty, with jitter so a fleet of goroutines
// that all failed on the same dead tunnel do not all come back at once.
func Backoff(attempt int) time.Duration {
	base := min(30*time.Second, time.Second<<min(attempt, 5))
	return base + time.Duration(rand.IntN(500))*time.Millisecond
}

// Condense makes a provider message fit one log line and one table cell. A
// gateway that is down answers with an HTML error page, and a detail field
// holding twenty lines of markup turns a status table into something nobody
// can read.
func Condense(text string) string {
	text = strings.Join(strings.Fields(text), " ")
	if len(text) > 200 {
		text = text[:200] + "..."
	}
	return text
}
