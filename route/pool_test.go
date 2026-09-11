package route

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tamnd/llm"
)

// clock is a fake that only moves when a test says so. Cooldowns here are
// measured in half hours and six hour bans, and a test that waited them out
// would not be a test.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

func newClock() *clock {
	return &clock{now: time.Date(2026, 9, 11, 12, 0, 0, 0, time.UTC)}
}

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *clock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// checker answers for a route without a server. The func is per route name so
// a test can retire one host in the middle of a run.
type checker struct {
	mu    sync.Mutex
	reply map[string]Health
	calls map[string]int
	now   func() time.Time
}

func newChecker(now func() time.Time) *checker {
	return &checker{reply: map[string]Health{}, calls: map[string]int{}, now: now}
}

func (c *checker) set(name string, state State, detail string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.reply[name] = Health{Route: name, State: state, Detail: detail}
}

func (c *checker) Probe(_ context.Context, value Route) Health {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls[value.Name]++
	health, ok := c.reply[value.Name]
	if !ok {
		health = Health{Route: value.Name, State: StateLive, Detail: "ok"}
	}
	health.Route = value.Name
	health.Model = value.Model
	health.CheckedAt = c.now()
	return health
}

func (c *checker) count(name string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls[name]
}

// answer is a transport that returns whatever it is told to.
type answer struct {
	text string
	err  error
}

func (a answer) Complete(context.Context, llm.Request) (llm.Response, error) {
	if a.err != nil {
		return llm.Response{}, a.err
	}
	return llm.Response{Text: a.text}, nil
}

func testPool(t *testing.T, routes ...Route) (*Pool, *clock, *checker) {
	t.Helper()
	c := newClock()
	probe := newChecker(c.Now)
	pool := NewPool(Registry{Routes: routes})
	pool.Now = c.Now
	pool.Prober = probe
	pool.Build = func(r Route, _ time.Duration, _ int) (llm.Completer, error) {
		return answer{text: "from " + r.Name}, nil
	}
	return pool, c, probe
}

func TestPickTakesTheCheapestLiveRoute(t *testing.T) {
	pool, _, _ := testPool(t, gateway("cheap", 1), gateway("dear", 50))
	picked, client, release, err := pool.Pick(context.Background())
	if err != nil {
		t.Fatalf("Pick: %v", err)
	}
	defer release()
	if picked.Name != "cheap" {
		t.Errorf("picked %q, want the cheapest", picked.Name)
	}
	response, err := client.Complete(context.Background(), llm.Request{})
	if err != nil || response.Text != "from cheap" {
		t.Errorf("the client belongs to another route: %q %v", response.Text, err)
	}
}

// A probe result is trusted for HealthTTL and then asked again. Reprobing
// every call would cost more than the calls.
func TestPickReusesAFreshProbe(t *testing.T) {
	pool, c, probe := testPool(t, gateway("one", 1))
	for range 3 {
		_, _, release, err := pool.Pick(context.Background())
		if err != nil {
			t.Fatalf("Pick: %v", err)
		}
		release()
	}
	if got := probe.count("one"); got != 1 {
		t.Errorf("probed %d times inside the TTL, want 1", got)
	}
	c.advance(HealthTTL + time.Minute)
	_, _, release, err := pool.Pick(context.Background())
	if err != nil {
		t.Fatalf("Pick: %v", err)
	}
	release()
	if got := probe.count("one"); got != 2 {
		t.Errorf("probed %d times, want a reprobe once the record went stale", got)
	}
}

// Every route cold: the error must say when the first one comes back, because
// somebody at a terminal wants to know how long rather than that everything
// is down.
func TestPickWhenEveryRouteIsCold(t *testing.T) {
	pool, c, _ := testPool(t, gateway("a", 1), gateway("b", 2))
	pool.Fail("a", errors.New("chat completions returned 429: out of turns"))
	pool.Fail("b", errors.New("dial tcp 127.0.0.1:1: connect: connection refused"))

	_, _, _, err := pool.Pick(context.Background())
	if err == nil {
		t.Fatal("a cold pool handed out a route")
	}
	if !strings.Contains(err.Error(), "first to return") {
		t.Errorf("error = %v, want the time the first route comes back", err)
	}
	when, name := pool.EarliestReset()
	// The transport failure gets the one minute floor; the quota gets half an
	// hour. So the unreachable route is the one to wait for.
	if name != "b" {
		t.Errorf("earliest is %q, want the one with the shortest cooldown", name)
	}
	if got := when.Sub(c.Now()); got != MinCooldown {
		t.Errorf("earliest reset in %s, want %s", got, MinCooldown)
	}

	c.advance(MinCooldown + time.Second)
	picked, _, release, err := pool.Pick(context.Background())
	if err != nil {
		t.Fatalf("Pick after the cooldown: %v", err)
	}
	release()
	if picked.Name != "b" {
		t.Errorf("picked %q, want the route whose cooldown ran out", picked.Name)
	}
}

// A route that reports a model it does not serve is gone for the life of the
// process. No amount of waiting brings it back; the fix is an edit to the
// route file.
func TestAGoneRouteIsRetiredForGood(t *testing.T) {
	pool, c, _ := testPool(t, gateway("gone", 1), gateway("fine", 2))
	pool.Fail("gone", errors.New("404: No endpoints found that support image input"))

	c.advance(48 * time.Hour)
	picked, _, release, err := pool.Pick(context.Background())
	if err != nil {
		t.Fatalf("Pick: %v", err)
	}
	release()
	if picked.Name != "gone" {
		return
	}
	t.Error("a retired route came back after a long wait")
}

// A route that retires mid-run must not take the run with it: the next call
// goes somewhere else and the caller is told.
func TestRouteRetiresMidRun(t *testing.T) {
	pool, _, probe := testPool(t, gateway("first", 1), gateway("second", 2))
	var log []string
	pool.Logf = func(format string, args ...any) { log = append(log, fmt.Sprintf(format, args...)) }

	picked, _, release, err := pool.Pick(context.Background())
	if err != nil || picked.Name != "first" {
		t.Fatalf("first pick: %q %v", picked.Name, err)
	}
	release()

	// The host stops serving the model between one call and the next.
	probe.set("first", StateGone, "model is not served here")
	pool.Fail("first", errors.New("unknown model: test-model"))

	picked, _, release, err = pool.Pick(context.Background())
	if err != nil {
		t.Fatalf("second pick: %v", err)
	}
	release()
	if picked.Name != "second" {
		t.Errorf("picked %q, want the failover", picked.Name)
	}
	if len(log) == 0 || !strings.Contains(log[0], "first") {
		t.Errorf("the failover was not reported: %v", log)
	}
}

// More callers than lanes must wait rather than overload a host or fail. A
// route with one lane holds one call at a time.
func TestLaneLimitHolds(t *testing.T) {
	one := gateway("one", 1)
	one.Concurrency = 1
	pool, _, _ := testPool(t, one)

	_, _, release, err := pool.Pick(context.Background())
	if err != nil {
		t.Fatalf("Pick: %v", err)
	}
	if pool.Lanes() != 1 {
		t.Errorf("lanes = %d, want 1", pool.Lanes())
	}

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if _, _, _, err := pool.Pick(ctx); err == nil {
		t.Fatal("a second call took a lane that was already held")
	} else if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("error = %v, want the wait to have been interrupted", err)
	}

	// Releasing gives the lane back.
	release()
	_, _, release2, err := pool.Pick(context.Background())
	if err != nil {
		t.Fatalf("Pick after release: %v", err)
	}
	release2()
}

// A probe that times out is a failure of the route, not of the run: the pool
// cools it and moves on.
func TestProbeFailureMovesOn(t *testing.T) {
	pool, _, probe := testPool(t, gateway("slow", 1), gateway("quick", 2))
	probe.set("slow", StateUnreachable, "context deadline exceeded")

	picked, _, release, err := pool.Pick(context.Background())
	if err != nil {
		t.Fatalf("Pick: %v", err)
	}
	release()
	if picked.Name != "quick" {
		t.Errorf("picked %q, want the route that answered", picked.Name)
	}
}

func TestCooldownsByCause(t *testing.T) {
	for _, c := range []struct {
		name string
		err  error
		want time.Duration
	}{
		// A rejected credential does not fix itself in a minute: somebody has
		// to log in.
		{"unauthorized", errors.New("returned 401 Unauthorized"), UnauthorizedCooldown},
		{"quota with nothing said", errors.New("returned 429 Too Many Requests"), QuotaCooldown},
		{"transport", errors.New("dial tcp: connection refused"), MinCooldown},
	} {
		t.Run(c.name, func(t *testing.T) {
			pool, clk, _ := testPool(t, gateway("one", 1))
			pool.Fail("one", c.err)
			when, _ := pool.EarliestReset()
			if got := when.Sub(clk.Now()); got != c.want {
				t.Errorf("cooldown = %s, want %s", got, c.want)
			}
		})
	}
}

// Repeated transport failures back off, and the backoff is bounded: a route
// that has failed all morning is retried every half hour, not every eight.
func TestTransportCooldownBacksOffAndIsBounded(t *testing.T) {
	pool, clk, _ := testPool(t, gateway("one", 1))
	err := errors.New("dial tcp: connection refused")
	var last time.Duration
	for i := range 12 {
		pool.Fail("one", err)
		when, _ := pool.EarliestReset()
		got := when.Sub(clk.Now())
		if i > 0 && got < last {
			t.Errorf("cooldown shrank from %s to %s", last, got)
		}
		if got > MaxCooldown {
			t.Fatalf("cooldown reached %s, past the %s ceiling", got, MaxCooldown)
		}
		last = got
	}
	if last != MaxCooldown {
		t.Errorf("cooldown settled at %s, want the ceiling %s", last, MaxCooldown)
	}
}

// A quota that names when it ends is waited out exactly, and one naming an
// instant already past is a stale window: try again shortly rather than sit
// out half an hour.
func TestQuotaCooldownFollowsTheResetInstant(t *testing.T) {
	pool, clk, probe := testPool(t, gateway("one", 1))
	probe.set("one", StateQuota, "out of turns")

	resets := clk.Now().Add(7 * time.Minute)
	pool.Fail("one", fmt.Errorf("returned 429: out of turns, resets_at: %d", resets.Unix()))
	when, _ := pool.EarliestReset()
	if got := when.Sub(clk.Now()).Round(time.Second); got != 7*time.Minute {
		t.Errorf("cooldown = %s, want the seven minutes the host named", got)
	}
}

func TestSucceedClearsTheRecord(t *testing.T) {
	pool, _, _ := testPool(t, gateway("one", 1))
	pool.Fail("one", errors.New("dial tcp: connection refused"))
	pool.Succeed("one")
	for _, health := range pool.Health() {
		if health.Route == "one" && health.State != StateLive {
			t.Errorf("state = %q after a success", health.State)
		}
	}
}

// A reader answers no questions, so it never enters the pool, and a pool with
// nothing in it is a configuration problem phrased as one.
func TestNewPoolSkipsReadersAndDisabled(t *testing.T) {
	reader := Route{Name: "r", Kind: KindReader, Host: "box", Reader: "local-ocr", Model: "m"}
	off := gateway("off", 1)
	off.Disabled = true
	pool := NewPool(Registry{Routes: []Route{reader, off}})
	if !pool.Empty() {
		t.Errorf("pool holds %v", pool.Routes())
	}
	_, _, _, err := pool.Pick(context.Background())
	if err == nil || !strings.Contains(err.Error(), "no routes configured") {
		t.Errorf("error = %v, want the one that says what to do next", err)
	}
}

// Vision is declared in the route file, not discovered. Finding out costs a
// call and a free gateway answers an image with a refusal, so a run with
// pages to read should know at the start that there is somewhere to send
// them.
func TestVisionPoolTakesOnlyTheRoutesThatSee(t *testing.T) {
	sees := gateway("sees", 2)
	sees.Vision = true
	pool := NewVisionPool(Registry{Routes: []Route{gateway("blind", 1), sees}})
	routes := pool.Routes()
	if len(routes) != 1 || routes[0].Name != "sees" {
		t.Errorf("vision pool = %v", routes)
	}
}

// An exec route has no HTTP transport. Without a builder the pool must say so
// plainly rather than fail at the first call with something obscure.
func TestExecRouteNeedsABuilder(t *testing.T) {
	pool := NewPool(Registry{Routes: []Route{{Name: "cli", Kind: KindExec, Command: "codex", Model: "m"}}})
	pool.Now = newClock().Now
	pool.Prober = newChecker(pool.Now)
	_, _, _, err := pool.Pick(context.Background())
	if err == nil || !strings.Contains(err.Error(), "builder") {
		t.Errorf("error = %v, want it to name the missing builder", err)
	}
}

func TestTableIsOneLinePerRoute(t *testing.T) {
	rows := []Health{
		{Route: "a", State: StateLive, Detail: "ok", Model: "m", Transport: "pool"},
		{Route: "b", State: StateQuota, Detail: "out of turns", Model: "m",
			ResetsAt: time.Date(2026, 9, 11, 18, 0, 0, 0, time.UTC)},
		// A route answering on a lesser model than the one asked for is the
		// whole of the news, so the board shows both.
		{Route: "c", State: StateLive, Detail: "ok", Model: "big", Answered: "big-mini"},
	}
	table := Table(rows)
	if lines := strings.Count(strings.TrimSpace(table), "\n"); lines != len(rows) {
		t.Errorf("table has %d lines after the header, want %d:\n%s", lines, len(rows), table)
	}
	if !strings.Contains(table, "big -> big-mini") {
		t.Errorf("the downgrade is not on the board:\n%s", table)
	}
	if !strings.Contains(table, "resets 2026-09-11 18:00 UTC") {
		t.Errorf("the reset time is not on the board:\n%s", table)
	}
}
