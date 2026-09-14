package route

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/tamnd/llm"
)

// errNoBuilder is a route the pool cannot build a transport for, which is a
// caller that put an exec route in a pool without registering Build. It is a
// sentinel so Pick can tell a configuration mistake from a host having a bad
// afternoon.
var errNoBuilder = errors.New("the caller registered no builder for it, so it cannot be called")

// Cooldowns after a failure. They are not uniform because the causes are not:
// a rejected credential does not fix itself in a minute, and a daily limit
// does not clear in five.
const (
	MinCooldown          = time.Minute
	MaxCooldown          = 30 * time.Minute
	QuotaCooldown        = 30 * time.Minute
	UnauthorizedCooldown = 6 * time.Hour
	// HealthTTL is how long a probe result is trusted before a reprobe.
	HealthTTL = 10 * time.Minute
)

// entry is a route plus what the pool has learned about it.
type entry struct {
	route    Route
	health   Health
	coldTill time.Time
	failures int
	retired  bool // gone for the life of the process
	inflight int
}

// Checker is the health check the pool runs before a route's first use. It is
// an interface so a test can answer without a server and so a caller that
// already knows the fleet is up can skip it.
type Checker interface {
	Probe(ctx context.Context, value Route) Health
}

// Pool picks a route, notices when it dies, and moves to the next one.
//
// Failover is per model call, not per document and not per run. One page's
// retry may land on a different host from its first attempt, and the result
// records which one answered.
type Pool struct {
	Prober     Checker
	Timeout    time.Duration
	MaxRetries int
	// Build makes the transport for a route, and is how a caller puts an exec
	// route into the pool: this package cannot construct one without importing
	// llm/exec, and llm/exec imports this one for Route. Returning (nil, nil)
	// means "not mine", and the pool falls back to the HTTP client.
	//
	// It is a hook rather than an interface because there is exactly one thing
	// to decide and a caller who does not have an exec route never sets it.
	Build func(r Route, timeout time.Duration, maxRetries int) (llm.Completer, error)
	// Logf reports every failover, because a document assembled from two hosts
	// should never be a surprise after the fact.
	Logf func(string, ...any)
	Now  func() time.Time

	mu      sync.Mutex
	entries []*entry
}

// NewPool builds a pool from a registry, skipping disabled routes and readers.
func NewPool(registry Registry) *Pool {
	pool := &Pool{}
	for _, value := range registry.Enabled() {
		// A reader reads page images and answers nothing, so it never enters
		// the pool. See Kind.Answers.
		if !value.Answers() {
			continue
		}
		pool.entries = append(pool.entries, &entry{route: value})
	}
	return pool
}

// NewVisionPool builds a pool of the routes that will take an image. It is a
// separate constructor rather than a flag on Pick because a caller with pages
// to read wants to know at the start of a run that there is somewhere to send
// them, not an hour in.
func NewVisionPool(registry Registry) *Pool {
	pool := &Pool{}
	for _, value := range registry.Enabled() {
		if !value.Answers() || !value.Vision {
			continue
		}
		pool.entries = append(pool.entries, &entry{route: value})
	}
	return pool
}

// NewJobPool builds a pool of the routes that will do a named job, which is
// the routes that name it and the routes that name none. See Route.Jobs.
//
// Same reasoning as NewVisionPool: a caller whose fleet has nothing that
// will take this job should hear so before the run rather than after it,
// and Empty is what says so.
func NewJobPool(registry Registry, job string) *Pool {
	pool := &Pool{}
	for _, value := range registry.Enabled() {
		if !value.Answers() || !value.Does(job) {
			continue
		}
		pool.entries = append(pool.entries, &entry{route: value})
	}
	return pool
}

// Empty reports whether the pool holds no routes at all, which is a
// configuration problem and not a transport one.
func (p *Pool) Empty() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.entries) == 0
}

func (p *Pool) now() time.Time {
	if p.Now != nil {
		return p.Now()
	}
	return time.Now()
}

func (p *Pool) probe(ctx context.Context, value Route) Health {
	if p.Prober == nil {
		return Prober{}.Probe(ctx, value)
	}
	return p.Prober.Probe(ctx, value)
}

func (p *Pool) logf(format string, args ...any) {
	if p.Logf != nil {
		p.Logf(format, args...)
	}
}

// Lanes is the total number of calls the pool will carry at once, which is the
// sum over live routes of what each one will take.
func (p *Pool) Lanes() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	total := 0
	for _, value := range p.entries {
		if !value.retired {
			total += value.route.Lanes()
		}
	}
	return max(1, total)
}

// Pick returns the best route that is not cold and not already full, probing
// lazily. A route with no health record is probed before its first use, and a
// record older than HealthTTL is refreshed.
//
// The returned release must be called when the caller is done with the route,
// which is what frees its lane.
func (p *Pool) Pick(ctx context.Context) (Route, llm.Completer, func(), error) {
	if p.Empty() {
		return Route{}, nil, nil, ErrNoRoutes()
	}
	for {
		candidate, err := p.next(ctx)
		if err != nil {
			return Route{}, nil, nil, err
		}
		client, err := p.clientFor(candidate)
		if err != nil {
			p.release(candidate.Name)
			// A route with no transport is a configuration mistake, and no
			// amount of failing over fixes it. Cooling it down and trying the
			// next one would bury the one sentence that says what to do, under
			// a report that every route is cold.
			if errors.Is(err, errNoBuilder) {
				return Route{}, nil, nil, err
			}
			p.Fail(candidate.Name, err)
			continue
		}
		released := false
		return candidate, client, func() {
			if !released {
				released = true
				p.release(candidate.Name)
			}
		}, nil
	}
}

// clientFor builds the transport for a route. An exec route has no HTTP
// transport, so a caller that wants one in its pool registers a factory with
// Build; without one, the route reports the omission rather than failing at
// its first call with something obscure.
func (p *Pool) clientFor(candidate Route) (llm.Completer, error) {
	if p.Build != nil {
		if client, err := p.Build(candidate, p.Timeout, p.MaxRetries); client != nil || err != nil {
			return client, err
		}
	}
	if !candidate.Kind.OverHTTP() {
		return nil, fmt.Errorf("route %s is a %s route: %w", candidate.Name, candidate.Kind, errNoBuilder)
	}
	return candidate.Client(p.Timeout, p.MaxRetries)
}

// next takes a lane on the first route that has one going spare.
func (p *Pool) next(ctx context.Context) (Route, error) {
	for {
		p.mu.Lock()
		now := p.now()
		var chosen *entry
		full := false
		for _, value := range p.entries {
			if value.retired || now.Before(value.coldTill) {
				continue
			}
			if value.inflight >= value.route.Lanes() {
				full = true
				continue
			}
			chosen = value
			break
		}
		if chosen == nil {
			if full {
				// Every usable route is busy. This is a caller that has more
				// goroutines than the fleet has lanes, and the honest answer
				// is to wait rather than to fail or to overload a host.
				p.mu.Unlock()
				if err := wait(ctx, 500*time.Millisecond); err != nil {
					return Route{}, err
				}
				continue
			}
			err := p.coldError(now)
			p.mu.Unlock()
			return Route{}, err
		}
		chosen.inflight++
		fresh := chosen.health.State != "" && now.Sub(chosen.health.CheckedAt) < HealthTTL
		candidate := chosen.route
		p.mu.Unlock()

		if fresh {
			return candidate, nil
		}
		health := p.probe(ctx, candidate)
		p.mu.Lock()
		chosen.health = health
		if health.State.Usable() {
			chosen.failures = 0
			p.mu.Unlock()
			return candidate, nil
		}
		chosen.inflight--
		p.applyLocked(chosen, Signal{State: health.State, Detail: health.Detail, ResetsAt: health.ResetsAt})
		p.mu.Unlock()
		p.logf("%s is %s: %s", candidate.Name, health.State, health.Detail)
	}
}

func wait(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (p *Pool) release(name string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, value := range p.entries {
		if value.route.Name == name && value.inflight > 0 {
			value.inflight--
			return
		}
	}
}

// Fail records that a route failed a real call and takes it out for a while.
func (p *Pool) Fail(name string, err error) {
	if err == nil {
		return
	}
	signal := llm.ClassifyError(err)
	found := false
	p.mu.Lock()
	for _, value := range p.entries {
		if value.route.Name != name {
			continue
		}
		value.health = Health{
			Route: name, State: signal.State, Detail: signal.Detail,
			ResetsAt: signal.ResetsAt, Model: value.route.Model, CheckedAt: p.now(),
		}
		p.applyLocked(value, signal)
		found = true
		break
	}
	p.mu.Unlock()
	if found {
		// Outside the lock: the callback belongs to the caller and may do
		// anything, including come back into the pool.
		p.logf("%s is %s: %s", name, signal.State, signal.Detail)
	}
}

// Succeed clears the failure count after a route completes a call.
func (p *Pool) Succeed(name string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, value := range p.entries {
		if value.route.Name == name {
			value.failures = 0
			health := value.health
			health.Route = name
			health.State = StateLive
			health.Detail = "ok"
			health.Model = value.route.Model
			health.CheckedAt = p.now()
			value.health = health
			return
		}
	}
}

// applyLocked sets the cooldown for a failure. The caller holds the lock.
func (p *Pool) applyLocked(value *entry, signal Signal) {
	now := p.now()
	switch signal.State {
	case StateGone:
		// No amount of waiting brings a model back that the host does not
		// serve. The fix is an edit to the route file.
		value.retired = true
		value.coldTill = now.Add(100 * 365 * 24 * time.Hour)
	case StateQuota:
		// A host that says nothing gets the default window. One that names an
		// instant already past is reporting a stale window, and the honest
		// reading of that is try again shortly, not wait half an hour and not
		// go straight back in.
		cooldown := QuotaCooldown
		switch {
		case !signal.ResetsAt.IsZero() && signal.ResetsAt.After(now):
			cooldown = signal.ResetsAt.Sub(now)
		case !signal.ResetsAt.IsZero():
			cooldown = MinCooldown
		case signal.RetryAfter > 0:
			cooldown = signal.RetryAfter
		}
		value.coldTill = now.Add(max(cooldown, MinCooldown))
	case StateUnauthorized:
		value.coldTill = now.Add(UnauthorizedCooldown)
	default:
		value.failures++
		value.coldTill = now.Add(cooldownFor(value.failures))
	}
}

// cooldownFor doubles from the minimum to the maximum.
func cooldownFor(failures int) time.Duration {
	delay := MinCooldown
	for range max(0, failures-1) {
		delay *= 2
		if delay >= MaxCooldown {
			return MaxCooldown
		}
	}
	return delay
}

// coldError names the earliest reset, because somebody waiting at a terminal
// wants to know how long, not just that everything is down.
func (p *Pool) coldError(now time.Time) error {
	states := make([]string, 0, len(p.entries))
	for _, value := range p.entries {
		state := value.health.State
		if state == "" {
			state = StateUnknown
		}
		states = append(states, fmt.Sprintf("%s: %s", value.route.Name, state))
	}
	found := strings.Join(states, ", ")
	earliest, name := p.earliestLocked(now)
	if earliest.IsZero() {
		return fmt.Errorf("no routes available (%s)", found)
	}
	return fmt.Errorf("every route is cold (%s); %s is the first to return at %s, in %s",
		found, name, earliest.UTC().Format(time.RFC3339), earliest.Sub(now).Round(time.Second))
}

// EarliestReset is when the first cold route becomes usable again, so an
// unattended run can sleep instead of exiting.
func (p *Pool) EarliestReset() (time.Time, string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.earliestLocked(p.now())
}

func (p *Pool) earliestLocked(now time.Time) (time.Time, string) {
	var earliest time.Time
	name := ""
	for _, value := range p.entries {
		if value.retired {
			continue
		}
		if value.coldTill.Before(now) {
			return now, value.route.Name
		}
		if earliest.IsZero() || value.coldTill.Before(earliest) {
			earliest = value.coldTill
			name = value.route.Name
		}
	}
	return earliest, name
}

// Health returns what the pool currently believes about every route.
func (p *Pool) Health() []Health {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]Health, 0, len(p.entries))
	for _, value := range p.entries {
		health := value.health
		if health.Route == "" {
			health = Health{Route: value.route.Name, State: StateUnknown, Model: value.route.Model}
		}
		out = append(out, health)
	}
	return out
}

// Routes lists the routes the pool holds, in the order it will try them.
func (p *Pool) Routes() []Route {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]Route, 0, len(p.entries))
	for _, value := range p.entries {
		out = append(out, value.route)
	}
	return out
}

// ProbeAll checks every route the pool holds and returns the results in rank
// order. It is what a doctor command prints.
func (p *Pool) ProbeAll(ctx context.Context) []Health {
	p.mu.Lock()
	entries := append([]*entry(nil), p.entries...)
	p.mu.Unlock()

	results := make([]Health, len(entries))
	var group sync.WaitGroup
	for index, value := range entries {
		group.Go(func() { results[index] = p.probe(ctx, value.route) })
	}
	group.Wait()

	p.mu.Lock()
	for index, value := range entries {
		value.health = results[index]
		if !results[index].State.Usable() {
			p.applyLocked(value, Signal{State: results[index].State,
				Detail: results[index].Detail, ResetsAt: results[index].ResetsAt})
		}
	}
	p.mu.Unlock()
	return results
}

// Table renders health results as a board.
func Table(results []Health) string {
	widths := []int{len("route"), len("state"), len("model"), len("transport")}
	for _, row := range results {
		widths[0] = max(widths[0], len(row.Route))
		widths[1] = max(widths[1], len(string(row.State)))
		widths[2] = max(widths[2], len(modelColumn(row)))
		widths[3] = max(widths[3], len(row.Transport))
	}
	var out strings.Builder
	line := func(route, state, latency, model, transport, detail string) {
		fmt.Fprintf(&out, "%-*s  %-*s  %8s  %-*s  %-*s  %s\n",
			widths[0], route, widths[1], state, latency, widths[2], model, widths[3], transport, detail)
	}
	line("route", "state", "latency", "model", "transport", "detail")
	for _, row := range results {
		detail := row.Detail
		if !row.ResetsAt.IsZero() {
			detail = fmt.Sprintf("%s, resets %s", detail, row.ResetsAt.UTC().Format("2006-01-02 15:04 UTC"))
		}
		line(row.Route, string(row.State), fmt.Sprintf("%.2fs", row.Latency.Seconds()),
			modelColumn(row), row.Transport, detail)
	}
	return out.String()
}

// modelColumn is what the board should say the route answers on.
//
// The configured model on its own is a claim about what was asked for, and a
// reader takes it for what will arrive. When a deep probe has found out
// otherwise, both are shown, because the difference between them is the whole
// of the news.
func modelColumn(row Health) string {
	if row.Downgraded() {
		return row.Model + " -> " + row.Answered
	}
	return row.Model
}
