package fleet

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/tamnd/llm"
	"github.com/tamnd/llm/route"
)

// Ready is one host's answer to one question: can you take work now, and if
// not, when.
//
// It is one type for every kind of host, and the kind is on it, because the
// question is the same for all of them and only the way of asking differs.
// The package this was extracted from had the question itself specialised —
// CanOCR, which really asked "does this host have a signed in browser
// profile" — and when a machine with a card joined the pool it answered no
// about the best reader in the fleet, printed "this host can read nothing" in
// the board, and gated the only host with a GPU on an Xvfb it will never
// start. So: one question, and a Prober per kind that knows how to ask it.
type Ready struct {
	// Host is what reports call this host, which is the route name. The ssh
	// destination is the route's Host and is deliberately not repeated here:
	// it is the user's ~/.ssh/config business and has no place in a table.
	Host string
	Kind route.Kind
	// OK is whether work sent now has somewhere to land.
	OK bool
	// Detail is one line, for a table cell. It says why when OK is false.
	Detail string
	// Soonest is how long until this host can take work. Zero is now, and
	// less than zero is nothing ready with no time on it to read.
	//
	// The difference between the two is worth the trouble. A host counts a
	// cooldown down to "0m left" and holds it there for the last seconds of
	// it, and a board that read that as no time at all sent a sweep to sleep
	// on another host instead: one host at 2m0s, a sleep of 120s, then twenty
	// minutes of waiting on a second while the first sat there with its
	// cooldown run out.
	Soonest time.Duration
	// TimedOut says the host did not answer inside the deadline, which is a
	// different state from a host that answered and has nothing ready. A table
	// that printed the transport error into a row of counts made a box that
	// was merely slow read the same as a box with every profile banned.
	TimedOut bool

	// Counts is what a pooled host has, and is zero for every other kind. See
	// Counts for why this is numbers and not rows.
	Counts Counts
	// Model is what a reader is serving, for a row that has no counts to print.
	Model     string
	CheckedAt time.Time
}

// Counts is a pooled host's slots, counted and nothing else.
//
// A pooled route reads through signed in browser profiles, and the tool on the
// far side prints one line per profile slot with its state on it — and with
// the account's email address. None of that address is kept. The parser reads
// the state off each line and drops the rest, so nothing downstream can print
// an address it never received. That is deliberate, and it is the reason this
// is counts rather than rows.
type Counts struct {
	// Verified is the slots that hold a real signed in session. The rest of
	// the slots on a box are empty or half registered and cannot answer
	// anything.
	Verified int
	// Free is the verified slots that are neither banned nor held. The table
	// on the far side calls this column "ready".
	Free   int
	Banned int
	// Locked is a slot a live process holds. Stale is a lock left by a process
	// that is gone, which is a slot nothing is using and no work can reach.
	Locked int
	Stale  int
}

// Prober asks one host whether it can take work.
type Prober interface {
	Ready(ctx context.Context, r route.Route) Ready
}

// Fleet picks the right prober for a route and asks it.
//
// The dispatch is on the route's kind and nothing else. Guessing from what is
// installed on the box gets it wrong on exactly the host it matters for: the
// browser tool is installed on the reader too and answers nothing there, so a
// rule of the form "it has the browser tool, so it is a browser box"
// reproduces the bug Ready exists to fix.
type Fleet struct {
	// Runner reaches the boxes. Required for pooled and reader routes; unused
	// for anything asked over HTTP from here.
	Runner Runner
	// HTTPClient is for routes asked from here. Nil means one built from
	// Timeout.
	HTTPClient *http.Client
	// Timeout bounds one question to one host.
	Timeout time.Duration
	Now     func() time.Time
}

// DefaultTimeout is what one host is given to answer. Generous for an ssh
// login to a box under load, short enough that a fleet of five is a report and
// not a coffee break.
const DefaultTimeout = 30 * time.Second

func (f Fleet) now() time.Time {
	if f.Now != nil {
		return f.Now()
	}
	return time.Now().UTC()
}

// Prober is the one that answers for this kind of route, or nil for a kind
// whose readiness is not a fleet question.
func (f Fleet) Prober(kind route.Kind) Prober {
	switch kind {
	case route.KindPool:
		return PoolProber{Runner: f.Runner, Timeout: f.Timeout, Now: f.Now}
	case route.KindReader:
		return ServerProber{Runner: f.Runner, Timeout: f.Timeout, Now: f.Now}
	case route.KindGateway, route.KindDirect:
		return HTTPProber{Client: f.HTTPClient, Timeout: f.Timeout, Now: f.Now}
	default:
		return nil
	}
}

// Ready asks one route's host.
func (f Fleet) Ready(ctx context.Context, r route.Route) Ready {
	prober := f.Prober(r.Kind)
	if prober == nil {
		// An exec route is a program on this machine and has no host to ask.
		// Saying so is better than inventing a false yes, which would put it
		// at the top of a table of boxes it is not one of.
		return Ready{Host: r.Name, Kind: r.Kind, OK: true, Soonest: 0, CheckedAt: f.now(),
			Detail: "a program on this machine, with no host to ask"}
	}
	return prober.Ready(ctx, r)
}

// ReadyAll asks every host at once, in the order given.
//
// Together rather than one after another: they are independent boxes, the trip
// is most of the cost, and five hosts at up to thirty seconds each is two and
// a half minutes of somebody watching a terminal. Results come back in the
// order of the routes, so a report reads the same way twice.
func (f Fleet) ReadyAll(ctx context.Context, routes []route.Route) []Ready {
	out := make([]Ready, len(routes))
	var group sync.WaitGroup
	for index, r := range routes {
		group.Go(func() { out[index] = f.Ready(ctx, r) })
	}
	group.Wait()
	return out
}

// Wait is how long to sit before asking again, across the whole fleet. Zero
// means something can take work now.
//
// A fleet where every host failed to answer also returns zero, because a run
// that goes and finds out is better than a sleep decided on no information.
func Wait(boards []Ready) time.Duration {
	soonest := time.Duration(-1)
	for _, board := range boards {
		// Soonest of exactly zero is a host whose cooldown has run out to the
		// minute and is coming back within it, which is a thing to wait no
		// time at all for. Every prober sets this field deliberately; less
		// than zero is the "nothing ready, no time on it" case. See
		// Ready.Soonest.
		if board.OK || board.Soonest == 0 {
			return 0
		}
		if board.Soonest > 0 && (soonest < 0 || board.Soonest < soonest) {
			soonest = board.Soonest
		}
	}
	if soonest < 0 {
		return 0
	}
	return soonest
}

// PoolProber asks a host with a pool of browser sessions how many of them can
// take work, by running the pool tool's own accounts command over ssh.
type PoolProber struct {
	Runner Runner
	// Tool is the command to run. Empty means AccountsScript, which finds the
	// tool wherever it is installed on that box.
	Tool    string
	Timeout time.Duration
	Now     func() time.Time
}

// Ready runs the accounts command and counts what comes back.
func (p PoolProber) Ready(ctx context.Context, r route.Route) Ready {
	board := Ready{Host: r.Name, Kind: route.KindPool, CheckedAt: now(p.Now)}
	if p.Runner == nil {
		board.Detail = "no ssh runner, so this host cannot be asked"
		board.Soonest = -1
		return board
	}
	ctx, cancel := deadline(ctx, p.Timeout)
	defer cancel()
	script := AccountsScript
	if strings.TrimSpace(p.Tool) != "" {
		script = fmt.Sprintf("%q accounts 2>&1", p.Tool)
	}
	out, err := p.Runner.Run(ctx, r.Host, script)
	if err != nil {
		board.Detail = llm.Condense(err.Error())
		board.TimedOut = timedOut(err)
		board.Soonest = -1
		return board
	}
	counts, soonest := ParseAccounts(out)
	board.Counts = counts
	switch {
	case counts.Verified == 0:
		board.Detail = "no signed in session in the pool, so this host can answer nothing"
		board.Soonest = -1
	case counts.Free > 0:
		board.OK = true
		board.Detail = fmt.Sprintf("%d of %d free", counts.Free, counts.Verified)
	default:
		board.Soonest = soonest
		board.Detail = fmt.Sprintf("%d verified, none free: %d banned, %d held, %d stale",
			counts.Verified, counts.Banned, counts.Locked, counts.Stale)
	}
	return board
}

// ServerScript asks a model server whether it is up.
//
// It is asked over the loopback on the box rather than from here, because that
// is where the endpoint is: a reader serves 127.0.0.1 and there is no tunnel
// to it, since it answers page images and not questions and nothing here
// should be able to send it a conversation by accident.
//
// /models is the listing every OpenAI-shaped server answers, so this reports
// both that the port is open and that what is behind it is the right sort of
// thing. A socket check would say yes to anything that happened to bind.
const ServerScript = `echo "answers=$(curl -fsS --max-time 5 -o /dev/null -w 'yes' 'READER_URL/models' 2>/dev/null || echo '')"`

// ServerProber asks a host serving its own weights whether the model server on
// it answers.
//
// That is a reader's whole readiness, exactly as the counts are a pooled
// host's. It has no accounts to run out of and no cooldown to count down, so
// it is either ready or it is down, and a board that gave it a cooldown would
// be making one up.
type ServerProber struct {
	Runner  Runner
	Timeout time.Duration
	Now     func() time.Time
}

// Ready asks the reader's own model server, from the box it runs on.
func (p ServerProber) Ready(ctx context.Context, r route.Route) Ready {
	board := Ready{Host: r.Name, Kind: route.KindReader, Model: r.ServedModel, CheckedAt: now(p.Now), Soonest: -1}
	if board.Model == "" {
		board.Model = r.Model
	}
	if p.Runner == nil {
		board.Detail = "no ssh runner, so this host cannot be asked"
		return board
	}
	if strings.TrimSpace(r.ReaderURL) == "" {
		board.Detail = "no reader url, so there is no model server to ask about"
		return board
	}
	ctx, cancel := deadline(ctx, p.Timeout)
	defer cancel()
	script := strings.ReplaceAll(ServerScript, "READER_URL", strings.TrimSuffix(r.ReaderURL, "/"))
	out, err := p.Runner.Run(ctx, r.Host, script)
	if err != nil {
		board.Detail = llm.Condense(err.Error())
		board.TimedOut = timedOut(err)
		return board
	}
	if !strings.Contains(out, "answers=yes") {
		board.Detail = "its model server does not answer on " + r.ReaderURL
		return board
	}
	board.OK = true
	board.Soonest = 0
	what := "a reader"
	if board.Model != "" {
		what = "serving " + board.Model
	}
	board.Detail = what + ", and has no accounts to run out of"
	return board
}

// HTTPProber asks an endpoint from this machine, with a GET rather than a
// completion.
//
// A measured round trip to one of these for nine tokens is about two and a
// half minutes, so probing a fleet that way costs ten and would make a
// readiness check useless as a cron guard. A GET answers in milliseconds.
type HTTPProber struct {
	Client *http.Client
	// Path is what to GET, relative to the route's base URL. Empty means
	// /models, which every OpenAI-shaped server answers.
	Path    string
	Timeout time.Duration
	Now     func() time.Time
}

// Ready gets the endpoint and reads the answer through llm.Classify, so that a
// quota with an hour on it reads as an hour and not as a dead host.
func (p HTTPProber) Ready(ctx context.Context, r route.Route) Ready {
	board := Ready{Host: r.Name, Kind: r.Kind, Model: r.Wire(), CheckedAt: now(p.Now), Soonest: -1}
	path := p.Path
	if path == "" {
		path = "/models"
	}
	ctx, cancel := deadline(ctx, p.Timeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, r.Endpoint(path), nil)
	if err != nil {
		board.Detail = err.Error()
		return board
	}
	if key := r.Key(); key != "" {
		request.Header.Set("Authorization", "Bearer "+key)
	}
	request.Header.Set("User-Agent", llm.UserAgent())
	client := p.Client
	if client == nil {
		client = &http.Client{Timeout: p.timeout()}
	}
	response, err := client.Do(request)
	if err != nil {
		signal := llm.ClassifyError(err)
		board.Detail = signal.Detail
		board.TimedOut = timedOut(err)
		return board
	}
	defer func() { _ = response.Body.Close() }()
	raw, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if response.StatusCode >= 200 && response.StatusCode < 300 {
		board.OK = true
		board.Soonest = 0
		board.Detail = fmt.Sprintf("%s answers on %s", r.Kind, path)
		return board
	}
	signal := llm.Classify(response.StatusCode, response.Header, raw)
	board.Detail = signal.Detail
	switch {
	case signal.RetryAfter > 0:
		board.Soonest = signal.RetryAfter
	case !signal.ResetsAt.IsZero():
		if left := signal.ResetsAt.Sub(now(p.Now)); left > 0 {
			board.Soonest = left
		} else {
			board.Soonest = 0
		}
	}
	return board
}

func (p HTTPProber) timeout() time.Duration {
	if p.Timeout > 0 {
		return p.Timeout
	}
	return DefaultTimeout
}

func now(clock func() time.Time) time.Time {
	if clock != nil {
		return clock()
	}
	return time.Now().UTC()
}

func deadline(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	return context.WithTimeout(ctx, timeout)
}

// Table renders readiness the way a status command prints it.
//
// Two shapes of row, because there are two things a row can be saying. A
// pooled host has counts. Everything else has a sentence: a reader has a model
// and whether it answers, an endpoint has whether it answered, and printing
// zeroes in the account columns for either would be true and misleading.
func Table(boards []Ready) string {
	var out strings.Builder
	fmt.Fprintf(&out, "%-12s  %-8s  %8s  %5s  %6s  %6s  %5s  %s\n",
		"host", "kind", "verified", "free", "banned", "held", "stale", "state")
	for _, board := range boards {
		if board.TimedOut {
			fmt.Fprintf(&out, "%-12s  %-8s  did not answer inside the deadline, so nothing is known about it\n",
				board.Host, board.Kind)
			continue
		}
		if board.Kind != route.KindPool || board.Counts.Verified == 0 {
			fmt.Fprintf(&out, "%-12s  %-8s  %s\n", board.Host, board.Kind, board.Detail)
			continue
		}
		back := "now"
		switch {
		case board.OK:
		case board.Soonest < 0:
			back = "not for a while"
		case board.Soonest > 0:
			back = board.Soonest.Round(time.Minute).String()
		default:
			back = "any minute"
		}
		c := board.Counts
		fmt.Fprintf(&out, "%-12s  %-8s  %8d  %5d  %6d  %6d  %5d  %s\n",
			board.Host, board.Kind, c.Verified, c.Free, c.Banned, c.Locked, c.Stale, back)
	}
	return out.String()
}
