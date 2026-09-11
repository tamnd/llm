package fleet

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tamnd/llm/route"
)

// accountsTable is the shape the pool tool prints, recorded rather than
// guessed: a box drawing frame, a header, one row per slot, and the state in a
// mixture of cases because the far side does not keep one.
//
// The addresses are invented. The point of the fixture is that none of them
// reaches anything this package returns.
const accountsTable = `
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ account               ┃ state      ┃ ready               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ one@example.invalid   │ ✓          │ yes                 │
│ two@example.invalid   │ ✓ BANNED   │ 1h5m left (09:49)   │
│ three@example.invalid │ ✓ BANNED   │ 12m left (09:49)    │
│ four@example.invalid  │ ✓ LOCKED   │ held by 41233       │
│ five@example.invalid  │ ✓ stale-lock │ pid 9 is gone     │
│ six@example.invalid   │            │ not registered      │
└───────────────────────┴────────────┴─────────────────────┘
`

func TestParseAccounts(t *testing.T) {
	counts, soonest := ParseAccounts(accountsTable)
	want := Counts{Verified: 5, Free: 1, Banned: 2, Locked: 1, Stale: 1}
	if counts != want {
		t.Errorf("counts = %+v, want %+v", counts, want)
	}
	// The boxes do not keep this machine's clock — one said 09:49 while it was
	// 14:49 here — so the cooldown is read off the left column and not off the
	// time printed beside it.
	if soonest != 12*time.Minute {
		t.Errorf("soonest = %s, want the nearest slot's own countdown", soonest)
	}
}

// Reading the lock in lower case alone found neither of the two held slots a
// fleet really had and counted each of them free, which was enough to keep a
// sweep awake.
func TestParseAccountsReadsTheStateWhateverItsCase(t *testing.T) {
	for _, c := range []struct {
		name string
		line string
		want Counts
	}{
		{"upper banned", "│ a@example.invalid │ ✓ BANNED │ 5m left │", Counts{Verified: 1, Banned: 1}},
		{"lower banned", "│ a@example.invalid │ ✓ banned │ 5m left │", Counts{Verified: 1, Banned: 1}},
		{"upper lock", "│ a@example.invalid │ ✓ LOCKED │ held │", Counts{Verified: 1, Locked: 1}},
		{"lower lock", "│ a@example.invalid │ ✓ locked │ held │", Counts{Verified: 1, Locked: 1}},
		{"stale lock", "│ a@example.invalid │ ✓ stale-lock │ gone │", Counts{Verified: 1, Stale: 1}},
		{"free", "│ a@example.invalid │ ✓ │ yes │", Counts{Verified: 1, Free: 1}},
		{"not a slot", "│ a@example.invalid │ │ not registered │", Counts{}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got, _ := ParseAccounts(c.line); got != c.want {
				t.Errorf("counts = %+v, want %+v", got, c.want)
			}
		})
	}
}

// A host counts a cooldown down to "0m left" and holds it there for the last
// seconds of it. Zero is a thing to wait no time at all for; below zero is a
// host with nothing ready and no time to read off it.
func TestParseAccountsTellsZeroFromNothing(t *testing.T) {
	_, soonest := ParseAccounts("│ a@example.invalid │ ✓ BANNED │ 0m left │")
	if soonest != 0 {
		t.Errorf("soonest = %s, want no wait at all", soonest)
	}
	if _, soonest := ParseAccounts("│ a@example.invalid │ ✓ BANNED │ until tomorrow │"); soonest >= 0 {
		t.Errorf("soonest = %s, want it below zero when nothing said how long", soonest)
	}
	if _, soonest := ParseAccounts("nothing here at all"); soonest >= 0 {
		t.Errorf("soonest = %s, want it below zero", soonest)
	}
}

// A new column on the far side costs a host its counts for one run and cannot
// invent slots that are not there.
func TestParseAccountsSkipsWhatItDoesNotKnow(t *testing.T) {
	counts, _ := ParseAccounts("Traceback (most recent call last):\n  File \"x\", line 1\nValueError: no config\n")
	if counts != (Counts{}) {
		t.Errorf("counts = %+v, want nothing out of a stack trace", counts)
	}
}

// fake is a Runner that answers from a recorded map and remembers what it was
// asked, so every probe in this package runs without a box.
type fake struct {
	mu   sync.Mutex
	out  map[string]string
	err  map[string]error
	sent []string
}

func (f *fake) Run(_ context.Context, host, command string) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sent = append(f.sent, host+": "+command)
	if err := f.err[host]; err != nil {
		return "", err
	}
	return f.out[host], nil
}

func (f *fake) commands() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.sent...)
}

func pool(name string) route.Route {
	return route.Route{Name: name, Kind: route.KindPool, Host: "box-" + name,
		BaseURL: "http://127.0.0.1:18771/v1", Model: "m"}
}

func TestPoolProberCountsWhatCameBack(t *testing.T) {
	runner := &fake{out: map[string]string{"box-server1": accountsTable}}
	board := PoolProber{Runner: runner}.Ready(context.Background(), pool("server1"))

	if !board.OK {
		t.Fatalf("board = %+v, want a host with a free slot ready", board)
	}
	if board.Counts.Free != 1 || board.Counts.Verified != 5 {
		t.Errorf("counts = %+v", board.Counts)
	}
	if board.Host != "server1" {
		t.Errorf("host = %q, want the route name and not the ssh destination", board.Host)
	}
	// The ssh destination is the user's ~/.ssh/config business and has no place
	// in a report, and neither has an address off the account table.
	if strings.Contains(board.Detail, "@") || strings.Contains(board.Detail, "box-") {
		t.Errorf("detail = %q", board.Detail)
	}
	// The table is drawn by a console library, which writes to the console it
	// opened, so the command has to fold that stream in.
	if !strings.Contains(runner.commands()[0], "2>&1") {
		t.Errorf("command = %q, want stderr folded in", runner.commands()[0])
	}
}

func TestPoolProberWhenNothingIsFree(t *testing.T) {
	held := `│ a@example.invalid │ ✓ BANNED │ 20m left │
│ b@example.invalid │ ✓ LOCKED │ held │`
	board := PoolProber{Runner: &fake{out: map[string]string{"box-server1": held}}}.
		Ready(context.Background(), pool("server1"))
	if board.OK {
		t.Fatal("a host with every slot banned or held said it was ready")
	}
	if board.Soonest != 20*time.Minute {
		t.Errorf("soonest = %s, want the banned slot's countdown", board.Soonest)
	}
}

// A host that is up with no session in its pool answers every call with a
// refusal, and nothing is coming back on its own.
func TestPoolProberWithAnEmptyPool(t *testing.T) {
	board := PoolProber{Runner: &fake{out: map[string]string{"box-server1": ""}}}.
		Ready(context.Background(), pool("server1"))
	if board.OK || board.Soonest >= 0 {
		t.Errorf("board = %+v, want nothing ready and no time on it", board)
	}
	if !strings.Contains(board.Detail, "nothing") {
		t.Errorf("detail = %q", board.Detail)
	}
}

// A table that printed the transport error into a row of counts made a box
// that was merely slow read the same as a box with every profile banned.
func TestPoolProberSaysWhenAHostNeverAnswered(t *testing.T) {
	runner := &fake{err: map[string]error{"box-server1": errors.New("box-server1: context deadline exceeded after 30s")}}
	board := PoolProber{Runner: runner}.Ready(context.Background(), pool("server1"))
	if board.OK || !board.TimedOut {
		t.Errorf("board = %+v, want a host that did not answer", board)
	}
	if board.Soonest >= 0 {
		t.Errorf("soonest = %s, want no promise about a host that said nothing", board.Soonest)
	}
	if got := Table([]Ready{board}); !strings.Contains(got, "did not answer") {
		t.Errorf("table:\n%s", got)
	}
}

func TestPoolProberWithoutARunner(t *testing.T) {
	board := PoolProber{}.Ready(context.Background(), pool("server1"))
	if board.OK || !strings.Contains(board.Detail, "ssh") {
		t.Errorf("board = %+v", board)
	}
}

func reader() route.Route {
	return route.Route{Name: "gpu", Kind: route.KindReader, Host: "box-gpu", Reader: "local-ocr",
		ReaderURL: "http://127.0.0.1:8000/v1/", Model: "olmocr", ServedModel: "allenai/olmOCR-2-7B"}
}

func TestServerProberAsksTheModelServerFromTheBox(t *testing.T) {
	runner := &fake{out: map[string]string{"box-gpu": "answers=yes\n"}}
	board := ServerProber{Runner: runner}.Ready(context.Background(), reader())
	if !board.OK || board.Soonest != 0 {
		t.Fatalf("board = %+v", board)
	}
	// A reader has no accounts to run out of, so a row for it is a sentence
	// and not a set of zeroes in the account columns.
	if !strings.Contains(board.Detail, "allenai/olmOCR-2-7B") {
		t.Errorf("detail = %q, want the weights it is serving", board.Detail)
	}
	command := runner.commands()[0]
	if strings.Contains(command, "READER_URL") {
		t.Errorf("command = %q, the placeholder was not filled", command)
	}
	// The endpoint is on the box's own loopback and there is no tunnel to it:
	// it answers page images, not questions.
	if !strings.Contains(command, "http://127.0.0.1:8000/v1/models") {
		t.Errorf("command = %q", command)
	}
}

func TestServerProberWhenTheModelServerIsDown(t *testing.T) {
	board := ServerProber{Runner: &fake{out: map[string]string{"box-gpu": "answers=\n"}}}.
		Ready(context.Background(), reader())
	if board.OK {
		t.Fatal("a reader with nothing listening said it was ready")
	}
	// Not a cooldown. It is either up or it is down, and a board that gave it
	// a time would be making one up.
	if board.Soonest >= 0 {
		t.Errorf("soonest = %s", board.Soonest)
	}
	if !strings.Contains(board.Detail, "127.0.0.1:8000") {
		t.Errorf("detail = %q, want where it looked", board.Detail)
	}
}

func TestServerProberWithNoURL(t *testing.T) {
	r := reader()
	r.ReaderURL = ""
	board := ServerProber{Runner: &fake{}}.Ready(context.Background(), r)
	if board.OK || !strings.Contains(board.Detail, "no reader url") {
		t.Errorf("board = %+v", board)
	}
}

func TestHTTPProberGetsTheCatalogue(t *testing.T) {
	var asked string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked = r.URL.Path
		_, _ = w.Write([]byte(`{"data":[{"id":"free-model"}]}`))
	}))
	t.Cleanup(server.Close)

	r := route.Route{Name: "free", Kind: route.KindGateway, BaseURL: server.URL + "/v1", Model: "free-model"}
	board := HTTPProber{Client: server.Client()}.Ready(context.Background(), r)
	if !board.OK || board.Soonest != 0 {
		t.Fatalf("board = %+v", board)
	}
	// A completion on one of these has measured about two and a half minutes,
	// which would make a readiness check useless as a guard. A GET answers in
	// milliseconds.
	if asked != "/v1/models" {
		t.Errorf("asked %q", asked)
	}
}

func TestHTTPProberReadsAQuotaAsATime(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Retry-After", "600")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate_limit_exceeded"}}`))
	}))
	t.Cleanup(server.Close)

	r := route.Route{Name: "free", Kind: route.KindGateway, BaseURL: server.URL + "/v1", Model: "m"}
	board := HTTPProber{Client: server.Client()}.Ready(context.Background(), r)
	if board.OK {
		t.Fatal("a host out of quota said it was ready")
	}
	if board.Soonest < 9*time.Minute || board.Soonest > 11*time.Minute {
		t.Errorf("soonest = %s, want the ten minutes the header asked for", board.Soonest)
	}
}

func TestHTTPProberOnARejectedKey(t *testing.T) {
	var sent string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sent = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid_api_key"}}`))
	}))
	t.Cleanup(server.Close)

	t.Setenv("TEST_FLEET_KEY", "a-key")
	r := route.Route{Name: "free", Kind: route.KindGateway, BaseURL: server.URL + "/v1",
		Model: "m", APIKeyEnv: "TEST_FLEET_KEY"}
	board := HTTPProber{Client: server.Client()}.Ready(context.Background(), r)
	if board.OK || board.Soonest >= 0 {
		t.Errorf("board = %+v, want nothing ready and no time on it", board)
	}
	if sent != "Bearer a-key" {
		t.Errorf("authorization = %q", sent)
	}
}

// Guessing the prober from what is installed on the box gets it wrong on
// exactly the host it matters for: the browser tool is installed on the reader
// too and answers nothing there.
func TestFleetPicksTheProberByKind(t *testing.T) {
	f := Fleet{Runner: &fake{}}
	for _, c := range []struct {
		kind route.Kind
		want string
	}{
		{route.KindPool, "fleet.PoolProber"},
		{route.KindReader, "fleet.ServerProber"},
		{route.KindGateway, "fleet.HTTPProber"},
		{route.KindDirect, "fleet.HTTPProber"},
	} {
		if got := f.Prober(c.kind); got == nil {
			t.Errorf("%s has no prober", c.kind)
		} else if name := fmt.Sprintf("%T", got); name != c.want {
			t.Errorf("%s asked %s, want %s", c.kind, name, c.want)
		}
	}
	if f.Prober(route.KindExec) != nil {
		t.Error("an exec route was given a host to ask")
	}
}

// An exec route is a program on this machine. Saying so is better than
// inventing a false yes, which would put it at the top of a table of boxes it
// is not one of.
func TestFleetReadyOnAnExecRoute(t *testing.T) {
	board := Fleet{}.Ready(context.Background(),
		route.Route{Name: "cli", Kind: route.KindExec, Command: "codex", Model: "m"})
	if !board.OK || board.Soonest != 0 {
		t.Errorf("board = %+v", board)
	}
	if !strings.Contains(board.Detail, "this machine") {
		t.Errorf("detail = %q", board.Detail)
	}
}

// Five hosts at up to thirty seconds each is two and a half minutes of
// somebody watching a terminal, so they are asked together — and the answers
// come back in the order of the routes, so a report reads the same twice.
func TestReadyAllKeepsTheOrderOfTheRoutes(t *testing.T) {
	runner := &fake{out: map[string]string{
		"box-a": accountsTable,
		"box-b": "",
		"box-c": accountsTable,
	}}
	routes := []route.Route{pool("a"), pool("b"), pool("c")}
	boards := Fleet{Runner: runner, Timeout: time.Second}.ReadyAll(context.Background(), routes)
	if len(boards) != 3 {
		t.Fatalf("got %d boards", len(boards))
	}
	for index, want := range []string{"a", "b", "c"} {
		if boards[index].Host != want {
			t.Errorf("board %d is %s, want %s", index, boards[index].Host, want)
		}
	}
	if boards[1].OK {
		t.Error("the empty box was reported ready")
	}
}

func TestWait(t *testing.T) {
	for _, c := range []struct {
		name   string
		boards []Ready
		want   time.Duration
	}{
		{"something is ready", []Ready{{Soonest: -1}, {OK: true}}, 0},
		// A cooldown counted down to zero is coming back within the minute,
		// and sleeping past it once cost twenty minutes on another host.
		{"a cooldown just ran out", []Ready{{Soonest: -1}, {Soonest: 0}}, 0},
		{"the nearest wait", []Ready{{Soonest: 30 * time.Minute}, {Soonest: 5 * time.Minute}}, 5 * time.Minute},
		// A run that goes and finds out beats a sleep decided on no
		// information.
		{"nobody answered", []Ready{{Soonest: -1}, {Soonest: -1}}, 0},
		{"no hosts at all", nil, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := Wait(c.boards); got != c.want {
				t.Errorf("Wait = %s, want %s", got, c.want)
			}
		})
	}
}

// Two shapes of row: a pooled host has counts, and everything else has a
// sentence. Printing zeroes in the account columns for a reader would be true
// and misleading.
func TestTableHasARowShapePerKind(t *testing.T) {
	counts, _ := ParseAccounts(accountsTable)
	boards := []Ready{
		{Host: "server1", Kind: route.KindPool, OK: true, Counts: counts, Detail: "1 of 5 free"},
		{Host: "server2", Kind: route.KindPool, Counts: Counts{Verified: 2, Banned: 2}, Soonest: 20 * time.Minute},
		{Host: "gpu", Kind: route.KindReader, OK: true, Detail: "serving allenai/olmOCR-2-7B, and has no accounts to run out of"},
	}
	got := Table(boards)
	if strings.Contains(got, "@") {
		t.Errorf("an address reached the board:\n%s", got)
	}
	for _, want := range []string{"verified", "server1", "now", "20m0s", "allenai/olmOCR-2-7B"} {
		if !strings.Contains(got, want) {
			t.Errorf("table has no %q:\n%s", want, got)
		}
	}
	lines := strings.Split(strings.TrimSpace(got), "\n")
	if len(lines) != 4 {
		t.Errorf("table has %d lines, want a header and one row each:\n%s", len(lines), got)
	}
}

// A command that stops for a passphrase in an unattended run hangs until
// somebody notices days later.
func TestSSHAlwaysAsksInBatchMode(t *testing.T) {
	args := SSH{Options: []string{"StrictHostKeyChecking=accept-new"}}.args("box", "echo hi")
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "BatchMode=yes") || !strings.Contains(joined, "ConnectTimeout=10") {
		t.Errorf("args = %v", args)
	}
	if !strings.Contains(joined, "StrictHostKeyChecking=accept-new") {
		t.Errorf("the caller's own options were dropped: %v", args)
	}
	if args[len(args)-2] != "box" || args[len(args)-1] != "echo hi" {
		t.Errorf("args = %v, want the host then the command", args)
	}
	if _, err := (SSH{}).Run(context.Background(), "  ", "echo hi"); err == nil {
		t.Error("a command with no host was run")
	}
}

func TestTimedOut(t *testing.T) {
	for _, c := range []struct {
		err  error
		want bool
	}{
		{nil, false},
		{errors.New("box: context deadline exceeded after 30s"), true},
		{errors.New("ssh: connect to host box port 22: Operation timed out"), true},
		{errors.New("dial tcp: i/o timeout"), true},
		{errors.New("ssh: connect to host box port 22: Connection refused"), false},
	} {
		if got := timedOut(c.err); got != c.want {
			t.Errorf("timedOut(%v) = %t", c.err, got)
		}
	}
}
