package fleet

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tamnd/llm"
	"github.com/tamnd/llm/route"
)

// process is a tunnel this test controls, so nothing here needs a rented box
// or an open port.
type process struct {
	pid    int
	killed atomic.Bool
	done   chan error
}

func newProcess(pid int) *process { return &process{pid: pid, done: make(chan error, 1)} }

func (p *process) PID() int { return p.pid }

func (p *process) Kill() error {
	if p.killed.CompareAndSwap(false, true) {
		p.done <- nil
	}
	return nil
}

func (p *process) Wait() error { return <-p.done }

func up() Link {
	return Link{Route: "server1", Host: "box-server1", LocalPort: 18771, RemotePort: 18771,
		Check: func(context.Context) error { return nil }}
}

// A port that is already answering is somebody else's tunnel, or this command
// run twice. Starting a second ssh on it fails in a way that reads as a broken
// host.
func TestUpAdoptsAPortThatIsAlreadyAnswering(t *testing.T) {
	var started int
	s := &Supervisor{Start: func(Link) (Process, error) { started++; return newProcess(1), nil }}
	tunnels, err := s.Up(context.Background(), []Link{up()})
	if err != nil {
		t.Fatalf("Up: %v", err)
	}
	if started != 0 {
		t.Errorf("started %d tunnels over a port that was already serving", started)
	}
	if len(tunnels) != 1 || tunnels[0].Route != "server1" {
		t.Fatalf("tunnels = %+v", tunnels)
	}
	// No pid, because this process does not own it and must not kill it.
	if tunnels[0].PID != 0 {
		t.Errorf("adopted tunnel claims pid %d", tunnels[0].PID)
	}
}

func TestUpStartsATunnelAndWaitsForIt(t *testing.T) {
	var checks atomic.Int32
	link := up()
	link.Check = func(context.Context) error {
		if checks.Add(1) == 1 {
			return errors.New("connection refused")
		}
		return nil
	}
	s := &Supervisor{Start: func(Link) (Process, error) { return newProcess(4242), nil }}
	tunnels, err := s.Up(context.Background(), []Link{link})
	if err != nil {
		t.Fatalf("Up: %v", err)
	}
	if len(tunnels) != 1 || tunnels[0].PID != 4242 {
		t.Fatalf("tunnels = %+v", tunnels)
	}
	if tunnels[0].LocalPort != 18771 || tunnels[0].Started.IsZero() {
		t.Errorf("tunnel = %+v", tunnels[0])
	}
}

// A tunnel that is up but dead is worse than one that is down, because every
// call on it hangs for the full timeout. One that never answers is killed
// rather than left running.
func TestUpKillsATunnelThatNeverAnswers(t *testing.T) {
	link := up()
	link.Check = func(context.Context) error { return errors.New("connection refused") }
	var started *process
	s := &Supervisor{Start: func(Link) (Process, error) { started = newProcess(4242); return started, nil }}

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if _, err := s.Up(ctx, []Link{link}); err == nil {
		t.Fatal("a tunnel that never answered was reported up")
	}
	if started == nil || !started.killed.Load() {
		t.Error("the ssh that never came up was left running")
	}
	if len(s.running) != 0 {
		t.Errorf("running = %v, want it forgotten", s.running)
	}
}

// A partial fleet is a working fleet. One box being down for a month is the
// ordinary case, and refusing to run without it would mean never running.
func TestUpIsAPartialFleet(t *testing.T) {
	good := up()
	bad := Link{Route: "server2", Host: "box-server2", LocalPort: 18772, RemotePort: 18772,
		Check: func(context.Context) error { return errors.New("no route to host") }}

	var logs []string
	s := &Supervisor{
		Logf:  func(format string, args ...any) { logs = append(logs, fmt.Sprintf(format, args...)) },
		Start: func(Link) (Process, error) { return newProcess(7), nil },
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	tunnels, err := s.Up(ctx, []Link{good, bad})
	if err != nil {
		t.Fatalf("Up: %v, want the fleet that did come up", err)
	}
	if len(tunnels) != 1 || tunnels[0].Route != "server1" {
		t.Fatalf("tunnels = %+v", tunnels)
	}
	if !strings.Contains(strings.Join(logs, "\n"), "server2") {
		t.Errorf("the host that stayed down was not mentioned: %v", logs)
	}
}

// One failed check is a blip on a laptop's wifi; two in a row on the poll is a
// tunnel that is not coming back on its own.
func TestWatchRestartsATunnelThatKeepsFailing(t *testing.T) {
	var checks atomic.Int32
	link := up()
	link.Check = func(context.Context) error {
		if checks.Add(1) <= FailuresBeforeRestart {
			return errors.New("connection refused")
		}
		return nil
	}

	var (
		mu        sync.Mutex
		logs      []string
		restarted = make(chan struct{}, 1)
	)
	s := &Supervisor{Logf: func(format string, args ...any) {
		line := fmt.Sprintf(format, args...)
		mu.Lock()
		logs = append(logs, line)
		mu.Unlock()
		if strings.Contains(line, "restarted") {
			select {
			case restarted <- struct{}{}:
			default:
			}
		}
	}}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	go s.Watch(ctx, []Link{link}, 5*time.Millisecond)

	select {
	case <-restarted:
	case <-ctx.Done():
		t.Fatal("a tunnel that failed every check was never restarted")
	}
	mu.Lock()
	defer mu.Unlock()
	joined := strings.Join(logs, "\n")
	for _, want := range []string{"failed check 1", "failed check 2"} {
		if !strings.Contains(joined, want) {
			t.Errorf("log has no %q:\n%s", want, joined)
		}
	}
}

func TestDownKillsWhatThisProcessStarted(t *testing.T) {
	started := newProcess(4242)
	s := &Supervisor{Start: func(Link) (Process, error) { return started, nil }}
	link := up()
	var checks atomic.Int32
	link.Check = func(context.Context) error {
		if checks.Add(1) == 1 {
			return errors.New("nothing there yet")
		}
		return nil
	}
	tunnels, err := s.Up(context.Background(), []Link{link})
	if err != nil {
		t.Fatal(err)
	}
	// A tunnel the state file remembers with no pid on it is one nothing here
	// can kill, and it must not turn into an error.
	tunnels = append(tunnels, Tunnel{Route: "adopted", LocalPort: 18772})
	if errs := s.Down(tunnels); len(errs) != 0 {
		t.Errorf("Down: %v", errs)
	}
	if !started.killed.Load() {
		t.Error("the tunnel this process started is still running")
	}
}

// Every listener worth tunnelling to binds 127.0.0.1, and the only way in is
// ssh. These are the flags that make one notice a silent connection and exit
// rather than hang.
func TestTunnelArgs(t *testing.T) {
	args := strings.Join(tunnelArgs(up()), " ")
	for _, want := range []string{
		"-N", "-L 18771:127.0.0.1:18771", "BatchMode=yes", "ExitOnForwardFailure=yes",
		"ServerAliveInterval=20", "ServerAliveCountMax=3",
		// A host configured with ControlMaster auto hands the forward to a
		// shared master a previous probe left behind: the pid recorded here
		// belongs to a client that exits.
		"ControlMaster=no", "ControlPath=none", "box-server1",
	} {
		if !strings.Contains(args, want) {
			t.Errorf("args = %q, want %q in them", args, want)
		}
	}
}

func TestLinksAreTheRoutesThatNeedATunnel(t *testing.T) {
	registry := route.Registry{Routes: []route.Route{
		{Name: "server1", Kind: route.KindPool, Host: "box1", BaseURL: "http://127.0.0.1:18771/v1",
			Model: "m", LocalPort: 18771, RemotePort: 18771, Rank: 1},
		{Name: "gpu", Kind: route.KindReader, Host: "box2", Reader: "local-ocr", Model: "m",
			LocalPort: 18772, RemotePort: 8000, Rank: 2},
		// A gateway is reached over the open internet and has nothing to
		// forward, whatever ports somebody typed into its entry.
		{Name: "free", Kind: route.KindGateway, BaseURL: "https://example.invalid/v1", Model: "m",
			LocalPort: 18773, RemotePort: 18773, Rank: 3},
		{Name: "half", Kind: route.KindPool, Host: "box4", BaseURL: "http://127.0.0.1:18774/v1",
			Model: "m", LocalPort: 18774, Rank: 4},
		{Name: "off", Kind: route.KindPool, Host: "box5", BaseURL: "http://127.0.0.1:18775/v1",
			Model: "m", LocalPort: 18775, RemotePort: 18775, Rank: 5, Disabled: true},
	}}
	links := Links(registry)
	if len(links) != 2 {
		t.Fatalf("links = %+v, want the two routes reached over ssh with both ends named", links)
	}
	if links[0].Route != "server1" || links[1].Route != "gpu" {
		t.Errorf("links = %+v", links)
	}
	// Printed by a status command so a person can run the same ssh by hand.
	if got := links[1].Forward(); got != "18772:127.0.0.1:8000" {
		t.Errorf("forward = %q", got)
	}
}

func TestStateRoundTrips(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fleet.json")
	// A missing file means no fleet has been brought up in this configuration
	// yet, which is not an error.
	empty, err := LoadState(path)
	if err != nil {
		t.Fatalf("LoadState of a missing file: %v", err)
	}
	if len(empty.Tunnels) != 0 || empty.Tools == nil {
		t.Errorf("state = %+v, want an empty one ready to write into", empty)
	}

	want := State{
		Written: time.Now().UTC().Truncate(time.Second),
		Tunnels: []Tunnel{{Host: "box1", Route: "server1", LocalPort: 18771, RemotePort: 18771,
			PID: os.Getpid(), Started: time.Now().UTC().Truncate(time.Second)}},
		Tools: map[string]string{"box1": "/home/user/chatgpt-tool/.venv/bin/chatgpt-tool"},
	}
	if err := want.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	got, err := LoadState(path)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	tunnel, ok := got.Find("server1")
	if !ok || tunnel.LocalPort != 18771 {
		t.Errorf("Find = %+v, %t", tunnel, ok)
	}
	if _, ok := got.Find("nothing"); ok {
		t.Error("Find invented a tunnel")
	}
	if got.Tools["box1"] == "" {
		t.Error("the discovered tool path was lost")
	}
	// The state file can carry a path from somebody's home directory, so it is
	// written for its owner only.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if mode := info.Mode().Perm(); mode != 0o600 {
		t.Errorf("mode = %o, want 0600", mode)
	}
}

// The difference between a fleet that is up and a file that says it is.
func TestStaleIsTheTunnelsWhoseProcessIsGone(t *testing.T) {
	state := State{Tunnels: []Tunnel{
		{Route: "mine", PID: os.Getpid()},
		{Route: "gone", PID: gonePID(t)},
		{Route: "adopted"},
	}}
	stale := state.Stale()
	if len(stale) != 1 || stale[0].Route != "gone" {
		t.Errorf("stale = %+v", stale)
	}
}

func TestStatePathFollowsTheApp(t *testing.T) {
	dir := t.TempDir()
	llm.Configure(llm.Config{App: "papers", ConfigDir: dir})
	t.Cleanup(func() { llm.Configure(llm.Config{}) })

	if got := StatePath(); got != filepath.Join(dir, "fleet.json") {
		t.Errorf("path = %q, want it beside the route file", got)
	}
	t.Setenv("PAPERS_FLEET_STATE", "/tmp/somewhere-else.json")
	if got := StatePath(); got != "/tmp/somewhere-else.json" {
		t.Errorf("path = %q, want the environment's", got)
	}
}

func TestAliveAndKill(t *testing.T) {
	if !Alive(os.Getpid()) {
		t.Error("this process reported itself dead")
	}
	if Alive(0) || Alive(-1) {
		t.Error("a pid that is not a pid reported alive")
	}
	if Alive(gonePID(t)) {
		t.Error("a process that has exited reported alive")
	}
	if err := Kill(0); err == nil {
		t.Error("Kill with no pid was accepted")
	}
	// A pid that is gone is the desired state, so it is not an error.
	if err := Kill(gonePID(t)); err != nil {
		t.Errorf("Kill of a process that has already exited: %v", err)
	}
}

// gonePID is the pid of a process that has run and been waited for. A pid can
// in principle be reused, so the test checks before relying on it.
func gonePID(t *testing.T) int {
	t.Helper()
	cmd := exec.Command("/bin/sh", "-c", "exit 0")
	if err := cmd.Run(); err != nil {
		t.Skipf("cannot run a throwaway process here: %v", err)
	}
	pid := cmd.Process.Pid
	if Alive(pid) {
		t.Skipf("pid %d was reused between exit and the check", pid)
	}
	return pid
}
