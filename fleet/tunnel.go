package fleet

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/tamnd/llm/route"
)

// Backoff for a tunnel that will not stay up. The floor is five seconds
// because a box that is rebooting is not helped by being dialled ten times a
// second, and the ceiling is five minutes because a box that has been down for
// an hour is usually down for a reason a person has to fix.
const (
	MinBackoff = 5 * time.Second
	MaxBackoff = 5 * time.Minute
	// UpDeadline is how long bringing a tunnel up waits for it to answer
	// before calling it failed.
	UpDeadline = 30 * time.Second
	// PollInterval is how often a supervised tunnel is checked.
	PollInterval = 30 * time.Second
	// FailuresBeforeRestart is how many checks in a row must fail. One is a
	// blip on a laptop's wifi; two in a row on a thirty second poll is a
	// tunnel that is not coming back on its own.
	FailuresBeforeRestart = 2
)

// Link is one tunnel to keep up.
type Link struct {
	// Route is the name in the route file, which is what reports and errors
	// call this host.
	Route      string
	Host       string
	LocalPort  int
	RemotePort int
	// Check answers whether the far end is serving. It is a function rather
	// than a URL because what counts as healthy is the caller's business, not
	// this package's.
	Check func(context.Context) error
}

// Links are the tunnels a registry needs, one per route that reaches its host
// over ssh and names both ends of a forward.
func Links(registry route.Registry) []Link {
	var out []Link
	for _, r := range registry.Enabled() {
		if !r.Kind.OverSSH() || r.LocalPort <= 0 || r.RemotePort <= 0 || r.Host == "" {
			continue
		}
		out = append(out, Link{Route: r.Name, Host: r.Host,
			LocalPort: r.LocalPort, RemotePort: r.RemotePort})
	}
	return out
}

// Forward is the -L argument for a link, which a status command prints so a
// person can run the same ssh by hand.
func (l Link) Forward() string {
	return strconv.Itoa(l.LocalPort) + ":127.0.0.1:" + strconv.Itoa(l.RemotePort)
}

// Supervisor starts tunnels and puts them back when they die.
type Supervisor struct {
	// Binary is the ssh to run. Empty means the one on PATH.
	Binary string
	Logf   func(string, ...any)
	// Start is how a tunnel process is launched, replaceable so a test does
	// not need a rented box and an open port.
	Start func(link Link) (Process, error)

	mu      sync.Mutex
	running map[string]Process
}

// Process is a started tunnel. The interface exists so a test can hand back
// something it controls.
type Process interface {
	PID() int
	Kill() error
	// Wait blocks until the process exits.
	Wait() error
}

func (s *Supervisor) logf(format string, args ...any) {
	if s.Logf != nil {
		s.Logf(format, args...)
	}
}

// Up starts every link and waits for each to answer, or for UpDeadline.
//
// The links come up together rather than one after another. Three hosts at up
// to thirty seconds each is a minute and a half of somebody watching a
// terminal, and they have nothing to do with each other.
func (s *Supervisor) Up(ctx context.Context, links []Link) ([]Tunnel, error) {
	out := make([]Tunnel, len(links))
	errs := make([]error, len(links))
	var group sync.WaitGroup
	for index, link := range links {
		group.Go(func() { out[index], errs[index] = s.start(ctx, link) })
	}
	group.Wait()

	live := out[:0]
	var failed []error
	for index, tunnel := range out {
		if errs[index] != nil {
			failed = append(failed, errs[index])
			continue
		}
		live = append(live, tunnel)
	}
	// A partial fleet is a working fleet. One box being down for a month is
	// the ordinary case, and refusing to run without it would mean never
	// running at all.
	if len(live) == 0 && len(failed) > 0 {
		return nil, errors.Join(failed...)
	}
	for _, err := range failed {
		s.logf("%v", err)
	}
	return live, nil
}

func (s *Supervisor) start(ctx context.Context, link Link) (Tunnel, error) {
	// A port that is already answering is somebody else's tunnel, or this
	// command run twice. Starting a second ssh on it fails in a way that reads
	// as a broken host, so check first and adopt what is there.
	if link.Check != nil {
		probe, cancel := context.WithTimeout(ctx, 3*time.Second)
		err := link.Check(probe)
		cancel()
		if err == nil {
			s.logf("%s is already up on 127.0.0.1:%d", link.Route, link.LocalPort)
			return Tunnel{Host: link.Host, Route: link.Route, LocalPort: link.LocalPort,
				RemotePort: link.RemotePort, Started: time.Now().UTC()}, nil
		}
	}

	process, err := s.launch(link)
	if err != nil {
		return Tunnel{}, fmt.Errorf("%s: start tunnel: %w", link.Route, err)
	}
	s.mu.Lock()
	if s.running == nil {
		s.running = map[string]Process{}
	}
	s.running[link.Route] = process
	s.mu.Unlock()

	if err := s.waitHealthy(ctx, link); err != nil {
		_ = process.Kill()
		s.mu.Lock()
		delete(s.running, link.Route)
		s.mu.Unlock()
		return Tunnel{}, err
	}
	return Tunnel{Host: link.Host, Route: link.Route, LocalPort: link.LocalPort,
		RemotePort: link.RemotePort, PID: process.PID(), Started: time.Now().UTC()}, nil
}

func (s *Supervisor) launch(link Link) (Process, error) {
	if s.Start != nil {
		return s.Start(link)
	}
	binary := s.Binary
	if binary == "" {
		binary = "ssh"
	}
	cmd := exec.Command(binary, tunnelArgs(link)...)
	// Its own process group, so it outlives the command that started it and so
	// a stray Ctrl-C in the terminal does not take the fleet down with it.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &sshProcess{cmd: cmd}, nil
}

func tunnelArgs(link Link) []string {
	return []string{
		"-N", "-L", link.Forward(),
		"-o", "BatchMode=yes",
		"-o", "ExitOnForwardFailure=yes",
		// A tunnel that is up but dead is worse than one that is down, because
		// every call on it hangs for the full timeout. These make ssh notice a
		// silent connection in about a minute and exit, which the supervisor
		// sees and restarts.
		"-o", "ServerAliveInterval=20",
		"-o", "ServerAliveCountMax=3",
		// A tunnel gets its own connection. A host configured with
		// ControlMaster auto and ControlPersist hands the forward to whatever
		// shared master a previous probe left behind: the pid recorded here
		// belongs to a client that exits, bringing the fleet down kills
		// nothing, and the forward dies on its own some minutes after the last
		// ordinary ssh to that host. Measured, not guessed.
		"-o", "ControlMaster=no",
		"-o", "ControlPath=none",
		link.Host,
	}
}

type sshProcess struct {
	cmd  *exec.Cmd
	once sync.Once
	done chan error
}

func (p *sshProcess) PID() int { return p.cmd.Process.Pid }

func (p *sshProcess) Kill() error {
	// The whole group: ssh may have children, and killing only the leader
	// leaves the forward held open by something with no parent.
	if err := syscall.Kill(-p.cmd.Process.Pid, syscall.SIGTERM); err != nil {
		return p.cmd.Process.Kill()
	}
	return nil
}

func (p *sshProcess) Wait() error {
	p.once.Do(func() {
		p.done = make(chan error, 1)
		go func() { p.done <- p.cmd.Wait() }()
	})
	return <-p.done
}

func (s *Supervisor) waitHealthy(ctx context.Context, link Link) error {
	if link.Check == nil {
		// Nothing to ask. Give ssh a moment to bind the port and take its word.
		select {
		case <-ctx.Done():
			return gaveUp(link, ctx.Err())
		case <-time.After(time.Second):
			return nil
		}
	}
	deadline := time.Now().Add(UpDeadline)
	var last error
	for {
		probe, cancel := context.WithTimeout(ctx, 5*time.Second)
		last = link.Check(probe)
		cancel()
		if last == nil {
			return nil
		}
		if ctx.Err() != nil {
			return gaveUp(link, ctx.Err())
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("%s: no answer on 127.0.0.1:%d after %s: %w",
				link.Route, link.LocalPort, UpDeadline, last)
		}
		select {
		case <-ctx.Done():
			return gaveUp(link, ctx.Err())
		case <-time.After(time.Second):
		}
	}
}

// gaveUp names the host in a cancellation.
//
// A bare "context deadline exceeded" in the log of a fleet coming up says that
// something did not start and not which box it was, which is the one thing the
// line is read for.
func gaveUp(link Link, err error) error {
	return fmt.Errorf("%s: gave up waiting for 127.0.0.1:%d: %w", link.Route, link.LocalPort, err)
}

// Watch keeps the links up until the context is cancelled. It is what a
// long run holds open in the background.
func (s *Supervisor) Watch(ctx context.Context, links []Link, poll time.Duration) {
	if poll <= 0 {
		poll = PollInterval
	}
	var group sync.WaitGroup
	for _, link := range links {
		group.Go(func() { s.watchOne(ctx, link, poll) })
	}
	group.Wait()
}

func (s *Supervisor) watchOne(ctx context.Context, link Link, poll time.Duration) {
	failures, backoff := 0, MinBackoff
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(poll):
		}
		if link.Check == nil {
			continue
		}
		probe, cancel := context.WithTimeout(ctx, 10*time.Second)
		err := link.Check(probe)
		cancel()
		if err == nil {
			if failures > 0 {
				s.logf("%s is back", link.Route)
			}
			failures, backoff = 0, MinBackoff
			continue
		}
		failures++
		s.logf("%s failed check %d: %v", link.Route, failures, err)
		if failures < FailuresBeforeRestart {
			continue
		}
		s.stop(link.Route)
		if _, err := s.start(ctx, link); err != nil {
			s.logf("%s: restart failed, waiting %s: %v", link.Route, backoff, err)
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
			backoff = min(backoff*2, MaxBackoff)
			continue
		}
		s.logf("%s restarted", link.Route)
		failures, backoff = 0, MinBackoff
	}
}

func (s *Supervisor) stop(name string) {
	s.mu.Lock()
	process, ok := s.running[name]
	delete(s.running, name)
	s.mu.Unlock()
	if ok {
		_ = process.Kill()
	}
}

// Down kills every tunnel this process started, then anything left over from
// an earlier run that the state file remembers.
func (s *Supervisor) Down(tunnels []Tunnel) []error {
	var errs []error
	s.mu.Lock()
	running := s.running
	s.running = nil
	s.mu.Unlock()
	for name, process := range running {
		if err := process.Kill(); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", name, err))
		}
	}
	for _, tunnel := range tunnels {
		if tunnel.PID <= 0 {
			continue
		}
		if _, ok := running[tunnel.Route]; ok {
			continue
		}
		if err := Kill(tunnel.PID); err != nil {
			errs = append(errs, fmt.Errorf("%s (pid %d): %w", tunnel.Route, tunnel.PID, err))
		}
	}
	return errs
}

// Kill stops a process this tool started in an earlier run. A pid that is gone
// is the desired state, so it is not an error.
func Kill(pid int) error {
	if pid <= 0 {
		return fmt.Errorf("no pid")
	}
	if err := syscall.Kill(-pid, syscall.SIGTERM); err == nil {
		return nil
	}
	process, err := os.FindProcess(pid)
	if err != nil {
		return nil
	}
	// ErrProcessDone is what the runtime says about a child it waited for;
	// ESRCH is what the kernel says about a pid this process never owned,
	// which is every pid read back out of the state file. Both mean the thing
	// this was asked to stop is already stopped.
	if err := process.Signal(syscall.SIGTERM); err != nil &&
		!errors.Is(err, os.ErrProcessDone) && !errors.Is(err, syscall.ESRCH) {
		return err
	}
	return nil
}

// Alive reports whether a pid from the state file is still running. Signal 0
// is the ordinary way to ask, and it answers for a process this shell does not
// own as long as it is ours.
func Alive(pid int) bool {
	if pid <= 0 {
		return false
	}
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return process.Signal(syscall.Signal(0)) == nil
}
