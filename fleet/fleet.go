// Package fleet is the machines behind the routes: ssh to them, tunnels to
// their loopback listeners, and one question put to each — is there anything
// here that can take work right now.
//
// Every listener worth tunnelling to binds 127.0.0.1, which is correct and
// stays that way. The only path in is ssh, so ssh is the transport for
// everything here: the port forwards, the probes, and the batch work that does
// not go over HTTP at all.
//
// Nothing in this package names a host, a port, a user or a key. Those live in
// the route file and in the user's own ~/.ssh/config, which is where a
// personal fleet belongs and not in a library.
package fleet

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/tamnd/llm"
)

// Runner runs a command on a host. It is an interface because every test in
// this package would otherwise need rented Linux boxes.
type Runner interface {
	Run(ctx context.Context, host string, command string) (string, error)
}

// SSH runs commands with the ssh binary, using the user's own config.
//
// Reading ~/.ssh/config is the point: one box logs in as a user and another as
// root, the key has a name, and none of that belongs in a repository. A host
// here is whatever ssh calls it.
type SSH struct {
	// Binary is the ssh to run. Empty means the one on PATH.
	Binary string
	// Timeout bounds one command. It is not the tunnel's timeout; a tunnel runs
	// for hours.
	Timeout time.Duration
	// Options are extra -o flags. BatchMode is always set, because a command
	// that stops for a passphrase in an unattended run hangs until somebody
	// notices days later.
	Options []string
}

func (s SSH) binary() string {
	if strings.TrimSpace(s.Binary) != "" {
		return s.Binary
	}
	return "ssh"
}

func (s SSH) args(host string, rest ...string) []string {
	args := []string{"-o", "BatchMode=yes", "-o", "ConnectTimeout=10"}
	for _, option := range s.Options {
		args = append(args, "-o", option)
	}
	args = append(args, host)
	return append(args, rest...)
}

// Run executes a shell command on the host and returns its standard output.
//
// Standard error is folded into the error rather than dropped, because the one
// thing worth knowing when a remote command fails is what it printed.
func (s SSH) Run(ctx context.Context, host, command string) (string, error) {
	if strings.TrimSpace(host) == "" {
		return "", fmt.Errorf("no host")
	}
	if s.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.Timeout)
		defer cancel()
	}
	cmd := exec.CommandContext(ctx, s.binary(), s.args(host, command)...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	out := stdout.String()
	if err != nil {
		detail := strings.TrimSpace(stderr.String())
		if detail == "" {
			detail = strings.TrimSpace(out)
		}
		if ctx.Err() != nil {
			return out, fmt.Errorf("%s: %w after %s", host, ctx.Err(), s.Timeout)
		}
		return out, fmt.Errorf("%s: %s: %w", host, llm.Condense(detail), err)
	}
	return out, nil
}

// timedOut says the host never answered, as against answering something that
// could not be read.
//
// It matches on the message rather than on a sentinel because the runner is an
// interface and the ssh one wraps whatever the command did into text. Both
// wordings are here: the deadline this run set, and the one the far end's own
// TCP stack reports when a box is up but not listening.
func timedOut(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "deadline exceeded") ||
		strings.Contains(s, "context canceled") ||
		strings.Contains(s, "timed out") ||
		strings.Contains(s, "timeout")
}
