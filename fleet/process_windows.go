//go:build windows

package fleet

import (
	"errors"
	"os/exec"
	"syscall"
)

// The same two operations, on the platform that does not have them.
//
// Windows has no process groups in the POSIX sense and no way to signal one,
// so the supervisor here is weaker than it is elsewhere and this file says how
// rather than pretending otherwise. A tunnel started on Windows gets the
// closest thing available, a new console process group, which keeps a Ctrl-C
// in the terminal from reaching it. Stopping one falls back to killing the
// process itself, and an ssh that left a child behind will leave it behind.
//
// This exists because a package that does not compile on Windows makes every
// package importing it not compile there either, which is a large cost for a
// feature nobody on that platform is asking for.

// ownGroup puts the child in a console group of its own.
func ownGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
}

// errNoGroups is returned rather than a nil error, so the caller takes its
// fallback path and kills the process it actually started.
var errNoGroups = errors.New("windows has no process group to signal")

func killGroup(int) error { return errNoGroups }
