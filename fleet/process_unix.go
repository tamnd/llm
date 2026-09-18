//go:build !windows

package fleet

import (
	"os/exec"
	"syscall"
)

// ownGroup puts the child in its own process group, so it outlives the command
// that started it and so a stray Ctrl-C in the terminal does not take the fleet
// down with it.
func ownGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killGroup stops a whole process group. ssh may have children, and killing
// only the leader leaves the forward held open by something with no parent.
func killGroup(pid int) error {
	return syscall.Kill(-pid, syscall.SIGTERM)
}
