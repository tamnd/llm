package fleet

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tamnd/llm"
)

// State is what bringing the fleet up leaves behind, so that a status command,
// a teardown and every other command in another process can find the tunnels.
//
// It is discovered fact and running state, never configuration. The
// configuration is the route file. Nothing here is worth keeping if it is
// stale, which is why it records when it was written rather than pretending.
type State struct {
	Written time.Time `json:"written"`
	Tunnels []Tunnel  `json:"tunnels,omitempty"`
	// Tools is the absolute path of the pool tool per host, which differs per
	// box and so has to be discovered rather than assumed.
	Tools map[string]string `json:"tools,omitempty"`
}

// Tunnel is one ssh port forward.
type Tunnel struct {
	Host       string    `json:"host"`
	Route      string    `json:"route"`
	LocalPort  int       `json:"local_port"`
	RemotePort int       `json:"remote_port"`
	PID        int       `json:"pid"`
	Started    time.Time `json:"started"`
}

// StatePath is where the running state lives: beside the route file, under
// the app's own config directory, and overridable with <APP>_FLEET_STATE.
func StatePath() string {
	if value := strings.TrimSpace(llm.Env("FLEET_STATE")); value != "" {
		return value
	}
	return filepath.Join(llm.ConfigDir(), "fleet.json")
}

// LoadState reads the state file. A missing file is not an error: it means no
// fleet has been brought up in this configuration yet.
func LoadState(path string) (State, error) {
	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return State{Tools: map[string]string{}}, nil
	}
	if err != nil {
		return State{}, err
	}
	var value State
	if err := json.Unmarshal(raw, &value); err != nil {
		return State{}, fmt.Errorf("decode %s: %w", path, err)
	}
	if value.Tools == nil {
		value.Tools = map[string]string{}
	}
	return value, nil
}

// Save writes the state atomically, because a status command in one terminal
// reading a half written file while another is writing it is a confusing
// failure to debug and a trivial one to prevent.
func (s State) Save(path string) error {
	if directory := filepath.Dir(path); directory != "." {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			return err
		}
	}
	raw, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	temp := path + ".tmp"
	if err := os.WriteFile(temp, append(raw, '\n'), 0o600); err != nil {
		return err
	}
	return os.Rename(temp, path)
}

// Find returns the tunnel for a route.
func (s State) Find(name string) (Tunnel, bool) {
	for _, value := range s.Tunnels {
		if value.Route == name {
			return value, true
		}
	}
	return Tunnel{}, false
}

// Stale reports the tunnels the state remembers whose process is gone, which
// is the difference between a fleet that is up and a file that says it is.
func (s State) Stale() []Tunnel {
	var out []Tunnel
	for _, value := range s.Tunnels {
		if value.PID > 0 && !Alive(value.PID) {
			out = append(out, value)
		}
	}
	return out
}
