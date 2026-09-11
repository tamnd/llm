package llm

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

// Config is what a caller must tell the library about itself. There is one
// field because there is one thing the library cannot guess: which application
// this is.
//
// The name decides three paths and one cache prefix, every one of which was a
// literal in the code this was extracted from. Two projects sharing a proxy
// must not share its prompt cache, and two projects on one machine must not
// share a route file, because the routes are different and the second project
// to run would quietly inherit the first one's fleet.
type Config struct {
	// App is a short lowercase name: "papers", "bourbaki". It appears in
	// ~/.config/<app>/, in <APP>_ROUTES, in the cache key and in the user
	// agent, so it must be a plain identifier.
	App string
	// ConfigDir overrides ~/.config/<app>, for a test or for a caller that
	// keeps its configuration somewhere else.
	ConfigDir string
}

var (
	configMu sync.RWMutex
	config   = Config{App: "llm"}
)

var appPattern = regexp.MustCompile(`^[a-z][a-z0-9-]*$`)

// Configure sets the application name for the process. Call it once, early,
// before anything reads a route file.
//
// An invalid name is silently lowercased and stripped rather than rejected,
// because a library that panics at init over a cosmetic field is a library
// that breaks a caller's `--help`.
func Configure(c Config) {
	configMu.Lock()
	defer configMu.Unlock()
	name := sanitizeApp(c.App)
	if name == "" {
		name = "llm"
	}
	config = Config{App: name, ConfigDir: c.ConfigDir}
}

// App returns the configured application name.
func App() string {
	configMu.RLock()
	defer configMu.RUnlock()
	return config.App
}

// ConfigDir is where the route file and the fleet state live. It is
// $XDG_CONFIG_HOME/<app>, else ~/.config/<app>, and the current directory if
// there is no home, which is a bad answer but a better one than a path
// beginning with an empty string.
//
// It is ~/.config on macOS too, which os.UserConfigDir is not: there it
// answers ~/Library/Application Support. The platform convention loses here
// because a fleet is not one platform. The route file names Linux boxes, it
// is written on a laptop and read on a server, it sits beside the ssh config
// and the other dotfiles that go with it, and a person who has to remember
// which of two directories a file is in on which machine is a person who will
// eventually edit the wrong one.
func ConfigDir() string {
	configMu.RLock()
	dir, app := config.ConfigDir, config.App
	configMu.RUnlock()
	if dir != "" {
		return dir
	}
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		return filepath.Join(xdg, app)
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return filepath.Join(".config", app)
	}
	return filepath.Join(home, ".config", app)
}

// EnvName builds the environment variable for a suffix: EnvName("ROUTES") is
// PAPERS_ROUTES when the app is papers.
func EnvName(suffix string) string {
	app := strings.ToUpper(strings.ReplaceAll(App(), "-", "_"))
	return app + "_" + strings.ToUpper(suffix)
}

// Env reads an app-scoped variable, falling back to the LLM_-prefixed one. The
// fallback exists so a key shared by every project on this machine can be set
// once: LLM_PROXY_KEY works for all of them, PAPERS_PROXY_KEY overrides it for
// one.
func Env(suffix string) string {
	if value := strings.TrimSpace(os.Getenv(EnvName(suffix))); value != "" {
		return value
	}
	return strings.TrimSpace(os.Getenv("LLM_" + strings.ToUpper(suffix)))
}

// UserAgent identifies this library's traffic as the caller's.
func UserAgent() string { return App() + "/llm" }

func sanitizeApp(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	name = strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '-':
			return r
		case r == '_', r == ' ', r == '.':
			return '-'
		}
		return -1
	}, name)
	name = strings.Trim(name, "-")
	if !appPattern.MatchString(name) {
		return ""
	}
	return name
}
