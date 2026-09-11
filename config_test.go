package llm

import (
	"path/filepath"
	"testing"
)

func TestConfigureNamesEverything(t *testing.T) {
	t.Cleanup(func() { Configure(Config{}) })
	Configure(Config{App: "papers"})

	if App() != "papers" {
		t.Errorf("app = %q", App())
	}
	if EnvName("ROUTES") != "PAPERS_ROUTES" {
		t.Errorf("env name = %q", EnvName("ROUTES"))
	}
	if UserAgent() != "papers/llm" {
		t.Errorf("user agent = %q", UserAgent())
	}
	if base := filepath.Base(ConfigDir()); base != "papers" {
		t.Errorf("config dir = %q, want it under the app's own name", ConfigDir())
	}
}

// A cosmetic field is not worth a panic at init: a library that dies over a
// capital letter breaks a caller's --help.
func TestConfigureSanitizes(t *testing.T) {
	t.Cleanup(func() { Configure(Config{}) })
	for _, c := range []struct{ in, want string }{
		{"Papers", "papers"},
		{"papers reader", "papers-reader"},
		{"papers_reader", "papers-reader"},
		{"  bourbaki  ", "bourbaki"},
		{"", "llm"},
		{"!!!", "llm"},
		{"9lives", "llm"}, // a name must start with a letter to be a path and a variable
	} {
		Configure(Config{App: c.in})
		if App() != c.want {
			t.Errorf("Configure(%q) gave %q, want %q", c.in, App(), c.want)
		}
	}
}

func TestConfigDirOverride(t *testing.T) {
	dir := t.TempDir()
	t.Cleanup(func() { Configure(Config{}) })
	Configure(Config{App: "papers", ConfigDir: dir})
	if ConfigDir() != dir {
		t.Errorf("config dir = %q, want %q", ConfigDir(), dir)
	}
}

// A key shared by every project on a machine should be set once. The
// app-scoped name wins where both are set, so one project can differ.
func TestEnvFallsBackToTheSharedName(t *testing.T) {
	t.Cleanup(func() { Configure(Config{}) })
	Configure(Config{App: "papers"})

	t.Setenv("LLM_PROXY_KEY", "shared")
	if Env("PROXY_KEY") != "shared" {
		t.Errorf("env = %q, want the shared value", Env("PROXY_KEY"))
	}
	t.Setenv("PAPERS_PROXY_KEY", "mine")
	if Env("PROXY_KEY") != "mine" {
		t.Errorf("env = %q, want the app's own value", Env("PROXY_KEY"))
	}
}
