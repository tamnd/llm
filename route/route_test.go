package route

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tamnd/llm"
)

func gateway(name string, rank int) Route {
	return Route{Name: name, Kind: KindGateway, BaseURL: "http://127.0.0.1:1/v1",
		Model: "test-model", Rank: rank}
}

func TestValidateSaysWhatIsMissing(t *testing.T) {
	for _, c := range []struct {
		name  string
		route Route
		ok    bool
	}{
		{"a gateway", gateway("g", 1), true},
		{"no name", Route{Kind: KindGateway, BaseURL: "http://x", Model: "m"}, false},
		{"no model", Route{Name: "g", Kind: KindGateway, BaseURL: "http://x"}, false},
		{"unknown kind", Route{Name: "g", Kind: "browser", BaseURL: "http://x", Model: "m"}, false},
		{"gateway with no url", Route{Name: "g", Kind: KindGateway, Model: "m"}, false},
		{"exec with a command", Route{Name: "c", Kind: KindExec, Command: "codex", Model: "m"}, true},
		{"exec with no command", Route{Name: "c", Kind: KindExec, Model: "m"}, false},
		{"reader", Route{Name: "r", Kind: KindReader, Host: "box", Reader: "local-ocr", Model: "m"}, true},
		{"reader with no host", Route{Name: "r", Kind: KindReader, Reader: "local-ocr", Model: "m"}, false},
		{"reader with no program", Route{Name: "r", Kind: KindReader, Host: "box", Model: "m"}, false},
		{"negative lanes", Route{Name: "g", Kind: KindGateway, BaseURL: "http://x", Model: "m", Concurrency: -1}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := c.route.Validate()
			if (err == nil) != c.ok {
				t.Errorf("Validate = %v, want ok = %t", err, c.ok)
			}
		})
	}
}

func TestKinds(t *testing.T) {
	// A reader answers no questions, which is why it never enters a pool, and
	// only a session proxy has a /health worth asking.
	if KindReader.Answers() {
		t.Error("a reader was asked to answer questions")
	}
	if !KindPool.HasHealth() || KindGateway.HasHealth() {
		t.Error("health is a pool's endpoint and nobody else's")
	}
	if !KindExec.Valid() || Kind("browser").Valid() {
		t.Error("Valid disagrees with Kinds")
	}
	if KindExec.OverHTTP() || !KindDirect.OverHTTP() {
		t.Error("OverHTTP is wrong about a transport")
	}
	if !KindReader.OverSSH() || KindGateway.OverSSH() {
		t.Error("OverSSH is wrong about a transport")
	}
}

func TestEndpointTolerantOfTheBase(t *testing.T) {
	for _, base := range []string{"http://h:1", "http://h:1/", "http://h:1/v1", "http://h:1/v1/"} {
		r := Route{BaseURL: base}
		if got := r.Endpoint("/models"); got != "http://h:1/v1/models" {
			t.Errorf("base %q gave %q", base, got)
		}
	}
}

func TestWirePrefersTheServedName(t *testing.T) {
	r := Route{Model: "olmocr-2", ServedModel: "allenai/olmOCR-2-7B"}
	if r.Wire() != "allenai/olmOCR-2-7B" {
		t.Errorf("wire = %q, want what the far end calls the weights", r.Wire())
	}
	if (Route{Model: "m"}).Wire() != "m" {
		t.Error("a route with one name for its model lost it")
	}
}

// A key is read from the environment by the name the file gives, and a literal
// key never round trips through a serialised route file.
func TestKeyComesFromTheEnvironment(t *testing.T) {
	t.Setenv("TEST_ROUTE_KEY", "from-the-env")
	r := Route{APIKeyEnv: "TEST_ROUTE_KEY"}
	if r.Key() != "from-the-env" {
		t.Errorf("key = %q", r.Key())
	}
	r.APIKey = "from-the-flag"
	if r.Key() != "from-the-flag" {
		t.Errorf("an explicit key was ignored: %q", r.Key())
	}
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if got := string(raw); strings.Contains(got, "from-the-flag") {
		t.Errorf("a literal key was written into the route file: %s", got)
	}
}

func TestDurationRoundTrips(t *testing.T) {
	raw, err := json.Marshal(Duration(20 * time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != `"20m0s"` {
		t.Errorf("marshalled as %s, want something a person can read", raw)
	}
	var d Duration
	if err := json.Unmarshal([]byte(`"90s"`), &d); err != nil || d.Duration() != 90*time.Second {
		t.Errorf("unmarshal string: %v %s", err, d.Duration())
	}
	// A bare number is what somebody hand editing the file most likely meant.
	if err := json.Unmarshal([]byte(`45`), &d); err != nil || d.Duration() != 45*time.Second {
		t.Errorf("unmarshal number: %v %s", err, d.Duration())
	}
	if err := json.Unmarshal([]byte(`"soon"`), &d); err == nil {
		t.Error("nonsense was accepted as a duration")
	}
}

func TestRegistryValidateCatchesDuplicates(t *testing.T) {
	twice := Registry{Routes: []Route{gateway("a", 1), gateway("a", 2)}}
	if err := twice.Validate(); err == nil {
		t.Error("two routes with one name was accepted")
	}
	// Two tunnels on one local port is not a typo you find later: the second
	// forward fails, the first keeps serving, and every call meant for the
	// second host lands on the first.
	a := Route{Name: "a", Kind: KindPool, Host: "one", BaseURL: "http://127.0.0.1:18771/v1",
		Model: "m", LocalPort: 18771}
	b := a
	b.Name, b.Host = "b", "two"
	if err := (Registry{Routes: []Route{a, b}}).Validate(); err == nil {
		t.Error("two routes on one local port was accepted")
	}
	if err := (Registry{}).Validate(); err == nil {
		t.Error("an empty route file was accepted")
	}
}

func TestRegistryOrdersByRank(t *testing.T) {
	registry := Registry{Routes: []Route{gateway("slow", 40), gateway("fast", 5), gateway("off", 1)}}
	registry.Routes[2].Disabled = true
	names := []string{}
	for _, r := range registry.Enabled() {
		names = append(names, r.Name)
	}
	if len(names) != 2 || names[0] != "fast" || names[1] != "slow" {
		t.Errorf("enabled = %v, want the cheapest first and the disabled one gone", names)
	}
}

func TestRegistrySelectBeatsRankAndEnablesWhatItNames(t *testing.T) {
	off := gateway("off", 1)
	off.Disabled = true
	registry := Registry{Routes: []Route{gateway("a", 1), gateway("b", 2), off}}

	picked, err := registry.Select([]string{"b", "off"})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if picked.Routes[0].Name != "b" || picked.Routes[1].Name != "off" {
		t.Errorf("order = %v, want the caller's", picked.Names())
	}
	// Naming a disabled route is the override.
	if picked.Routes[1].Disabled {
		t.Error("a route named explicitly stayed disabled")
	}
	if _, err := registry.Select([]string{"nope"}); err == nil {
		t.Error("an unknown route name was accepted")
	}
}

func TestAnsweringAndSeeing(t *testing.T) {
	reader := Route{Name: "r", Kind: KindReader, Host: "box", Reader: "local-ocr", Model: "m",
		Vision: true, Rank: 5}
	seeing := gateway("sees", 2)
	seeing.Vision = true
	registry := Registry{Routes: []Route{gateway("blind", 1), seeing, reader}}

	if names := names(registry.Answering()); len(names) != 2 {
		t.Errorf("answering = %v, want the reader left out", names)
	}
	if names := names(registry.Seeing()); len(names) != 2 || names[0] != "sees" {
		t.Errorf("seeing = %v", names)
	}
}

func names(routes []Route) []string {
	out := make([]string, 0, len(routes))
	for _, r := range routes {
		out = append(out, r.Name)
	}
	return out
}

func TestLoadAndWrite(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "routes.json")
	want := Registry{Routes: []Route{gateway("a", 10), gateway("b", 1)}}
	if err := want.Write(path); err != nil {
		t.Fatalf("Write: %v", err)
	}
	got, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	// Load sorts, so the file's order does not decide failover.
	if got.Routes[0].Name != "b" {
		t.Errorf("loaded order = %v", got.Names())
	}
	if _, err := Load(filepath.Join(dir, "missing.json")); err == nil {
		t.Error("a missing file named explicitly was not an error")
	}
}

// A public library must not ship somebody's private fleet. The default is
// empty and the templates are unfilled, so nothing here names a host, a port
// or a key.
func TestDefaultIsEmptyAndSuggestionsAreBlank(t *testing.T) {
	if len(Default().Routes) != 0 {
		t.Error("the built-in registry names routes")
	}
	for _, r := range Suggest().Routes {
		if r.Host != "" {
			t.Errorf("template %s names a host", r.Name)
		}
		if r.Model != "" {
			t.Errorf("template %s names a model", r.Name)
		}
		if r.APIKey != "" {
			t.Errorf("template %s carries a key", r.Name)
		}
		if err := r.Validate(); err == nil && r.Kind != KindExec {
			t.Errorf("template %s validates as it stands, so it is configuration and not a template", r.Name)
		}
	}
}

func TestErrNoRoutesNamesTheAppAndThePath(t *testing.T) {
	llm.Configure(llm.Config{App: "papers", ConfigDir: t.TempDir()})
	t.Cleanup(func() { llm.Configure(llm.Config{}) })
	// Computed at call time, not at init: a package level var would be built
	// before the caller had a chance to name the application.
	if got := ErrNoRoutes().Error(); !strings.Contains(got, "papers") {
		t.Errorf("error = %q, want the app's own name in it", got)
	}
}
