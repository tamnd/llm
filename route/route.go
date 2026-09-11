// Package route names the hosts a run can send model calls to, orders them,
// and keeps the run moving when one of them stops answering.
//
// A single base URL is fine until the account behind it hits its daily limit,
// which on a browser session is a routine event rather than an exceptional
// one. Naming the hosts makes the fallback order something you can read and
// argue with instead of something implicit in a shell script.
package route

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/tamnd/llm"
)

// Kind is what sort of thing is at the other end.
//
// This started life in the library it was extracted from as three separate
// fields — a Gateway bool, a Command string and a Reader string — each added
// one at a time as a correct local change, until deciding whether a route
// could be asked a question meant reasoning about which combination of the
// three was set. The set was a taxonomy nobody had written down. This is it
// written down.
type Kind string

const (
	// KindPool is a proxy fronting a pool of browser sessions. It answers
	// /health with the size of its pool, which is how a host with nothing left
	// to answer with is caught before a document is sent to it.
	KindPool Kind = "pool"
	// KindGateway is a plain OpenAI-compatible endpoint. No /health, no pool,
	// and asking for either returns 404 and reads as broken when it is fine.
	KindGateway Kind = "gateway"
	// KindExec is a program on this machine, run rather than called.
	KindExec Kind = "exec"
	// KindReader reads images over ssh and answers no questions. It never
	// enters the ask pool.
	KindReader Kind = "reader"
	// KindDirect is a local model server: vLLM, ollama, llama.cpp. Neither a
	// gateway (no key, no allowance, and it does answer /v1/models) nor a
	// reader (it answers questions fine).
	KindDirect Kind = "direct"
)

// Kinds lists every kind, for a help string and for validation.
func Kinds() []Kind { return []Kind{KindPool, KindGateway, KindExec, KindReader, KindDirect} }

// Valid reports whether a kind is one this package knows.
func (k Kind) Valid() bool { return slices.Contains(Kinds(), k) }

// Answers says whether this kind can be asked a question.
//
// A reader cannot. Left in the pool it takes its turn, finds nothing to call,
// and fails every ask routed to it until the pool marks it dead — a slow way
// to learn something the route file already says.
func (k Kind) Answers() bool { return k != KindReader }

// HasHealth says whether GET /health is worth asking. Only a session pool
// serves it; a gateway 404s, and a route reported broken for that is a route
// nobody sends work to for no reason.
func (k Kind) HasHealth() bool { return k == KindPool }

// OverHTTP says whether this route is reached by making a request at all.
func (k Kind) OverHTTP() bool {
	return k == KindPool || k == KindGateway || k == KindDirect
}

// OverSSH says whether reaching this route needs a tunnel or a remote command.
func (k Kind) OverSSH() bool { return k == KindPool || k == KindReader }

// Route is one host the fleet can send work to.
type Route struct {
	Name string `json:"name"`
	Kind Kind   `json:"kind"`
	// BaseURL may stop at the server root or end at /v1.
	BaseURL string `json:"base_url,omitempty"`
	Model   string `json:"model"`

	// APIKeyEnv names the environment variable holding the key, so a route
	// file can be committed or shared without carrying the secret itself.
	APIKeyEnv string `json:"api_key_env,omitempty"`
	// APIKey is a literal key passed on the command line. It never round trips
	// through a route file, because a file meant to be shared would leak it
	// the first time somebody pasted one.
	APIKey string `json:"-"`

	// Command is the program to run, for an exec route, and Args is how to
	// invoke it. "{{MODEL}}" in an argument is replaced by Model.
	Command string   `json:"command,omitempty"`
	Args    []string `json:"args,omitempty"`

	// Host is the ssh destination. RemotePort and LocalPort describe the
	// tunnel: a forwarder maps LocalPort here to RemotePort there, and BaseURL
	// is expected to name LocalPort on the loopback.
	Host       string `json:"host,omitempty"`
	RemotePort int    `json:"remote_port,omitempty"`
	LocalPort  int    `json:"local_port,omitempty"`

	// Reader is the program at the far end of the ssh connection, for a reader
	// route. ReaderURL is where that program's own model server answers, on
	// the box itself, and it is not BaseURL: BaseURL means this route answers
	// questions, and a reader does not. Two URLs because they are two claims.
	Reader    string `json:"reader,omitempty"`
	ReaderURL string `json:"reader_url,omitempty"`
	// ServedModel is what the far model server calls the weights on the wire,
	// which is not what the corpus calls them. Model is the slug that goes
	// into a document's front matter and has to name the weights so a reading
	// can be reproduced; the server answers to whatever its own shortlist
	// entry is called. Sending the corpus slug over the wire gets a 404;
	// sending the served name into the front matter records a page as read by
	// something that names nothing a year from now.
	ServedModel string `json:"served_model,omitempty"`

	// Rank orders the routes, lowest first. It is not a quality score, it is
	// the order to try things in.
	Rank int `json:"rank"`
	// Concurrency is how many calls this host will take at once. It should be
	// measured, not guessed.
	Concurrency int      `json:"concurrency,omitempty"`
	Timeout     Duration `json:"timeout,omitempty"`
	// Vision says this route will accept an image.
	//
	// Declared, not discovered. Measured on one free gateway: every model
	// there answers a request carrying an image with "404 No endpoints found
	// that support image input". Finding that out costs a call, and a call is
	// the scarce thing. So the route file says, a route that does not declare
	// vision is never sent an image, and a caller asking a registry with none
	// for a vision route gets a sentence rather than a 404 an hour into a run.
	Vision   bool `json:"vision,omitempty"`
	Disabled bool `json:"disabled,omitempty"`
	// Note carries why a route is ranked or disabled where it is. A disabled
	// row with no explanation reads as an oversight.
	Note string `json:"note,omitempty"`
}

// Duration is a time.Duration that round trips through JSON as "20m" rather
// than as a nanosecond count nobody can read.
type Duration time.Duration

func (d Duration) Duration() time.Duration { return time.Duration(d) }

func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(time.Duration(d).String())
}

func (d *Duration) UnmarshalJSON(raw []byte) error {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		value, err := time.ParseDuration(text)
		if err != nil {
			return fmt.Errorf("parse duration %q: %w", text, err)
		}
		*d = Duration(value)
		return nil
	}
	// A bare number is read as seconds, which is what someone hand editing a
	// route file most likely meant.
	var seconds float64
	if err := json.Unmarshal(raw, &seconds); err != nil {
		return fmt.Errorf(`duration must be a string like "20m" or a number of seconds`)
	}
	*d = Duration(time.Duration(seconds * float64(time.Second)))
	return nil
}

// Validate reports what is missing rather than letting the route fail at its
// first call with something obscure.
func (r Route) Validate() error {
	if strings.TrimSpace(r.Name) == "" {
		return fmt.Errorf("route has no name")
	}
	if strings.TrimSpace(r.Model) == "" {
		return fmt.Errorf("route %s has no model", r.Name)
	}
	if !r.Kind.Valid() {
		return fmt.Errorf("route %s has unknown kind %q, want one of %s",
			r.Name, r.Kind, strings.Join(kindNames(), ", "))
	}
	switch r.Kind {
	case KindExec:
		if strings.TrimSpace(r.Command) == "" {
			return fmt.Errorf("route %s is an exec route with no command", r.Name)
		}
	case KindReader:
		if strings.TrimSpace(r.Host) == "" {
			return fmt.Errorf("route %s is a reader with no ssh host", r.Name)
		}
		if strings.TrimSpace(r.Reader) == "" {
			return fmt.Errorf("route %s is a reader with no reader program", r.Name)
		}
	default:
		if strings.TrimSpace(r.BaseURL) == "" {
			return fmt.Errorf("route %s has no base_url", r.Name)
		}
	}
	if r.Concurrency < 0 {
		return fmt.Errorf("route %s has negative concurrency", r.Name)
	}
	return nil
}

func kindNames() []string {
	out := make([]string, 0, len(Kinds()))
	for _, kind := range Kinds() {
		out = append(out, string(kind))
	}
	return out
}

// Answers says whether this route can be asked a question.
func (r Route) Answers() bool { return r.Kind.Answers() }

// Lanes is how many calls this route takes at once. A route file that omits it
// gets one, which is slow and correct, rather than unlimited, which is neither.
func (r Route) Lanes() int {
	if r.Concurrency > 0 {
		return r.Concurrency
	}
	return 1
}

// Key is the credential to send, preferring a literal one from the command
// line over the environment variable the route file names.
func (r Route) Key() string {
	if key := strings.TrimSpace(r.APIKey); key != "" {
		return key
	}
	if r.APIKeyEnv != "" {
		return strings.TrimSpace(os.Getenv(r.APIKeyEnv))
	}
	return ""
}

// Endpoint joins the base URL to a path, tolerating a base that already ends
// at /v1 and one that stops at the server root.
func (r Route) Endpoint(path string) string {
	base := strings.TrimRight(r.BaseURL, "/")
	if strings.HasSuffix(base, "/v1") {
		return base + path
	}
	return base + "/v1" + path
}

// Wire is the model slug to put on the wire, which is ServedModel when the far
// end has its own name for the weights.
func (r Route) Wire() string {
	if served := strings.TrimSpace(r.ServedModel); served != "" {
		return served
	}
	return r.Model
}

// DefaultTimeout is what a route with nothing set gets. A page through a
// browser session takes minutes, and anything resembling an HTTP default would
// cut every call short.
const DefaultTimeout = 20 * time.Minute

// Client builds the HTTP transport for a route. An exec route has no HTTP
// transport and is an error here; use llm/exec.
func (r Route) Client(timeout time.Duration, maxRetries int) (llm.Completer, error) {
	if err := r.Validate(); err != nil {
		return nil, err
	}
	if !r.Kind.OverHTTP() {
		return nil, fmt.Errorf("route %s is a %s route and is not called over HTTP", r.Name, r.Kind)
	}
	if r.Timeout > 0 {
		timeout = r.Timeout.Duration()
	}
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	return &llm.Client{
		URL:           r.Endpoint("/chat/completions"),
		APIKey:        r.Key(),
		HTTPClient:    &http.Client{Timeout: timeout},
		MaxRetries:    maxRetries,
		MaxRetryDelay: MaxRetryDelay,
		UserAgent:     llm.UserAgent(),
	}, nil
}

// MaxRetryDelay is the longest wait this library will sit out because a
// provider asked it to. See llm.DefaultMaxRetryDelay for why it is a minute.
const MaxRetryDelay = llm.DefaultMaxRetryDelay

// Registry is an ordered set of routes.
type Registry struct {
	Routes []Route `json:"routes"`
}

// DefaultPath is where the personal route file lives: <APP>_ROUTES if it is
// set, otherwise ~/.config/<app>/routes.json.
func DefaultPath() string {
	if value := llm.Env("ROUTES"); value != "" {
		return value
	}
	dir := llm.ConfigDir()
	if dir == "" {
		return ""
	}
	return filepath.Join(dir, "routes.json")
}

// Load reads a registry from disk.
func Load(path string) (Registry, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Registry{}, err
	}
	var value Registry
	if err := json.Unmarshal(raw, &value); err != nil {
		return Registry{}, fmt.Errorf("decode route file %s: %w", path, err)
	}
	if err := value.Validate(); err != nil {
		return Registry{}, fmt.Errorf("route file %s: %w", path, err)
	}
	value.sort()
	return value, nil
}

// LoadOrDefault reads the named file, falls back to the personal one, and
// finally to Default, which is empty. A file named explicitly and missing is
// an error, because ignoring it quietly would run the wrong hosts.
func LoadOrDefault(path string) (Registry, string, error) {
	if strings.TrimSpace(path) != "" {
		registry, err := Load(path)
		return registry, path, err
	}
	if personal := DefaultPath(); personal != "" {
		registry, err := Load(personal)
		if err == nil {
			return registry, personal, nil
		}
		if !os.IsNotExist(err) {
			return Registry{}, personal, err
		}
	}
	return Default(), "built-in", nil
}

// Write saves a registry for editing.
func (r Registry) Write(path string) error {
	if directory := filepath.Dir(path); directory != "." {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			return err
		}
	}
	raw, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	// A route file names hosts and key variables. It is not a secret but it is
	// not everyone's business either.
	return os.WriteFile(path, append(raw, '\n'), 0o600)
}

func (r Registry) Validate() error {
	if len(r.Routes) == 0 {
		return fmt.Errorf("route file lists no routes")
	}
	seen := map[string]bool{}
	ports := map[int]string{}
	for _, value := range r.Routes {
		if err := value.Validate(); err != nil {
			return err
		}
		if seen[value.Name] {
			return fmt.Errorf("route %s is listed twice", value.Name)
		}
		seen[value.Name] = true
		// Two tunnels on one local port is not a typo you find later. The
		// second forward fails, the first keeps serving, and every call meant
		// for the second host quietly lands on the first.
		if value.LocalPort > 0 {
			if other, ok := ports[value.LocalPort]; ok {
				return fmt.Errorf("routes %s and %s both want local port %d", other, value.Name, value.LocalPort)
			}
			ports[value.LocalPort] = value.Name
		}
	}
	return nil
}

func (r *Registry) sort() {
	slices.SortStableFunc(r.Routes, func(a, b Route) int { return a.Rank - b.Rank })
}

// Enabled returns the usable routes in rank order. It sorts rather than
// trusting the caller to have done it, because a registry built in code is
// easy to write out of order and the resulting failover would be silently
// wrong.
func (r Registry) Enabled() []Route {
	var out []Route
	for _, value := range r.Routes {
		if !value.Disabled {
			out = append(out, value)
		}
	}
	slices.SortStableFunc(out, func(a, b Route) int { return a.Rank - b.Rank })
	return out
}

// Answering returns the enabled routes that can be asked a question.
func (r Registry) Answering() []Route {
	var out []Route
	for _, value := range r.Enabled() {
		if value.Answers() {
			out = append(out, value)
		}
	}
	return out
}

// Seeing returns the enabled routes that will take an image.
func (r Registry) Seeing() []Route {
	var out []Route
	for _, value := range r.Enabled() {
		if value.Vision {
			out = append(out, value)
		}
	}
	return out
}

// Find returns the route with the given name.
func (r Registry) Find(name string) (Route, bool) {
	for _, value := range r.Routes {
		if value.Name == name {
			return value, true
		}
	}
	return Route{}, false
}

// Select restricts the registry to the named routes, in the order given. The
// caller's order beats rank, because naming routes explicitly is a statement
// about what to try first, and a named route that the file disables is
// included, since naming it is the override.
func (r Registry) Select(names []string) (Registry, error) {
	var out Registry
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		value, ok := r.Find(name)
		if !ok {
			return Registry{}, fmt.Errorf("unknown route %q, have %s", name, strings.Join(r.Names(), ", "))
		}
		value.Disabled = false
		out.Routes = append(out.Routes, value)
	}
	if len(out.Routes) == 0 {
		return Registry{}, fmt.Errorf("no routes selected")
	}
	return out, nil
}

// Names lists every route name in file order.
func (r Registry) Names() []string {
	out := make([]string, 0, len(r.Routes))
	for _, value := range r.Routes {
		out = append(out, value.Name)
	}
	return out
}

// Default is empty, on purpose.
//
// The library this was extracted from shipped a Default() naming real hosts,
// real ports and a measured fleet, which was right for one private repository
// and is wrong for a public library: the fleet is personal infrastructure and
// nobody else's machine has it. What is general is the shape, and that is
// Suggest.
func Default() Registry { return Registry{} }

// ErrNoRoutes is what a caller gets for an empty registry, phrased so the
// person reading it knows what to do next.
//
// It is a function and not a var because the path depends on the application
// name, and a package-level var would be computed at init, before the caller
// has had a chance to call llm.Configure.
func ErrNoRoutes() error {
	return fmt.Errorf("no routes configured: write %s, or run %s routes init",
		DefaultPath(), llm.App())
}

// Suggest returns one template per transport, with the addresses, ports, model
// slugs and ranks left for a caller to fill in. It is the starting point for a
// routes init command, not a working configuration: every template fails
// Validate until it is edited, which is the intended behaviour.
func Suggest() Registry {
	return Registry{Routes: []Route{
		{
			Name: "local-server", Kind: KindDirect,
			BaseURL: "http://127.0.0.1:8000/v1", Model: "", ServedModel: "",
			Rank: 5, Concurrency: 4, Timeout: Duration(20 * time.Minute), Vision: true,
			Note: "a model server on this machine or the next desk: vLLM, ollama, llama.cpp",
		},
		{
			Name: "box", Kind: KindPool, Host: "",
			BaseURL: "http://127.0.0.1:18771/v1", Model: "",
			APIKeyEnv: llm.EnvName("PROXY_KEY"), RemotePort: 0, LocalPort: 18771,
			Rank: 10, Concurrency: 1, Timeout: Duration(20 * time.Minute),
			Note: "a proxy fronting browser sessions, over an ssh tunnel; set host and remote_port",
		},
		{
			Name: "gateway", Kind: KindGateway,
			BaseURL: "", Model: "", APIKeyEnv: "",
			Rank: 40, Concurrency: 2, Timeout: Duration(5 * time.Minute),
			Note: "a plain OpenAI-compatible endpoint; one route per model, so one allowance per route",
		},
		{
			Name: "cli", Kind: KindExec, Command: "codex", Model: "",
			Rank: 100, Concurrency: 2, Timeout: Duration(5 * time.Minute),
			Note: "a subscription reached by running a program on this machine",
		},
		{
			Name: "reader", Kind: KindReader, Host: "", Reader: "",
			ReaderURL: "http://127.0.0.1:8801/v1", Model: "", ServedModel: "",
			Rank: 5, Concurrency: 8, Timeout: Duration(20 * time.Minute), Vision: true,
			Note: "reads page images over ssh and answers no questions",
		},
	}}
}
