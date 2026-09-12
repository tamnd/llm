package exec

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tamnd/llm"
	"github.com/tamnd/llm/route"
)

// recorded is what the codex CLI prints for one answered turn: one JSON object
// a line. The text is invented; the shape is what the parser has to cope with,
// including a turn that thinks aloud before it answers and a notice the CLI
// printed in the middle of its own stream.
const recorded = `{"type":"thread.started","thread_id":"th_0001"}
{"type":"turn.started"}
{"type":"item.completed","item":{"type":"reasoning","text":"working it out"}}
{"type":"item.completed","item":{"type":"agent_message","text":"a first pass"}}
a notice that is not JSON
{"type":"item.completed","item":{"type":"agent_message","text":"the answer"}}
{"type":"turn.completed","usage":{"input_tokens":1200,"cached_input_tokens":200,"output_tokens":300}}
`

// replay hands back a recorded run and records what it was asked.
type replay struct {
	stdout, stderr string
	err            error

	name  string
	args  []string
	stdin string
}

func (r *replay) run(_ context.Context, name string, args []string, stdin string) ([]byte, []byte, error) {
	r.name, r.args, r.stdin = name, args, stdin
	return []byte(r.stdout), []byte(r.stderr), r.err
}

func codexRunner(out *replay) *Runner {
	runner := Codex("codex-full", "gpt-5.1-codex")
	runner.Run = out.run
	return runner
}

func TestCompleteReadsTheLastMessage(t *testing.T) {
	out := &replay{stdout: recorded}
	response, err := codexRunner(out).Complete(context.Background(),
		llm.Request{Instructions: "Translate faithfully.", Input: "a paragraph"})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	// A turn that thinks aloud completes more than one message and the last is
	// the one addressed to whoever asked.
	if response.Text != "the answer" {
		t.Errorf("text = %q", response.Text)
	}
	if response.ID != "th_0001" {
		t.Errorf("id = %q", response.ID)
	}
	if response.Usage.InputTokens != 1200 || response.Usage.OutputTokens != 300 {
		t.Errorf("usage = %+v", response.Usage)
	}
	if response.Usage.TotalTokens == 0 {
		t.Error("the total was not filled in")
	}
	if response.Route != "codex-full" || response.Model != "gpt-5.1-codex" {
		t.Errorf("answer came back as %s on %s", response.Route, response.Model)
	}
	if response.Elapsed <= 0 {
		t.Error("no elapsed time was recorded")
	}
}

// The prompt is sent byte for byte identical to what the other transports
// send, instructions first, or an answer cannot be compared with another
// transport's.
func TestCompleteSendsInstructionsFirstAndFillsTheModel(t *testing.T) {
	out := &replay{stdout: recorded}
	runner := codexRunner(out)
	if _, err := runner.Complete(context.Background(),
		llm.Request{Instructions: "Translate faithfully.", Input: "a paragraph", Model: "gpt-5.1-codex-mini"}); err != nil {
		t.Fatal(err)
	}
	if out.stdin != "Translate faithfully.\n\na paragraph" {
		t.Errorf("stdin = %q", out.stdin)
	}
	if out.name != CodexBin {
		t.Errorf("ran %q", out.name)
	}
	// {{MODEL}} is replaced by the request's model, which is what lets one
	// route file describe the cheap lane and the full one.
	joined := strings.Join(out.args, " ")
	if !strings.Contains(joined, "gpt-5.1-codex-mini") || strings.Contains(joined, "{{MODEL}}") {
		t.Errorf("args = %v", out.args)
	}
	// A question asked from here must not be able to write to this disk.
	for _, want := range []string{"exec", "--skip-git-repo-check", "read-only"} {
		if !strings.Contains(joined, want) {
			t.Errorf("args = %v, want %q in them", out.args, want)
		}
	}
}

// The CLI retries a call it can retry, so an error line early in the stream
// can be followed by a perfectly good answer.
func TestAnErrorFollowedByAnAnswerIsNotAFailure(t *testing.T) {
	out := &replay{stdout: `{"type":"error","message":"stream disconnected before completion"}
{"type":"item.completed","item":{"type":"agent_message","text":"the answer"}}
`}
	response, err := codexRunner(out).Complete(context.Background(), llm.Request{Input: "a paragraph"})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.Text != "the answer" {
		t.Errorf("text = %q", response.Text)
	}
}

// A turn that fails prints what the endpoint said and then prints that the
// turn failed. The first is the sentence somebody can act on; the second is
// "nope".
func TestTheFirstReasonIsKept(t *testing.T) {
	out := &replay{stdout: `{"type":"error","message":"{\"error\":{\"message\":\"usage_limit_reached: try again in 4 hours\"}}"}
{"type":"turn.failed","error":{"message":"turn failed"}}
`}
	_, err := codexRunner(out).Complete(context.Background(), llm.Request{Input: "a paragraph"})
	if err == nil {
		t.Fatal("a failed turn was reported as an answer")
	}
	if !strings.Contains(err.Error(), "try again in 4 hours") {
		t.Errorf("error = %q, want the sentence the endpoint gave and not \"turn failed\"", err)
	}
	if strings.Contains(err.Error(), "turn failed") {
		t.Errorf("error = %q, want the first reason and not the last", err)
	}
	if !strings.Contains(err.Error(), "codex-full") {
		t.Errorf("error = %q, want the route named", err)
	}
	// The classifier has to be able to see a quota in it, since this is what a
	// pool cools a route down on.
	if signal := llm.ClassifyError(err); signal.State != llm.StateQuota {
		t.Errorf("state = %q, want quota", signal.State)
	}
}

func TestTurnFailedAloneIsStillReported(t *testing.T) {
	out := &replay{stdout: `{"type":"turn.failed","error":{"message":"model not found"}}` + "\n"}
	_, err := codexRunner(out).Complete(context.Background(), llm.Request{Input: "x"})
	if err == nil || !strings.Contains(err.Error(), "model not found") {
		t.Errorf("err = %v", err)
	}
}

func TestAnEmptyStreamIsNotAnAnswer(t *testing.T) {
	out := &replay{stdout: `{"type":"thread.started","thread_id":"th_1"}` + "\n"}
	_, err := codexRunner(out).Complete(context.Background(), llm.Request{Input: "x"})
	if err == nil || !strings.Contains(err.Error(), "nothing") {
		t.Errorf("err = %v, want it to say the run answered with nothing", err)
	}
}

// A chunk and its translation go on one line of this stream, and the default
// scanner buffer is sixty four kilobytes.
func TestALongAnswerIsNotTruncated(t *testing.T) {
	long := strings.Repeat("a", 200_000)
	out := &replay{stdout: `{"type":"item.completed","item":{"type":"agent_message","text":"` + long + `"}}` + "\n"}
	response, err := codexRunner(out).Complete(context.Background(), llm.Request{Input: "x"})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(response.Text) != len(long) {
		t.Errorf("answer is %d bytes, want %d", len(response.Text), len(long))
	}
}

func TestCompleteRefusesWhatItCannotDo(t *testing.T) {
	out := &replay{stdout: recorded}
	for _, c := range []struct {
		name    string
		runner  *Runner
		request llm.Request
		want    string
	}{
		{"no program", &Runner{Name: "cli", Model: "m", Run: out.run}, llm.Request{Input: "x"}, "no program"},
		{"no model", &Runner{Name: "cli", Bin: "codex", Run: out.run}, llm.Request{Input: "x"}, "no model"},
		{"no question", codexRunner(out), llm.Request{Input: "   "}, "no question"},
		// A Runner with no ImageFlag names a program that reads no pictures.
		// It says so before the call rather than answering about the text
		// alone, which is a wrong answer and not an error.
		{"an image", &Runner{Name: "cli", Bin: "codex", Model: "m", Run: out.run},
			llm.Request{Input: "x", Images: []llm.Image{{Data: []byte{1}}}}, "cannot read an image"},
		{"an empty image", codexRunner(out),
			llm.Request{Input: "x", Images: []llm.Image{{MediaType: "image/png"}}}, "no bytes in it"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := c.runner.Complete(context.Background(), c.request)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("err = %v, want %q in it", err, c.want)
			}
		})
	}
}

// The CLI waiting on a login or on a network that is not there is a thing to
// fail out of rather than to hang on.
func TestCompleteGivesUpAtItsTimeout(t *testing.T) {
	runner := Codex("codex-full", "m")
	runner.Timeout = 20 * time.Millisecond
	runner.Run = func(ctx context.Context, _ string, _ []string, _ string) ([]byte, []byte, error) {
		<-ctx.Done()
		return nil, nil, ctx.Err()
	}
	started := time.Now()
	if _, err := runner.Complete(context.Background(), llm.Request{Input: "x"}); err == nil {
		t.Fatal("a hung CLI answered")
	}
	if took := time.Since(started); took > 5*time.Second {
		t.Errorf("waited %s on a 20ms timeout", took)
	}
}

// The real path, with a shell standing in for the CLI. A program that reports
// a refusal on standard output and exits non-zero is a program whose standard
// output is the useful half: "exit status 1" says nothing.
func TestNonZeroExitWithAUsefulStdout(t *testing.T) {
	runner := &Runner{
		Bin: "/bin/sh", Name: "cli", Model: "m", Timeout: 10 * time.Second,
		Args:  []string{"-c", `printf '{"type":"item.completed","item":{"type":"agent_message","text":"the answer"}}\n'; printf 'progress\n' >&2; exit 1`},
		Parse: ParseCodex,
	}
	response, err := runner.Complete(context.Background(), llm.Request{Input: "x"})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if response.Text != "the answer" {
		t.Errorf("text = %q", response.Text)
	}
}

// Nothing on standard output and a non-zero exit is the case where standard
// error is all there is, and one condensed line of it beats the status code.
func TestNonZeroExitWithNothingUsefulReportsStderr(t *testing.T) {
	runner := &Runner{
		Bin: "/bin/sh", Name: "cli", Model: "m", Timeout: 10 * time.Second,
		Args: []string{"-c", `printf 'not logged in\nrun codex login\n' >&2; exit 1`},
	}
	_, err := runner.Complete(context.Background(), llm.Request{Input: "x"})
	if err == nil {
		t.Fatal("a failed run was reported as an answer")
	}
	if !strings.Contains(err.Error(), "not logged in") {
		t.Errorf("error = %q, want what the program said", err)
	}
	if strings.Count(err.Error(), "\n") != 0 {
		t.Errorf("error is not one line: %q", err)
	}
}

func TestPlainText(t *testing.T) {
	response, err := PlainText([]byte("  the answer\n"), nil)
	if err != nil || response.Text != "the answer" {
		t.Errorf("PlainText = %q, %v", response.Text, err)
	}
	if _, err := PlainText(nil, []byte("could not start\nsecond line")); err == nil {
		t.Error("a silent program with a complaint on stderr was accepted")
	}
}

func TestBuildOnlyBuildsExecRoutes(t *testing.T) {
	gateway := route.Route{Name: "g", Kind: route.KindGateway, BaseURL: "http://x/v1", Model: "m"}
	client, err := Build(gateway, time.Minute, 1)
	if client != nil || err != nil {
		t.Errorf("Build of a %s route = %v, %v, want it declined", gateway.Kind, client, err)
	}

	r := route.Route{Name: "cli", Kind: route.KindExec, Command: CodexBin, Model: "gpt-5.1-codex",
		Timeout: route.Duration(90 * time.Second)}
	built, err := Build(r, time.Minute, 1)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	runner, ok := built.(*Runner)
	if !ok {
		t.Fatalf("Build gave %T", built)
	}
	// No args in the route file means the parser and the arguments this
	// package ships for the CLI it knows.
	if len(runner.Args) == 0 || runner.Parse == nil {
		t.Errorf("runner = %+v, want the codex defaults", runner)
	}
	if runner.Timeout != 90*time.Second {
		t.Errorf("timeout = %s, want the route's own", runner.Timeout)
	}
	if runner.Model != "gpt-5.1-codex" {
		t.Errorf("model = %q", runner.Model)
	}
}

// A route that names its own arguments gets them, and the wire name of the
// model rather than the local one.
func TestBuildKeepsTheRoutesOwnArguments(t *testing.T) {
	r := route.Route{Name: "cli", Kind: route.KindExec, Command: CodexBin,
		Args: []string{"exec", "-m", "{{MODEL}}", "-"}, Model: "codex", ServedModel: "gpt-5.1-codex"}
	built, err := Build(r, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	runner := built.(*Runner)
	if len(runner.Args) != 4 {
		t.Errorf("args = %v, want the file's", runner.Args)
	}
	if runner.Model != "gpt-5.1-codex" {
		t.Errorf("model = %q, want what the far end calls it", runner.Model)
	}
	if runner.Timeout != DefaultTimeout {
		t.Errorf("timeout = %s, want the default", runner.Timeout)
	}
}

// A picture goes to the program as a file, because that is what a CLI takes.
func TestAnImageIsWrittenWhereTheProgramCanOpenIt(t *testing.T) {
	out := &replay{stdout: recorded}
	var saw []string
	runner := codexRunner(out)
	runner.Run = func(ctx context.Context, name string, args []string, stdin string) ([]byte, []byte, error) {
		for i, arg := range args {
			if arg != "--image" || i+1 >= len(args) {
				continue
			}
			b, err := os.ReadFile(args[i+1])
			if err != nil {
				t.Errorf("the picture is not where the program was told: %v", err)
				continue
			}
			saw = append(saw, args[i+1]+"="+string(b))
		}
		return out.run(ctx, name, args, stdin)
	}
	_, err := runner.Complete(context.Background(), llm.Request{
		Input:  "read these",
		Images: []llm.Image{{MediaType: "image/png", Data: []byte("first")}, {MediaType: "image/jpeg", Data: []byte("second")}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(saw) != 2 {
		t.Fatalf("the program was given %d pictures, want 2: %v", len(saw), saw)
	}
	// In the order they were asked about, because that is the order the CLI
	// hands them to the model.
	if !strings.HasSuffix(saw[0], "=first") || !strings.HasSuffix(saw[1], "=second") {
		t.Errorf("the pictures came out in the wrong order: %v", saw)
	}
	// Named for what they hold, because a CLI works out what a file is from
	// its name as often as from its bytes.
	if !strings.Contains(saw[0], ".png=") || !strings.Contains(saw[1], ".jpg=") {
		t.Errorf("the pictures are not named for what they hold: %v", saw)
	}
}

// The pictures go in front of the argument that means "the prompt is on
// standard input", because an option after that one is read as the prompt.
func TestThePicturesComeBeforeTheRestOfTheCommandLine(t *testing.T) {
	out := &replay{stdout: recorded}
	runner := codexRunner(out)
	if _, err := runner.Complete(context.Background(), llm.Request{
		Input:  "read this",
		Images: []llm.Image{{MediaType: "image/png", Data: []byte("a page")}},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if out.args[0] != "--image" {
		t.Errorf("the command line starts %v", out.args[:min(4, len(out.args))])
	}
	if out.args[len(out.args)-1] != "-" {
		t.Errorf("the command line ends %v", out.args[len(out.args)-3:])
	}
}

// Nothing is left on the disk of the machine that asked.
func TestThePicturesAreTakenAwayAfterwards(t *testing.T) {
	out := &replay{stdout: recorded}
	var path string
	runner := codexRunner(out)
	runner.Run = func(ctx context.Context, name string, args []string, stdin string) ([]byte, []byte, error) {
		for i, arg := range args {
			if arg == "--image" && i+1 < len(args) {
				path = args[i+1]
			}
		}
		return out.run(ctx, name, args, stdin)
	}
	if _, err := runner.Complete(context.Background(), llm.Request{
		Input:  "read this",
		Images: []llm.Image{{MediaType: "image/png", Data: []byte("a page")}},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if path == "" {
		t.Fatal("the program was given no picture")
	}
	if _, err := os.Stat(filepath.Dir(path)); !os.IsNotExist(err) {
		t.Errorf("%s is still there", filepath.Dir(path))
	}
}

// An exec route named in a route file gets the flag the program takes
// pictures on, whether or not the file spells out the arguments.
func TestBuildGivesACodexRouteItsEye(t *testing.T) {
	for _, c := range []struct {
		name  string
		route route.Route
		want  string
	}{
		{"the default arguments", route.Route{Name: "cli", Kind: route.KindExec, Command: "codex", Model: "m"}, CodexImageFlag},
		{"arguments of its own", route.Route{Name: "cli", Kind: route.KindExec, Command: "codex", Model: "m",
			Args: []string{"exec", "-"}}, CodexImageFlag},
		{"a flag of its own", route.Route{Name: "cli", Kind: route.KindExec, Command: "codex", Model: "m",
			ImageFlag: "-i"}, "-i"},
		{"a program nobody here knows", route.Route{Name: "cli", Kind: route.KindExec, Command: "reader", Model: "m"}, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			completer, err := Build(c.route, time.Minute, 1)
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			runner, ok := completer.(*Runner)
			if !ok {
				t.Fatalf("Build returned %T", completer)
			}
			if runner.ImageFlag != c.want {
				t.Errorf("image flag = %q, want %q", runner.ImageFlag, c.want)
			}
		})
	}
}
