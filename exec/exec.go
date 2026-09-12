// Package exec is a route that is a program on this machine rather than an
// endpoint somewhere else.
//
// A subscription reached through its own CLI is a third transport beside a
// proxy and a gateway, and it has a different shape of limit: the boxes run
// out of uploads and turns, the free gateways answer 429 with fourteen hours
// on it, and a run spends most of its wall clock waiting for one or the other
// to come back. A subscription already paid for, spoken to from here with no
// browser, no rented box and no key in an environment variable, is not that.
//
// Runner implements llm.Completer, so the router cannot tell one from an HTTP
// route.
package exec

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/tamnd/llm"
	"github.com/tamnd/llm/route"
)

// DefaultTimeout is what one question is given before it is abandoned.
//
// Well over what a chunk takes — a full model has measured about forty seconds
// on six thousand characters and a cut down one about seventeen. The number is
// for the case where the CLI is waiting on a login or on a network that is not
// there, which is a thing to fail out of rather than to hang on.
const DefaultTimeout = 5 * time.Minute

// Runner is a program that answers a question on standard output.
type Runner struct {
	// Bin is the program, on PATH rather than at an absolute path, because a
	// CLI that updates itself makes a pinned path go stale.
	Bin string
	// Args is how to invoke it. "{{MODEL}}" in any argument is replaced by
	// Model. The prompt goes on standard input.
	Args  []string
	Model string
	// ImageFlag is the option the program takes a picture on, repeated once
	// per file, as in "--image". Empty for a program that reads no pictures,
	// which is the default, because a Runner that quietly dropped an image
	// would answer a question nobody asked.
	//
	// The pictures are written to temporary files and the flag and the path
	// go in front of Args. A CLI takes a path and not bytes on standard
	// input, and standard input is already carrying the prompt.
	ImageFlag string
	// Name is the route this is, for an error message.
	Name    string
	Timeout time.Duration
	// Parse reads the answer out of what the program printed. Both streams are
	// handed over, because the useful half is not always the same one.
	Parse func(stdout, stderr []byte) (llm.Response, error)
	// Run executes the program. It is a field so a test can run this without
	// the CLI, a subscription or a network.
	Run func(ctx context.Context, name string, args []string, stdin string) (stdout, stderr []byte, err error)
}

// Complete puts one question and returns the answer.
func (r *Runner) Complete(ctx context.Context, request llm.Request) (llm.Response, error) {
	if strings.TrimSpace(r.Bin) == "" {
		return llm.Response{}, fmt.Errorf("exec route %s names no program", r.name())
	}
	if len(request.Images) > 0 && r.ImageFlag == "" {
		// A route that cannot take a picture says so before the call rather
		// than failing inside it, or worse answering about the text alone.
		return llm.Response{}, fmt.Errorf("exec route %s cannot read an image", r.name())
	}
	model := strings.TrimSpace(request.Model)
	if model == "" {
		model = r.Model
	}
	if model == "" {
		return llm.Response{}, fmt.Errorf("exec route %s names no model", r.name())
	}
	// The prompt is sent byte for byte identical to what the other transports
	// send, instructions first. A prompt that differs per transport is an
	// answer that cannot be compared with another transport's, and the whole
	// point of having three is that the cheap one's mistakes get asked again
	// on the expensive one.
	prompt := strings.TrimSpace(request.Input)
	if instructions := strings.TrimSpace(request.Instructions); instructions != "" {
		prompt = instructions + "\n\n" + prompt
	}
	if prompt == "" {
		return llm.Response{}, fmt.Errorf("exec route %s: there is no question to ask", r.name())
	}

	timeout := r.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	paths, clean, err := spill(request.Images)
	if err != nil {
		return llm.Response{}, fmt.Errorf("exec route %s: %w", r.name(), err)
	}
	defer clean()

	// The pictures go in front of whatever Args says, because the last of
	// Args is the argument that means "the prompt is on standard input" and
	// an option after it is an option the CLI reads as the prompt.
	args := make([]string, 0, len(r.Args)+2*len(paths))
	for _, path := range paths {
		args = append(args, r.ImageFlag, path)
	}
	for _, arg := range r.Args {
		args = append(args, strings.ReplaceAll(arg, "{{MODEL}}", model))
	}

	started := time.Now()
	stdout, stderr, err := r.run(ctx, r.Bin, args, prompt)
	if err != nil {
		return llm.Response{}, fmt.Errorf("exec route %s: %w", r.name(), err)
	}
	parse := r.Parse
	if parse == nil {
		parse = PlainText
	}
	response, err := parse(stdout, stderr)
	if err != nil {
		return llm.Response{}, fmt.Errorf("exec route %s: %w", r.name(), err)
	}
	if strings.TrimSpace(response.Text) == "" {
		return llm.Response{}, fmt.Errorf("exec route %s answered with nothing", r.name())
	}
	if response.Model == "" {
		response.Model = model
	}
	if response.Route == "" {
		response.Route = r.Name
	}
	response.Elapsed = time.Since(started)
	return response, nil
}

// spill writes the pictures where the program can open them, and returns the
// paths and the way to take them away again.
//
// They go in one directory of their own so that the cleanup is one call that
// cannot leave a file behind, and they are numbered in the order they were
// asked about because a CLI passes them to the model in the order of the
// flags and a page read in the wrong order is a page read wrong.
func spill(images []llm.Image) (paths []string, clean func(), err error) {
	clean = func() {}
	if len(images) == 0 {
		return nil, clean, nil
	}
	dir, err := os.MkdirTemp("", "llm-exec-")
	if err != nil {
		return nil, clean, err
	}
	clean = func() { os.RemoveAll(dir) }
	for index, image := range images {
		if len(image.Data) == 0 {
			clean()
			return nil, func() {}, fmt.Errorf("image %d has no bytes in it", index+1)
		}
		path := filepath.Join(dir, fmt.Sprintf("%03d%s", index+1, suffix(image.MediaType)))
		if err := os.WriteFile(path, image.Data, 0o600); err != nil {
			clean()
			return nil, func() {}, err
		}
		paths = append(paths, path)
	}
	return paths, clean, nil
}

// suffix is the extension a media type is written under. A CLI works out what
// a file holds from its name as often as from its bytes, and a picture with
// no extension is a picture some of them will not send.
func suffix(mediaType string) string {
	switch strings.ToLower(strings.TrimSpace(mediaType)) {
	case "image/jpeg", "image/jpg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	case "image/gif":
		return ".gif"
	default:
		return ".png"
	}
}

func (r *Runner) name() string {
	if r.Name != "" {
		return r.Name
	}
	return r.Bin
}

func (r *Runner) run(ctx context.Context, name string, args []string, stdin string) ([]byte, []byte, error) {
	if r.Run != nil {
		return r.Run(ctx, name, args, stdin)
	}
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdin = strings.NewReader(stdin)
	var out, errOut strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errOut
	err := cmd.Run()
	stdout, stderr := []byte(out.String()), []byte(errOut.String())
	// Read standard output even when the exit status is not zero. A CLI
	// reports a refusal, a rate limit and a model it does not serve on
	// standard output as a line of its stream, and writes its human progress
	// to standard error. "exit status 1" says nothing; the parsed line says
	// exactly what happened.
	if err != nil && len(stdout) == 0 {
		if detail := strings.TrimSpace(errOut.String()); detail != "" {
			return stdout, stderr, fmt.Errorf("%w: %s", err, llm.Condense(detail))
		}
		return stdout, stderr, err
	}
	return stdout, stderr, nil
}

// PlainText is the parser for a program that simply prints its answer.
func PlainText(stdout, stderr []byte) (llm.Response, error) {
	text := strings.TrimSpace(string(stdout))
	if text == "" {
		if detail := strings.TrimSpace(string(stderr)); detail != "" {
			return llm.Response{}, fmt.Errorf("printed nothing: %s", llm.Condense(detail))
		}
	}
	return llm.Response{Text: text}, nil
}

// CodexBin is the CLI this package ships a parser for, and CodexImageFlag is
// the option it takes a picture on.
const (
	CodexBin       = "codex"
	CodexImageFlag = "--image"
)

// Codex returns a Runner for the codex CLI on the named model.
//
// It is a constructor in the library because the CLI is personal
// infrastructure shared across every project on a machine, and a parser for
// its output format is not something to write twice.
//
// The arguments: exec, so nothing interactive is started. read-only, so a
// question cannot write to the disk of the machine that asked it. The git
// check is skipped because the question has nothing to do with the directory
// it is asked in.
func Codex(name, model string) *Runner {
	return &Runner{
		Bin:   CodexBin,
		Args:  []string{"exec", "-m", "{{MODEL}}", "--json", "--skip-git-repo-check", "-s", "read-only", "-"},
		Model: model, Name: name, Timeout: DefaultTimeout,
		// codex exec takes --image once per file and reads the prompt off
		// standard input, which is the protocol spill and Complete write to.
		ImageFlag: CodexImageFlag,
		Parse:     ParseCodex,
	}
}

// ParseCodex picks the answer out of what the codex CLI printed.
//
// The CLI writes one JSON object a line: the thread starting, the turn
// starting, each item it completed, and the turn completing with what the turn
// cost. The answer is the last completed item of type agent_message — the last
// rather than the first, because a turn that thinks aloud completes more than
// one and the last is the one addressed to whoever asked.
//
// A line that does not parse is passed over rather than being an error. The
// CLI prints its own notices on this stream from time to time, and a notice is
// not a reason to throw away an answer sitting three lines below it.
//
// The first reason is kept and not the last. A turn that fails prints what the
// endpoint said and then prints that the turn failed, and the first of those
// is the sentence somebody can act on: the model is not one this account
// serves, or the account is out of turns until Tuesday. The second is "nope".
func ParseCodex(stdout, _ []byte) (llm.Response, error) {
	var response llm.Response
	var why string
	scanner := bufio.NewScanner(strings.NewReader(string(stdout)))
	// A chunk and its translation go on one line of this stream, and the
	// default buffer is sixty four kilobytes.
	scanner.Buffer(make([]byte, 0, 64*1024), 32*1024*1024)
	for scanner.Scan() {
		var event struct {
			Type    string `json:"type"`
			Message string `json:"message"`
			Item    struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"item"`
			ThreadID string `json:"thread_id"`
			Usage    struct {
				Input  int `json:"input_tokens"`
				Cached int `json:"cached_input_tokens"`
				Output int `json:"output_tokens"`
			} `json:"usage"`
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(scanner.Bytes(), &event) != nil {
			continue
		}
		switch {
		case event.Type == "thread.started" && response.ID == "":
			response.ID = event.ThreadID
		case event.Type == "item.completed" && event.Item.Type == "agent_message":
			response.Text = event.Item.Text
		case event.Type == "turn.completed":
			response.Usage = llm.Usage{
				InputTokens:       event.Usage.Input,
				CachedInputTokens: event.Usage.Cached,
				OutputTokens:      event.Usage.Output,
			}.Normalized()
		case event.Type == "error" && why == "":
			why = codexReason(event.Message)
		case event.Type == "turn.failed" && why == "":
			why = codexReason(event.Error.Message)
		}
	}
	if err := scanner.Err(); err != nil {
		return llm.Response{}, fmt.Errorf("read codex stream: %w", err)
	}
	// A turn that failed and then answered is not a failure. The CLI retries a
	// call it can retry, so an error line early in the stream can be followed
	// by a perfectly good answer, and only an error with nothing after it is
	// worth reporting.
	if response.Text != "" {
		why = ""
	}
	if why != "" {
		return llm.Response{}, fmt.Errorf("%s", why)
	}
	response.Text = strings.TrimSpace(response.Text)
	return response, nil
}

// Build makes a Runner for an exec route and nothing for any other kind,
// which is the shape route.Pool.Build wants:
//
//	pool.Build = exec.Build
//
// The parser is chosen by the program named in the route file, because that
// is the only thing that says what its output looks like. A program this
// package has no parser for gets PlainText, which is right for anything that
// simply prints its answer and wrong silently for anything that does not, so
// a route naming something exotic should set its own Runner rather than come
// through here.
func Build(r route.Route, timeout time.Duration, _ int) (llm.Completer, error) {
	if r.Kind != route.KindExec {
		return nil, nil
	}
	if r.Timeout > 0 {
		timeout = r.Timeout.Duration()
	}
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	runner := &Runner{
		Bin: r.Command, Args: r.Args, Model: r.Wire(), Name: r.Name, Timeout: timeout,
		ImageFlag: r.ImageFlag,
	}
	if len(runner.Args) == 0 && r.Command == CodexBin {
		codex := Codex(r.Name, r.Wire())
		codex.Timeout = timeout
		if r.ImageFlag != "" {
			codex.ImageFlag = r.ImageFlag
		}
		return codex, nil
	}
	if r.Command == CodexBin {
		runner.Parse = ParseCodex
		// A route that spells out its own arguments still gets the flag this
		// CLI takes pictures on, because the flag is a property of the
		// program and not of the arguments somebody chose.
		if runner.ImageFlag == "" {
			runner.ImageFlag = CodexImageFlag
		}
	}
	return runner, nil
}

// codexReason pulls the sentence out of the CLI's error, which is a JSON
// object printed inside the message field of another JSON object.
//
// The plain message is kept when it does not parse, since a sentence is what
// the caller is going to log either way and half a sentence is better than a
// blob of punctuation.
func codexReason(message string) string {
	message = strings.TrimSpace(message)
	if message == "" {
		return "the run failed and said nothing"
	}
	var inner struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(message), &inner) == nil && inner.Error.Message != "" {
		return inner.Error.Message
	}
	return llm.Condense(message)
}
