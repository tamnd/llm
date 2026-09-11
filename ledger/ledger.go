// Package ledger is one line per ask, appended and never rewritten.
//
// It exists because everything else in this module reports what a route or a
// host *has*, and none of it reports whether any of it *worked*. A route file
// says a model is configured. A health probe says an endpoint answered a GET.
// An account board says a host has ten free sessions. All three can be true of
// a route that has not produced a usable answer in a day.
//
// The number that made this a package: of 17,123 translation asks across one
// week of runs, between a third and a half a day were lost before the question
// was read at all — out of quota, refused, or signed out. Nothing in the route
// file, the probe or the board said so. Only the record of asks did, and it is
// the number that changed the chunk size from fifteen spans to sixty, because
// it turned the price of an ask from cheap into the scarce resource.
//
// The file is JSONL because a run appends to it from several lanes at once and
// a half written line should cost one line rather than the file. Reading skips
// what it cannot parse for the same reason: a record of a week of work is not
// worth throwing away over a line that was being written when the power went.
package ledger

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/tamnd/llm"
)

// Entry is one ask and how it ended.
//
// Stage and Target are the caller's words for what was being done and to
// what — "translate" and a section id, "ocr" and a page. The library does not
// interpret either; they are here so a report can group by them.
type Entry struct {
	TS     time.Time `json:"ts"`
	App    string    `json:"app"`
	Stage  string    `json:"stage,omitempty"`
	Target string    `json:"target,omitempty"`
	Route  string    `json:"route,omitempty"`
	Model  string    `json:"model,omitempty"`
	OK     bool      `json:"ok"`
	// State is why it failed, in the same vocabulary the router cools a route
	// down by, so that a day's failures can be counted as quota, refusal,
	// transport and bug rather than as one undifferentiated pile.
	State llm.State `json:"state,omitempty"`
	// Error is one condensed line. It is the far end's own words where there
	// are any, because a month later that sentence is the only thing that
	// says what went wrong.
	Error     string    `json:"error,omitempty"`
	ElapsedMS int64     `json:"elapsed_ms,omitempty"`
	Usage     llm.Usage `json:"usage,omitzero"`
	Attempt   int       `json:"attempt,omitempty"`
}

// Elapsed is how long the ask took.
func (e Entry) Elapsed() time.Duration { return time.Duration(e.ElapsedMS) * time.Millisecond }

// Refused reports an ask that never reached the model: the quota was spent,
// the key was rejected, or the route no longer serves what was asked of it.
//
// It is the distinction the whole package is for. A refusal is free, instant
// and invisible in every other report, and a day made of them looks from the
// outside exactly like a day of work.
func (e Entry) Refused() bool {
	switch e.State {
	case llm.StateQuota, llm.StateUnauthorized, llm.StateGone:
		return true
	default:
		return false
	}
}

// Log appends entries to a file.
//
// One os.File held open with O_APPEND, and one Write per line under a mutex.
// Append mode is what makes two processes writing the same file safe against
// each other; the mutex is what keeps two lanes in this process from
// interleaving halves of a line.
type Log struct {
	// App stamps every line, so that one file can hold more than one tool's
	// asks and a report can still tell them apart. Empty means llm.App().
	App string
	Now func() time.Time

	mu   sync.Mutex
	file *os.File
	path string
}

// DefaultPath is where the record lives: beside the route file, under the
// app's own config directory, and overridable with <APP>_LEDGER.
func DefaultPath() string {
	if value := strings.TrimSpace(llm.Env("LEDGER")); value != "" {
		return value
	}
	return filepath.Join(llm.ConfigDir(), "ledger.jsonl")
}

// Open opens the file for appending, creating it and its directory.
func Open(path string) (*Log, error) {
	if strings.TrimSpace(path) == "" {
		path = DefaultPath()
	}
	if directory := filepath.Dir(path); directory != "." {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			return nil, err
		}
	}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	return &Log{file: file, path: path}, nil
}

// Path is the file being written, for a report that says where it read from.
func (l *Log) Path() string { return l.path }

// Close closes the file. A nil Log closes nothing, so a caller that chose not
// to keep a record needs no branch around it.
func (l *Log) Close() error {
	if l == nil || l.file == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	err := l.file.Close()
	l.file = nil
	return err
}

// Write appends one entry.
//
// A nil Log writes nothing and returns nothing, because recording asks is
// optional and a caller should not have to check.
func (l *Log) Write(entry Entry) error {
	if l == nil || l.file == nil {
		return nil
	}
	if entry.TS.IsZero() {
		entry.TS = l.now()
	}
	entry.TS = entry.TS.UTC().Truncate(time.Second)
	if entry.App == "" {
		entry.App = l.app()
	}
	entry.Error = llm.Condense(entry.Error)
	raw, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	_, err = l.file.Write(append(raw, '\n'))
	return err
}

// Record is Write for the ordinary case: a route was asked, and it either
// answered or it did not.
//
// The failure is classified here rather than by the caller so that every
// project writing one of these files uses the same words for the same faults,
// which is what makes two projects' records comparable at all.
func (l *Log) Record(stage, target, name string, attempt int, response llm.Response, err error) error {
	entry := Entry{Stage: stage, Target: target, Route: name, Attempt: attempt,
		Model: response.Model, Usage: response.Usage, ElapsedMS: response.Elapsed.Milliseconds()}
	if entry.Route == "" {
		entry.Route = response.Route
	}
	if err == nil {
		entry.OK = true
		return l.Write(entry)
	}
	signal := llm.ClassifyError(err)
	entry.State = signal.State
	entry.Error = err.Error()
	return l.Write(entry)
}

func (l *Log) app() string {
	if l.App != "" {
		return l.App
	}
	return llm.App()
}

func (l *Log) now() time.Time {
	if l.Now != nil {
		return l.Now()
	}
	return time.Now().UTC()
}

// Read reads a whole file.
//
// A line that does not parse is skipped rather than failing the read. See the
// package comment: the alternative is losing a week of record to one truncated
// line.
func Read(path string) ([]Entry, error) {
	return ReadSince(path, time.Time{})
}

// ReadSince reads the entries written at or after a time, which is how a
// report covers a day without reading a year.
func ReadSince(path string, since time.Time) ([]Entry, error) {
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		// Nothing has been asked yet in this configuration, which is not an
		// error and should not make a report refuse to print.
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()

	var out []Entry
	scanner := bufio.NewScanner(file)
	// A line carries no prompt and no answer, only counts, so this is roomy.
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 || line[0] != '{' {
			continue
		}
		var entry Entry
		if json.Unmarshal(line, &entry) != nil {
			continue
		}
		if !since.IsZero() && entry.TS.Before(since) {
			continue
		}
		out = append(out, entry)
	}
	if err := scanner.Err(); err != nil {
		return out, fmt.Errorf("read %s: %w", path, err)
	}
	return out, nil
}

// Row is one route and stage rolled up.
type Row struct {
	Route string
	Stage string
	Asks  int
	OK    int
	// Refused is the asks that never reached the model. See Entry.Refused.
	Refused int
	// States counts the failures by kind, so that a run of transport errors
	// and a spent quota are not one number.
	States map[llm.State]int
	Usage  llm.Usage
	// Elapsed is the total time spent, answered and failed alike. A day of
	// refusals takes no time at all, and that is itself the tell.
	Elapsed time.Duration
}

// Rate is the share of asks that came back with an answer, from zero to one.
func (r Row) Rate() float64 {
	if r.Asks == 0 {
		return 0
	}
	return float64(r.OK) / float64(r.Asks)
}

// Summary rolls entries up by route and stage.
//
// Sorted by route then stage rather than by volume, so that the same fleet
// prints in the same order every day and two reports can be read side by side.
func Summary(entries []Entry) []Row {
	index := map[string]*Row{}
	for _, entry := range entries {
		key := entry.Route + "\x00" + entry.Stage
		row := index[key]
		if row == nil {
			row = &Row{Route: entry.Route, Stage: entry.Stage, States: map[llm.State]int{}}
			index[key] = row
		}
		row.Asks++
		row.Usage = row.Usage.Add(entry.Usage)
		row.Elapsed += entry.Elapsed()
		if entry.OK {
			row.OK++
			continue
		}
		state := entry.State
		if state == "" {
			state = llm.StateUnknown
		}
		row.States[state]++
		if entry.Refused() {
			row.Refused++
		}
	}
	out := make([]Row, 0, len(index))
	for _, row := range index {
		out = append(out, *row)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Route != out[j].Route {
			return out[i].Route < out[j].Route
		}
		return out[i].Stage < out[j].Stage
	})
	return out
}

// Since is the entries from the last window, for a report of the day or of the
// hour. It reads the clock, so a test passes its own now through Filter.
func Since(entries []Entry, window time.Duration) []Entry {
	return Filter(entries, time.Now().UTC().Add(-window))
}

// Filter is the entries at or after a time.
func Filter(entries []Entry, cutoff time.Time) []Entry {
	var out []Entry
	for _, entry := range entries {
		if !entry.TS.Before(cutoff) {
			out = append(out, entry)
		}
	}
	return out
}

// Table renders a summary.
//
// The refused column is the one worth the width. Every other column can look
// healthy on a route that is answering nothing: the asks are being made, the
// time is being spent, and the tokens are zero because nobody read the
// question.
func Table(rows []Row) string {
	var out strings.Builder
	fmt.Fprintf(&out, "%-14s  %-10s  %6s  %6s  %8s  %9s  %9s  %s\n",
		"route", "stage", "asks", "ok", "refused", "in", "out", "why not")
	for _, row := range rows {
		name := row.Route
		if name == "" {
			name = "-"
		}
		stage := row.Stage
		if stage == "" {
			stage = "-"
		}
		fmt.Fprintf(&out, "%-14s  %-10s  %6d  %6d  %8d  %9d  %9d  %s\n",
			name, stage, row.Asks, row.OK, row.Refused,
			row.Usage.InputTokens, row.Usage.OutputTokens, why(row))
	}
	return out.String()
}

// why names the failures in descending order, which is the sentence somebody
// reading the table wants: not that forty asks failed, but that thirty eight
// of them were a quota and two were a bug.
func why(row Row) string {
	if len(row.States) == 0 {
		return ""
	}
	states := make([]llm.State, 0, len(row.States))
	for state := range row.States {
		states = append(states, state)
	}
	slices.SortFunc(states, func(a, b llm.State) int {
		if row.States[a] != row.States[b] {
			return row.States[b] - row.States[a]
		}
		return strings.Compare(string(a), string(b))
	})
	parts := make([]string, 0, len(states))
	for _, state := range states {
		parts = append(parts, fmt.Sprintf("%d %s", row.States[state], state))
	}
	return strings.Join(parts, ", ")
}
