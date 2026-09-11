package ledger

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tamnd/llm"
)

func open(t *testing.T) *Log {
	t.Helper()
	log, err := Open(filepath.Join(t.TempDir(), "runs", "ledger.jsonl"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = log.Close() })
	log.App = "papers"
	log.Now = func() time.Time { return time.Date(2026, 9, 11, 12, 0, 0, 0, time.UTC) }
	return log
}

func TestWriteStampsTheLine(t *testing.T) {
	log := open(t)
	// The far end's own words, as they arrive: several lines of them.
	err := log.Write(Entry{Stage: "translate", Target: "secA/001", Route: "server1",
		Model: "gpt-5.1", State: llm.StateBroken,
		Error: "upstream said:\n  <html>\n  502 Bad Gateway\n  </html>\n"})
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	entries, err := Read(log.Path())
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("read %d entries", len(entries))
	}
	entry := entries[0]
	if entry.App != "papers" {
		t.Errorf("app = %q, want one file able to hold more than one tool's asks", entry.App)
	}
	if !entry.TS.Equal(time.Date(2026, 9, 11, 12, 0, 0, 0, time.UTC)) {
		t.Errorf("ts = %s", entry.TS)
	}
	// A month later that sentence is the only thing that says what went wrong,
	// so it is kept — on one line, because a record is read as a table.
	if strings.Contains(entry.Error, "\n") {
		t.Errorf("error spans lines: %q", entry.Error)
	}
	if !strings.Contains(entry.Error, "502 Bad Gateway") {
		t.Errorf("error = %q, want what the far end said", entry.Error)
	}
	// The file can say which documents somebody is reading, so it is written
	// for its owner only.
	info, err := os.Stat(log.Path())
	if err != nil {
		t.Fatal(err)
	}
	if mode := info.Mode().Perm(); mode != 0o600 {
		t.Errorf("mode = %o, want 0600", mode)
	}
}

// Recording asks is optional and a caller should not have to branch around it.
func TestANilLogIsUsable(t *testing.T) {
	var log *Log
	if err := log.Write(Entry{Stage: "translate"}); err != nil {
		t.Errorf("Write: %v", err)
	}
	if err := log.Record("translate", "secA/001", "server1", 1, llm.Response{}, nil); err != nil {
		t.Errorf("Record: %v", err)
	}
	if err := log.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

// The failure is classified here rather than by the caller, so that every
// project writing one of these files uses the same words for the same faults.
func TestRecordClassifiesTheFailure(t *testing.T) {
	log := open(t)
	answered := llm.Response{Text: "an answer", Model: "gpt-5.1", Route: "server1",
		Usage: llm.Usage{InputTokens: 1200, OutputTokens: 300}, Elapsed: 41 * time.Second}
	if err := log.Record("translate", "secA/001", "", 1, answered, nil); err != nil {
		t.Fatal(err)
	}
	spent := errors.New("route free-1: 429 Too Many Requests: rate_limit_exceeded")
	if err := log.Record("translate", "secA/002", "free-1", 2, llm.Response{}, spent); err != nil {
		t.Fatal(err)
	}

	entries, err := Read(log.Path())
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("read %d entries", len(entries))
	}
	if !entries[0].OK || entries[0].Route != "server1" {
		t.Errorf("entry = %+v, want the route off the answer when the caller named none", entries[0])
	}
	if entries[0].Elapsed() != 41*time.Second {
		t.Errorf("elapsed = %s", entries[0].Elapsed())
	}
	if entries[0].Usage.InputTokens != 1200 {
		t.Errorf("usage = %+v", entries[0].Usage)
	}
	if entries[1].OK || entries[1].State != llm.StateQuota {
		t.Errorf("entry = %+v, want a quota", entries[1])
	}
	if entries[1].Attempt != 2 {
		t.Errorf("attempt = %d", entries[1].Attempt)
	}
}

// A refusal is free, instant and invisible in every other report, and a day
// made of them looks from the outside exactly like a day of work.
func TestRefused(t *testing.T) {
	for _, c := range []struct {
		state llm.State
		want  bool
	}{
		{llm.StateQuota, true},
		{llm.StateUnauthorized, true},
		{llm.StateGone, true},
		{llm.StateBroken, false},
		{llm.StateUnreachable, false},
		{"", false},
	} {
		if got := (Entry{State: c.state}).Refused(); got != c.want {
			t.Errorf("Refused with state %q = %t", c.state, got)
		}
	}
}

// A run appends from several lanes at once. The mutex is what keeps two of
// them from interleaving halves of a line.
func TestConcurrentLanesWriteWholeLines(t *testing.T) {
	log := open(t)
	const lanes, each = 8, 25
	var group sync.WaitGroup
	for lane := range lanes {
		group.Add(1)
		go func() {
			defer group.Done()
			for n := range each {
				if err := log.Write(Entry{Stage: "translate", Route: "server1",
					Target: strings.Repeat("x", 200), Attempt: lane*each + n + 1}); err != nil {
					t.Errorf("Write: %v", err)
				}
			}
		}()
	}
	group.Wait()

	entries, err := Read(log.Path())
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != lanes*each {
		t.Fatalf("read %d lines, want %d whole ones", len(entries), lanes*each)
	}
	seen := map[int]bool{}
	for _, entry := range entries {
		if len(entry.Target) != 200 {
			t.Fatalf("a line was cut: %+v", entry)
		}
		seen[entry.Attempt] = true
	}
	if len(seen) != lanes*each {
		t.Errorf("%d distinct lines, want %d", len(seen), lanes*each)
	}
}

// A record of a week of work is not worth throwing away over a line that was
// being written when the power went.
func TestReadSkipsWhatItCannotParse(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	content := `{"ts":"2026-09-11T12:00:00Z","app":"papers","route":"server1","ok":true}
{"ts":"2026-09-11T12:00:01Z","app":"papers","route":"serv
a note somebody appended by hand

{"ts":"2026-09-11T12:00:02Z","app":"papers","route":"server2","ok":false,"state":"quota"}
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	entries, err := Read(path)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("read %d entries, want the two whole ones", len(entries))
	}
	if entries[0].Route != "server1" || entries[1].Route != "server2" {
		t.Errorf("entries = %+v", entries)
	}
}

// Nothing has been asked yet in this configuration, which should not make a
// report refuse to print.
func TestReadOfAMissingFileIsEmpty(t *testing.T) {
	entries, err := Read(filepath.Join(t.TempDir(), "nothing.jsonl"))
	if err != nil || entries != nil {
		t.Errorf("Read = %v, %v", entries, err)
	}
}

func TestReadSinceAndFilter(t *testing.T) {
	log := open(t)
	noon := time.Date(2026, 9, 11, 12, 0, 0, 0, time.UTC)
	for index, at := range []time.Time{noon.Add(-48 * time.Hour), noon.Add(-time.Hour), noon} {
		if err := log.Write(Entry{TS: at, Route: "server1", Attempt: index}); err != nil {
			t.Fatal(err)
		}
	}
	recent, err := ReadSince(log.Path(), noon.Add(-2*time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if len(recent) != 2 {
		t.Errorf("read %d entries, want the two inside the window", len(recent))
	}
	all, err := Read(log.Path())
	if err != nil {
		t.Fatal(err)
	}
	if got := Filter(all, noon); len(got) != 1 {
		t.Errorf("Filter kept %d, want the one at the cutoff", len(got))
	}
	// Since reads the clock rather than taking a cutoff, so it is given
	// entries stamped against the same clock.
	live := []Entry{{TS: time.Now().UTC()}, {TS: time.Now().UTC().Add(-2 * time.Hour)}}
	if got := Since(live, time.Minute); len(got) != 1 {
		t.Errorf("Since kept %d, want the one written just now", len(got))
	}
}

func TestSummaryGroupsByRouteAndStage(t *testing.T) {
	entries := []Entry{
		{Route: "server1", Stage: "translate", OK: true, ElapsedMS: 41000,
			Usage: llm.Usage{InputTokens: 1000, OutputTokens: 200}},
		{Route: "server1", Stage: "translate", OK: true, ElapsedMS: 39000,
			Usage: llm.Usage{InputTokens: 1000, OutputTokens: 300}},
		{Route: "server1", Stage: "translate", State: llm.StateBroken},
		{Route: "server1", Stage: "ocr", OK: true},
		{Route: "free-1", Stage: "translate", State: llm.StateQuota},
		{Route: "free-1", Stage: "translate", State: llm.StateQuota},
		// A failure the caller did not classify is unknown, not an answer.
		{Route: "free-1", Stage: "translate"},
	}
	rows := Summary(entries)
	if len(rows) != 3 {
		t.Fatalf("rows = %+v", rows)
	}
	// Sorted by route then stage, so two days of reports can be read side by
	// side.
	if rows[0].Route != "free-1" || rows[1].Stage != "ocr" || rows[2].Stage != "translate" {
		t.Fatalf("order = %+v", rows)
	}
	free := rows[0]
	if free.Asks != 3 || free.OK != 0 || free.Refused != 2 {
		t.Errorf("free = %+v", free)
	}
	if free.States[llm.StateUnknown] != 1 {
		t.Errorf("states = %v, want the unclassified failure counted", free.States)
	}
	if free.Rate() != 0 {
		t.Errorf("rate = %v", free.Rate())
	}
	// A day of refusals takes no time at all, and that is itself the tell.
	if free.Elapsed != 0 {
		t.Errorf("elapsed = %s", free.Elapsed)
	}
	work := rows[2]
	if work.Asks != 3 || work.OK != 2 || work.Refused != 0 {
		t.Errorf("work = %+v", work)
	}
	if work.Usage.InputTokens != 2000 || work.Usage.OutputTokens != 500 {
		t.Errorf("usage = %+v", work.Usage)
	}
	if work.Elapsed != 80*time.Second {
		t.Errorf("elapsed = %s", work.Elapsed)
	}
	if rate := work.Rate(); rate < 0.66 || rate > 0.67 {
		t.Errorf("rate = %v", rate)
	}
	if (Row{}).Rate() != 0 {
		t.Error("a row with no asks reported a rate")
	}
}

// Not that forty asks failed, but that thirty eight of them were a quota and
// two were a bug.
func TestTableNamesTheFailuresInOrder(t *testing.T) {
	var entries []Entry
	for range 38 {
		entries = append(entries, Entry{Route: "free-1", Stage: "translate", State: llm.StateQuota})
	}
	for range 2 {
		entries = append(entries, Entry{Route: "free-1", Stage: "translate", State: llm.StateBroken})
	}
	entries = append(entries, Entry{OK: true})

	got := Table(Summary(entries))
	if !strings.Contains(got, "38 quota, 2 broken") {
		t.Errorf("table:\n%s", got)
	}
	// The refused column is the one worth the width.
	if !strings.Contains(got, "refused") {
		t.Errorf("table has no refused column:\n%s", got)
	}
	// A route or stage the caller left empty still gets a row, because an ask
	// nobody labelled still happened.
	if !strings.Contains(got, "-  ") {
		t.Errorf("table:\n%s", got)
	}
}

func TestDefaultPathFollowsTheApp(t *testing.T) {
	dir := t.TempDir()
	llm.Configure(llm.Config{App: "papers", ConfigDir: dir})
	t.Cleanup(func() { llm.Configure(llm.Config{}) })

	if got := DefaultPath(); got != filepath.Join(dir, "ledger.jsonl") {
		t.Errorf("path = %q", got)
	}
	// Open with no path takes it, so a caller that wants the default writes no
	// path logic of its own.
	log, err := Open("")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = log.Close() }()
	if log.Path() != filepath.Join(dir, "ledger.jsonl") {
		t.Errorf("opened %q", log.Path())
	}
	if err := log.Write(Entry{Route: "server1", OK: true}); err != nil {
		t.Fatal(err)
	}
	entries, err := Read(log.Path())
	if err != nil || len(entries) != 1 {
		t.Fatalf("read %d, %v", len(entries), err)
	}
	// The app stamp comes from the library's configuration when the caller
	// sets none.
	if entries[0].App != "papers" {
		t.Errorf("app = %q", entries[0].App)
	}

	t.Setenv("PAPERS_LEDGER", "/tmp/somewhere-else.jsonl")
	if got := DefaultPath(); got != "/tmp/somewhere-else.jsonl" {
		t.Errorf("path = %q, want the environment's", got)
	}
}

// Appended and never rewritten: a second process opening the same file adds to
// it rather than starting it again.
func TestOpeningAgainAppends(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	for range 2 {
		log, err := Open(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := log.Write(Entry{App: "papers", Route: "server1", OK: true}); err != nil {
			t.Fatal(err)
		}
		if err := log.Close(); err != nil {
			t.Fatal(err)
		}
	}
	entries, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Errorf("read %d entries, want the first run's kept", len(entries))
	}
}
