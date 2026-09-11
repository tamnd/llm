// Package queue is the work list on disk.
//
// At 151 seconds a call, nothing worth doing fits in one process lifetime. A
// run of a thousand pages is hours, laptops sleep, tunnels drop, and somebody
// will hit Ctrl-C. So the state of the work lives in files, one per job, and
// any process can pick up where the last one stopped.
//
// The design is four rules:
//
// Leases, not locks. A worker claims a job by renaming it into leased/ with a
// deadline. A worker that dies holds nothing: its lease expires and the job
// comes back. There is no lock file to clean up and no pid to trust.
//
// Content addressed ids. The id is a hash of what the job is, so running the
// pipeline again produces the same ids, and work already done is skipped by
// the file being there. No separate ledger to keep in step.
//
// One side effect. A job writes its output file and nothing else, by temp file
// and rename, so a job that dies halfway leaves either the old output or the
// new one and never half of either.
//
// Bounded attempts. Three by default. After that the job is dead and the audit
// has to name it, because a page that silently retries forever is a page that
// never appears in the corpus and never appears in a report either.
package queue

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"syscall"
	"time"
)

// Stage is a kind of work. Each has its own directory, so draining the page
// reading queue does not touch translation.
//
// It is an open string. The library it was extracted from had a closed enum of
// five, which was right there and wrong here: what stages exist is the
// caller's business, and a library that has to be edited to add one is a
// library in the way. Queue.Stages lists the ones that exist on disk.
type Stage string

// State is where a job is. These are directory names.
type State string

const (
	Pending State = "pending"
	Leased  State = "leased"
	Done    State = "done"
	Failed  State = "failed"
	Dead    State = "dead"
)

// States is every state, in the order a report should list them.
var States = []State{Pending, Leased, Done, Failed, Dead}

// DefaultMaxAttempts is how many times a job is tried before it is dead.
//
// Three, because the failures worth retrying are a dropped tunnel, a session
// that logged itself out, and a page the model garbled once. A fourth attempt
// has never fixed any of those; what fixes them is a person reading the
// reason, which is what dead is for.
const DefaultMaxAttempts = 3

// LeaseSlack is added to a job's expected duration to get its lease deadline.
// It covers whatever happens after the model has answered and before the job
// is marked done: copying a result back, validating it, writing it out.
const LeaseSlack = 5 * time.Minute

// Job is one unit of work.
type Job struct {
	ID     string `json:"id"`
	Stage  Stage  `json:"stage"`
	Target string `json:"target"`
	// InputSHA256 and PromptSHA256 are what the id is made of, kept in the
	// file so a job can say why it is not the same job as the one before it. A
	// new prompt is a new id, which is the point: the corpus is rebuilt rather
	// than silently left at the old prompt's output.
	InputSHA256  string    `json:"input_sha256,omitempty"`
	PromptSHA256 string    `json:"prompt_sha256,omitempty"`
	Attempts     int       `json:"attempts"`
	Created      time.Time `json:"created"`
	Lease        *Lease    `json:"lease,omitempty"`
	History      []Event   `json:"history,omitempty"`
	// Meta carries whatever the stage needs and the queue does not care about,
	// such as the dpi a retry should escalate to.
	Meta map[string]string `json:"meta,omitempty"`
}

// Lease is a claim with a deadline on it.
//
// Host is the route the ask went to and Worker is the machine the worker
// itself ran on, which are two different things: a run on this laptop leases a
// job and puts the question to a box. Worker is what makes the PID mean
// something, since a pid is only an identity on the machine that issued it.
// See Reap for the one thing it is used for.
type Lease struct {
	Host   string    `json:"host"`
	Until  time.Time `json:"until"`
	PID    int       `json:"pid"`
	Worker string    `json:"worker,omitempty"`
}

// Event is one attempt, kept so that a dead job says what went wrong every
// time rather than only the last time.
type Event struct {
	TS     time.Time `json:"ts"`
	Host   string    `json:"host"`
	OK     bool      `json:"ok"`
	Reason string    `json:"reason,omitempty"`
}

// NewID is the content address of a job: the same work always gets the same
// name, so a rerun skips what is done without keeping a ledger.
func NewID(stage Stage, target, inputSHA, promptSHA string) string {
	sum := sha256.Sum256([]byte(string(stage) + "\x00" + target + "\x00" + inputSHA + "\x00" + promptSHA))
	return hex.EncodeToString(sum[:])[:16]
}

// New builds a job with its id filled in.
func New(stage Stage, target, inputSHA, promptSHA string) Job {
	return Job{
		ID: NewID(stage, target, inputSHA, promptSHA), Stage: stage, Target: target,
		InputSHA256: inputSHA, PromptSHA256: promptSHA, Created: time.Now().UTC(),
	}
}

// Queue is a directory of job files.
type Queue struct {
	Root string
	// MaxAttempts is the bound. Zero means DefaultMaxAttempts.
	MaxAttempts int
	// Now is the clock, replaceable because everything here is about deadlines
	// and no test should wait for one.
	Now func() time.Time
}

// Open makes the root and returns a queue. Naming stages creates their
// directories up front, which is worth doing when a report should show an
// empty stage rather than not show it; a stage is otherwise created the first
// time a job is added to it.
func Open(root string, stages ...Stage) (*Queue, error) {
	q := &Queue{Root: root}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, err
	}
	for _, stage := range stages {
		if err := q.mkdirs(stage); err != nil {
			return nil, err
		}
	}
	return q, nil
}

func (q *Queue) mkdirs(stage Stage) error {
	if strings.TrimSpace(string(stage)) == "" {
		return fmt.Errorf("stage has no name")
	}
	for _, state := range States {
		if err := os.MkdirAll(q.dir(stage, state), 0o755); err != nil {
			return err
		}
	}
	return nil
}

// Stages lists the stages that exist on disk, in name order. With no closed
// enum this is the only honest answer to "what is in this queue".
func (q *Queue) Stages() ([]Stage, error) {
	entries, err := os.ReadDir(q.Root)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	out := make([]Stage, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			out = append(out, Stage(entry.Name()))
		}
	}
	slices.Sort(out)
	return out, nil
}

func (q *Queue) now() time.Time {
	if q.Now != nil {
		return q.Now().UTC()
	}
	return time.Now().UTC()
}

func (q *Queue) maxAttempts() int {
	if q.MaxAttempts > 0 {
		return q.MaxAttempts
	}
	return DefaultMaxAttempts
}

func (q *Queue) dir(stage Stage, state State) string {
	return filepath.Join(q.Root, string(stage), string(state))
}

func (q *Queue) path(stage Stage, state State, id string) string {
	return filepath.Join(q.dir(stage, state), id+".json")
}

// ErrEmpty is returned by Lease when there is nothing to do. It is a normal
// end of run, not a failure, so callers compare against it rather than logging
// it as an error.
var ErrEmpty = errors.New("no pending jobs")

// Add puts a job in pending unless it is already somewhere.
//
// Already somewhere includes done, which is what makes a rerun cheap: the
// second run of the same pipeline adds nothing and the queue is empty.
func (q *Queue) Add(job Job) (bool, error) {
	if job.ID == "" || job.Stage == "" {
		return false, fmt.Errorf("job needs an id and a stage")
	}
	for _, state := range States {
		if _, err := os.Stat(q.path(job.Stage, state, job.ID)); err == nil {
			return false, nil
		}
	}
	if job.Created.IsZero() {
		job.Created = q.now()
	}
	if err := q.mkdirs(job.Stage); err != nil {
		return false, err
	}
	return q.create(Pending, job)
}

// temp writes a job to a uniquely named file in the destination directory and
// returns its path.
//
// Unique matters. The first version of this used <id>.json.tmp, which is fine
// until two goroutines add the same job at once: they write the same temp file
// over each other and both rename it, and the count of what was inserted is
// wrong. Found by the test, not by reasoning about it.
func (q *Queue) temp(state State, job Job) (string, error) {
	raw, err := json.MarshalIndent(job, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(q.dir(job.Stage, state), 0o755); err != nil {
		return "", err
	}
	file, err := os.CreateTemp(q.dir(job.Stage, state), job.ID+".*.tmp")
	if err != nil {
		return "", err
	}
	name := file.Name()
	if _, err := file.Write(append(raw, '\n')); err != nil {
		file.Close()
		os.Remove(name)
		return "", err
	}
	if err := file.Close(); err != nil {
		os.Remove(name)
		return "", err
	}
	return name, os.Chmod(name, 0o644)
}

// write puts a job file down atomically, so a reader never sees half of one.
// An existing file is replaced, which is what an update wants.
func (q *Queue) write(state State, job Job) error {
	name, err := q.temp(state, job)
	if err != nil {
		return err
	}
	if err := os.Rename(name, q.path(job.Stage, state, job.ID)); err != nil {
		os.Remove(name)
		return err
	}
	return nil
}

// create puts a job file down only if it is not there, and says whether it
// did.
//
// The exclusion is os.Link, which fails with EEXIST and does so atomically.
// The stat above it is a cheap check across all five states; this is the one
// that actually decides, because between a stat and a write there is room for
// another process to do the same thing.
func (q *Queue) create(state State, job Job) (bool, error) {
	name, err := q.temp(state, job)
	if err != nil {
		return false, err
	}
	defer os.Remove(name)
	err = os.Link(name, q.path(job.Stage, state, job.ID))
	if errors.Is(err, os.ErrExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

func (q *Queue) read(state State, stage Stage, id string) (Job, error) {
	raw, err := os.ReadFile(q.path(stage, state, id))
	if err != nil {
		return Job{}, err
	}
	var job Job
	if err := json.Unmarshal(raw, &job); err != nil {
		return Job{}, fmt.Errorf("decode job %s: %w", id, err)
	}
	return job, nil
}

// ids lists the job ids in one directory, in a stable order so that two
// workers starting together try the same job first and exactly one of them
// wins the rename, rather than both wandering off and colliding later.
func (q *Queue) ids(stage Stage, state State) ([]string, error) {
	entries, err := os.ReadDir(q.dir(stage, state))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(entries))
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".json") {
			continue
		}
		out = append(out, strings.TrimSuffix(name, ".json"))
	}
	sort.Strings(out)
	return out, nil
}

// Lease claims the next pending job for a host, out of one group.
//
// The claim is the rename itself. Two workers that pick the same job both call
// rename, the loser gets ENOENT because the file is already gone, and it moves
// on to the next one. That is the whole mutual exclusion, and it works between
// processes and between machines sharing a directory, which a mutex in this
// program would not.
//
// The group is the part of the target before the slash, which for page
// reading is the document. It is a parameter and not an option because a
// caller that leases across documents is a caller that has already gone wrong:
// a reading run knows the page number from the target and resolves the image
// against the document it was started with, so a job from another document
// reads the wrong file and writes a real page of one over a real page of
// another. An empty group takes anything, and only a caller that owns the
// whole stage should pass it.
//
// The order is the order of the targets, which for page reading is the order
// of the pages. The file names are content hashes, so taking them as the
// directory lists them reads a document in no order at all, and that is not a
// neutral choice when a document takes days to read: read in hash order, the
// first fifty pages that come back are scattered through the whole book, no
// section is finished, and nothing downstream can run on any of it, because
// assembly needs a section whole.
func (q *Queue) Lease(stage Stage, host, group string, expected time.Duration) (Job, error) {
	return q.LeasePart(stage, host, group, nil, expected)
}

// LeasePart is Lease over part of a group.
//
// want is asked about each target in turn and only the ones it takes are
// leased. A group is the unit of ownership and this is not: the jobs it passes
// over stay pending for the next run, which is the whole point. A book is read
// a chapter at a time because a chapter is what assembly works in, and the
// front matter before chapter I belongs to no chapter and is also what the
// acceptance rules reject, so without a range the first half day of a week
// long read goes on pages nothing is waiting for and then throws them away.
//
// A nil want takes everything, which is Lease.
func (q *Queue) LeasePart(stage Stage, host, group string, want func(target string) bool, expected time.Duration) (Job, error) {
	if want == nil {
		return q.LeaseWhere(stage, host, group, nil, expected)
	}
	return q.LeaseWhere(stage, host, group, func(job Job) bool { return want(job.Target) }, expected)
}

// LeaseWhere is LeasePart shown the whole job rather than the target alone.
//
// What a worker can usefully take is not always a question about the name of
// the work. A translation chunk that a cut down model has already got wrong is
// work a cut down model should not take again: where two routes are the same
// subscription on the cheap model and on the full one, the cheap one is tried
// first because it is cheap, and without this the cheap lane picks its own
// failure back up, gets it wrong the same way, and spends the chunk's three
// attempts without the full model ever being asked.
//
// Attempts is what says so, and it says it honestly: Lease spends one, Fail
// keeps it spent, and Release gives it back, so a pending job with attempts on
// it is a job some model read and answered wrongly.
//
// A lane whose predicate leaves nothing gets ErrEmpty and stops, which is the
// right end of it. The work is still pending and a lane that can take it is
// still running.
func (q *Queue) LeaseWhere(stage Stage, host, group string, want func(Job) bool, expected time.Duration) (Job, error) {
	jobs, err := q.pending(stage, group)
	if err != nil {
		return Job{}, err
	}
	if want != nil {
		kept := jobs[:0]
		for _, job := range jobs {
			if want(job) {
				kept = append(kept, job)
			}
		}
		jobs = kept
	}
	for _, job := range jobs {
		id := job.ID
		from, to := q.path(stage, Pending, id), q.path(stage, Leased, id)
		if err := os.MkdirAll(q.dir(stage, Leased), 0o755); err != nil {
			return Job{}, err
		}
		if err := os.Rename(from, to); err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return Job{}, err
		}
		job.Attempts++
		job.Lease = &Lease{Host: host, Until: q.now().Add(expected + LeaseSlack), PID: os.Getpid(), Worker: workerName()}
		if err := q.write(Leased, job); err != nil {
			return Job{}, err
		}
		return job, nil
	}
	return Job{}, ErrEmpty
}

// GroupOf is the part of a target before the slash, or the whole target when
// there is no slash in it.
func GroupOf(target string) string {
	group, _, ok := strings.Cut(target, "/")
	if !ok {
		return target
	}
	return group
}

// pending reads the pending jobs of one group, in target order.
//
// Sorting needs every job read rather than only the ones looked at before a
// match, which is the cost of the order. It is a few hundred small files on
// local disk against a page that takes minutes on a rented box, so the cost is
// not worth avoiding. A job that goes missing between the listing and the read
// is skipped: another worker took it, which is exactly what the rename in
// Lease is there to settle.
//
// Page targets should be zero padded — document/0022 — so that ordering the
// strings orders the pages. For a stage that names a file or a section rather
// than a page this is alphabetical, which is at least an order somebody can
// predict.
func (q *Queue) pending(stage Stage, group string) ([]Job, error) {
	ids, err := q.ids(stage, Pending)
	if err != nil {
		return nil, err
	}
	out := make([]Job, 0, len(ids))
	for _, id := range ids {
		job, err := q.read(Pending, stage, id)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		if group != "" && GroupOf(job.Target) != group {
			continue
		}
		out = append(out, job)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Target < out[j].Target })
	return out, nil
}

// Finish records the result of a leased job.
//
// A failure that has attempts left goes back to pending, because the next
// attempt will very likely land on a different host and that is usually all it
// takes. A failure that is out of attempts is dead, and dead jobs are the
// output of this package that anybody actually reads.
func (q *Queue) Finish(job Job, ok bool, reason string) (State, error) {
	// The host is read before the lease is cleared. Doing it the other way
	// round writes an empty host on every event: a dead job that does not say
	// which hosts failed it is a dead job nobody can act on.
	host := hostOf(job)
	job.Lease = nil
	job.History = append(job.History, Event{TS: q.now(), Host: host, OK: ok, Reason: reason})

	state := Done
	switch {
	case ok:
		state = Done
	case job.Attempts >= q.maxAttempts():
		state = Dead
	default:
		state = Pending
	}
	if err := q.write(state, job); err != nil {
		return "", err
	}
	if err := os.Remove(q.path(job.Stage, Leased, job.ID)); err != nil && !os.IsNotExist(err) {
		return state, err
	}
	return state, nil
}

func hostOf(job Job) string {
	if job.Lease != nil {
		return job.Lease.Host
	}
	// Finish clears the lease before recording, so fall back to the last host
	// that touched it rather than writing an empty field.
	if n := len(job.History); n > 0 {
		return job.History[n-1].Host
	}
	return ""
}

// Fail is Finish with ok false, which is the common call and reads better at
// the call site than a bare false.
func (q *Queue) Fail(job Job, reason string) (State, error) { return q.Finish(job, false, reason) }

// LastAttempt says whether failing this job now would kill it.
//
// A caller that wants to do something other than give up on a page it cannot
// read has to know that before it calls Fail, because Fail has already moved
// the job by the time it returns the state. A page reader uses it to decide
// whether to write a page with a fault in it rather than leave the corpus with
// a hole where the page should be.
func (q *Queue) LastAttempt(job Job) bool { return job.Attempts >= q.maxAttempts() }

// Release hands a leased job back without spending the attempt.
//
// Fail is for a page that was read and came back wrong. This is for a batch
// that never went out at all: an ssh that would not connect, a copy with
// nowhere to write, a batch this program refused to assemble. Twenty one pages
// once went from pending to dead in forty one seconds, three attempts each,
// without a single image leaving the laptop, because the only way to give a
// job back was to fail it. Attempts are for the model's mistakes.
//
// The attempt is given back too, since Lease spent it, and the handing back is
// still written into the history: a job that quietly loses attempts is a job
// that can loop forever with nothing to read afterwards.
func (q *Queue) Release(job Job, reason string) error {
	host := hostOf(job)
	job.Lease = nil
	if job.Attempts > 0 {
		job.Attempts--
	}
	job.History = append(job.History, Event{TS: q.now(), Host: host, OK: false,
		Reason: "handed back without an attempt: " + reason})
	if err := q.write(Pending, job); err != nil {
		return err
	}
	if err := os.Remove(q.path(job.Stage, Leased, job.ID)); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Outstanding is every target with a job still to run, and where that job is.
//
// It exists because the id is content addressed on the input hash and a page
// image is not immutable: a retry re-renders page 70 at 600 dpi, the hash
// changes, and the next Add sees work it has never seen before and queues the
// same page twice. Both then land in one batch, pointing at one file, and the
// batch is refused. Targets, not hashes, are what one page means.
func (q *Queue) Outstanding(stage Stage) (map[string]State, error) {
	out := map[string]State{}
	for _, state := range []State{Dead, Pending, Leased} {
		jobs, err := q.List(stage, state)
		if err != nil {
			return nil, err
		}
		for _, job := range jobs {
			out[job.Target] = state
		}
	}
	return out, nil
}

// workerName is the machine the worker runs on, which is what gives its pid a
// meaning. An error from the operating system is answered with the empty
// string and not with a guess, because a wrong name here would let one machine
// reclaim another's live work. Reap treats the empty string as unknown and
// falls back to the deadline.
func workerName() string {
	n, err := os.Hostname()
	if err != nil {
		return ""
	}
	return n
}

// alive says a process is still there. It is only asked about a pid this same
// machine issued. Signal 0 delivers nothing and only reports whether the
// process can be signalled; EPERM means it exists and belongs to somebody
// else, which is still alive.
func alive(pid int) bool {
	if pid <= 0 {
		return false
	}
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	err = p.Signal(syscall.Signal(0))
	return err == nil || errors.Is(err, os.ErrPermission)
}

// Reap returns jobs whose lease has expired, and jobs whose worker is gone.
//
// This is the whole crash recovery story. A worker that is killed, or a laptop
// that sleeps through a lease, leaves a job in leased with a deadline in the
// past. Any worker starting up calls Reap and the job is pending again.
// Nothing has to notice the crash and nothing has to be cleaned up by hand.
//
// The deadline alone is slow about it. LeaseSlack puts the deadline the better
// part of an hour out, because a call answered in 151 seconds on a good day
// can take twenty minutes on a bad one, and a lease that expires under a
// worker still waiting on an answer gives the job to a second worker and pays
// for it twice. So the deadline is generous on purpose, and the cost of that
// is paid at a crash: when a disk filled once and every run died inside a
// minute, 25 of the 39 held leases sat there for the better part of an hour
// with nobody coming back for them, and six lanes spent that hour finding
// their next chunk already leased and giving up.
//
// A pid is enough to end that, so long as it is only asked about on the
// machine that issued it, which is what Lease.Worker records. The reasoning
// that put "no pid to trust" at the top of this file still holds for
// everything else: the lease is not a lock, nothing has to be cleaned up by
// hand, and a lease with no Worker on it, or one written by another machine,
// is left to its deadline. The direction of the risk is the safe one. A pid
// that has been reused reads as alive and the job waits for its deadline,
// which is what happens today. Reclaiming a live worker's job would need its
// pid to be dead while it runs.
func (q *Queue) Reap(stage Stage) ([]string, error) {
	ids, err := q.ids(stage, Leased)
	if err != nil {
		return nil, err
	}
	now := q.now()
	var reaped []string
	for _, id := range ids {
		job, err := q.read(Leased, stage, id)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return reaped, err
		}
		why := "lease expired, the worker did not come back"
		switch {
		case job.Lease == nil:
		case !now.Before(job.Lease.Until):
		case job.Lease.Worker != "" && job.Lease.Worker == workerName() && !alive(job.Lease.PID):
			why = "the worker is gone, so the lease is not waited out"
		default:
			continue
		}
		state, err := q.Finish(job, false, why)
		if err != nil {
			return reaped, err
		}
		if state != Done {
			reaped = append(reaped, id)
		}
	}
	return reaped, nil
}

// Retry moves failed and dead jobs back to pending and clears their attempt
// count, for after the thing that was breaking them has been fixed.
func (q *Queue) Retry(stage Stage, states ...State) (int, error) {
	if len(states) == 0 {
		states = []State{Failed, Dead}
	}
	var moved int
	for _, state := range states {
		if state == Pending || state == Done {
			return moved, fmt.Errorf("retry from %s makes no sense", state)
		}
		ids, err := q.ids(stage, state)
		if err != nil {
			return moved, err
		}
		for _, id := range ids {
			job, err := q.read(state, stage, id)
			if err != nil {
				return moved, err
			}
			job.Attempts, job.Lease = 0, nil
			if err := q.write(Pending, job); err != nil {
				return moved, err
			}
			if err := os.Remove(q.path(stage, state, id)); err != nil && !os.IsNotExist(err) {
				return moved, err
			}
			moved++
		}
	}
	return moved, nil
}

// Reset puts one job back in pending wherever it is, with its attempts
// cleared, and says whether it found it.
//
// Retry is the same idea over a whole state, for after a fix. This is for one
// job that is done and is wanted again anyway, which is what a --force flag
// asks for. The id is the content address of the work, so a chunk that is
// asked again is the same job and not a new one: the history of what the last
// answer was and which host gave it stays on the file, and that is the point
// of doing it this way rather than by writing a fresh id nobody can line up
// with the old one.
//
// A leased job is left alone. Resetting one would hand the same work to a
// second worker while the first is still holding it, and the lease is there to
// expire on its own.
func (q *Queue) Reset(stage Stage, id string) (bool, error) {
	for _, state := range []State{Done, Failed, Dead} {
		job, err := q.read(state, stage, id)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return false, err
		}
		job.Attempts, job.Lease = 0, nil
		if err := q.write(Pending, job); err != nil {
			return false, err
		}
		if err := os.Remove(q.path(stage, state, id)); err != nil && !os.IsNotExist(err) {
			return false, err
		}
		return true, nil
	}
	return false, nil
}

// Supersede drops the pending jobs that a newer version of the same work has
// replaced.
//
// keep is target to the id that stands today. A pending job whose target is in
// keep and whose id is not the one kept is work nobody wants any more, and it
// is removed.
//
// So is a pending job in one of keep's groups whose target is not in keep at
// all. That target is a piece of work the caller no longer has: a section cut
// into thirteen chunks today was cut into thirty when the English was longer,
// and chunks fourteen to thirty are still sitting in pending with nothing
// behind them. A lane leases one, the caller cannot find a chunk of that
// number, the job fails, and three failures in the same second is what dead
// means, so the queue fills with dead jobs that were never real work.
//
// The group is what makes it safe to say "not in keep at all". keep is built
// from one section and pending holds every section, so dropping everything
// outside it would empty the queue. Inside a group the caller has just listed
// the whole of the work, so a target of that group it did not list is a target
// it does not have.
//
// This is the cost of keying a job on its content. The id is the target and
// the input and the instructions together, which is what makes an unchanged
// chunk the same job across runs, and it also means that editing the
// instructions leaves the old job sitting in pending beside the new one.
// Measured on one corpus after a single prompt rewrite, 1380 pending jobs
// stood for 837 distinct chunks, so two runs in five were asking for something
// a lane beside them was already asking for.
//
// Only pending is touched. A leased job belongs to a worker that is holding it
// and will finish or lose it on its own, and done, failed and dead are the
// record of what happened, which is not this function's to edit.
func (q *Queue) Supersede(stage Stage, keep map[string]string) (int, error) {
	ids, err := q.ids(stage, Pending)
	if err != nil {
		return 0, err
	}
	groups := make(map[string]bool, len(keep))
	for target := range keep {
		groups[GroupOf(target)] = true
	}
	dropped := 0
	for _, id := range ids {
		job, err := q.read(Pending, stage, id)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return dropped, err
		}
		want, ok := keep[job.Target]
		switch {
		case ok && want == id:
			continue
		case !ok && !groups[GroupOf(job.Target)]:
			continue
		}
		if err := os.Remove(q.path(stage, Pending, id)); err != nil && !os.IsNotExist(err) {
			return dropped, err
		}
		dropped++
	}
	return dropped, nil
}

// Drain removes pending jobs. Done, failed and dead stay: they are the record
// of what happened, and a queue that forgets its dead jobs is a queue that
// reports a clean run.
func (q *Queue) Drain(stage Stage) (int, error) {
	ids, err := q.ids(stage, Pending)
	if err != nil {
		return 0, err
	}
	for _, id := range ids {
		if err := os.Remove(q.path(stage, Pending, id)); err != nil && !os.IsNotExist(err) {
			return 0, err
		}
	}
	return len(ids), nil
}

// Stats counts one stage.
type Stats struct {
	Stage   Stage         `json:"stage"`
	Counts  map[State]int `json:"counts"`
	Expired int           `json:"expired"`
}

// Total is every job in the stage, whatever state it is in.
func (s Stats) Total() int {
	var total int
	for _, count := range s.Counts {
		total += count
	}
	return total
}

// Stats counts the jobs in a stage and says how many leases have expired,
// which is the number that says a worker died rather than that work is in
// flight.
func (q *Queue) Stats(stage Stage) (Stats, error) {
	out := Stats{Stage: stage, Counts: map[State]int{}}
	for _, state := range States {
		ids, err := q.ids(stage, state)
		if err != nil {
			return out, err
		}
		out.Counts[state] = len(ids)
	}
	now := q.now()
	ids, err := q.ids(stage, Leased)
	if err != nil {
		return out, err
	}
	for _, id := range ids {
		job, err := q.read(Leased, stage, id)
		if err != nil {
			continue
		}
		if job.Lease == nil || !now.Before(job.Lease.Until) {
			out.Expired++
		}
	}
	return out, nil
}

// StatsAll counts every stage on disk, in name order.
func (q *Queue) StatsAll() ([]Stats, error) {
	stages, err := q.Stages()
	if err != nil {
		return nil, err
	}
	out := make([]Stats, 0, len(stages))
	for _, stage := range stages {
		stats, err := q.Stats(stage)
		if err != nil {
			return out, err
		}
		out = append(out, stats)
	}
	return out, nil
}

// List reads every job in a state, for reports and for the audit that has to
// name the dead ones.
func (q *Queue) List(stage Stage, state State) ([]Job, error) {
	ids, err := q.ids(stage, state)
	if err != nil {
		return nil, err
	}
	out := make([]Job, 0, len(ids))
	for _, id := range ids {
		job, err := q.read(state, stage, id)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return out, err
		}
		out = append(out, job)
	}
	return out, nil
}

// Find looks a job up wherever it is.
func (q *Queue) Find(stage Stage, id string) (Job, State, error) {
	for _, state := range States {
		job, err := q.read(state, stage, id)
		if err == nil {
			return job, state, nil
		}
		if !os.IsNotExist(err) {
			return Job{}, "", err
		}
	}
	return Job{}, "", os.ErrNotExist
}

// Table renders queue stats as a board.
func Table(rows []Stats) string {
	width := len("stage")
	for _, row := range rows {
		width = max(width, len(row.Stage))
	}
	var out strings.Builder
	fmt.Fprintf(&out, "%-*s  %8s  %8s  %8s  %8s  %8s  %8s\n",
		width, "stage", "pending", "leased", "done", "failed", "dead", "expired")
	for _, row := range rows {
		fmt.Fprintf(&out, "%-*s  %8d  %8d  %8d  %8d  %8d  %8d\n",
			width, row.Stage, row.Counts[Pending], row.Counts[Leased], row.Counts[Done],
			row.Counts[Failed], row.Counts[Dead], row.Expired)
	}
	return out.String()
}

// ParseState turns a command line argument into a state.
func ParseState(name string) (State, error) {
	state := State(strings.ToLower(strings.TrimSpace(name)))
	if slices.Contains(States, state) {
		return state, nil
	}
	names := make([]string, len(States))
	for index, value := range States {
		names[index] = string(value)
	}
	return "", fmt.Errorf("unknown state %q, want one of %s", name, strings.Join(names, ", "))
}
