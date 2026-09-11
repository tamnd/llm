package queue

import (
	"errors"
	"os"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"
)

const read Stage = "read"

// testQueue is a queue in a temporary directory. Passing a clock makes every
// deadline in the package a value the test sets rather than something it waits
// for.
func testQueue(t *testing.T, now *time.Time) *Queue {
	t.Helper()
	q, err := Open(t.TempDir(), read)
	if err != nil {
		t.Fatal(err)
	}
	if now != nil {
		q.Now = func() time.Time { return *now }
	}
	return q
}

func add(t *testing.T, q *Queue, target string) Job {
	t.Helper()
	job := New(read, target, "input-"+target, "prompt-v1")
	added, err := q.Add(job)
	if err != nil {
		t.Fatalf("Add %s: %v", target, err)
	}
	if !added {
		t.Fatalf("Add %s said it was already there", target)
	}
	return job
}

func TestNewIDIsTheContentAddress(t *testing.T) {
	first := NewID(read, "doc/0001", "sha-of-the-page", "sha-of-the-prompt")
	again := NewID(read, "doc/0001", "sha-of-the-page", "sha-of-the-prompt")
	if first != again {
		t.Errorf("the same work got two ids: %s and %s", first, again)
	}
	// Editing the instructions is new work, and it has to be a new id or a
	// rerun skips the page it was meant to redo.
	if edited := NewID(read, "doc/0001", "sha-of-the-page", "sha-v2"); edited == first {
		t.Error("a rewritten prompt kept the old id")
	}
	if len(first) != 16 {
		t.Errorf("id = %q, want something short enough to read in a directory listing", first)
	}
}

// A rerun of the same pipeline adds nothing, which is what makes it cheap.
// Done counts as already somewhere.
func TestAddIsIdempotentAcrossEveryState(t *testing.T) {
	q := testQueue(t, nil)
	job := add(t, q, "doc/0001")

	if added, err := q.Add(job); err != nil || added {
		t.Errorf("second Add = %t, %v, want it refused", added, err)
	}
	leased, err := q.Lease(read, "host", "doc", time.Minute)
	if err != nil {
		t.Fatalf("Lease: %v", err)
	}
	if state, err := q.Finish(leased, true, "read"); err != nil || state != Done {
		t.Fatalf("Finish = %s, %v", state, err)
	}
	if added, err := q.Add(job); err != nil || added {
		t.Errorf("a finished job was queued again: %t, %v", added, err)
	}
	if _, err := q.Lease(read, "host", "doc", time.Minute); !errors.Is(err, ErrEmpty) {
		t.Errorf("Lease after a clean run = %v, want the queue to be empty", err)
	}
}

func TestAddRefusesAJobWithNoIdentity(t *testing.T) {
	q := testQueue(t, nil)
	if _, err := q.Add(Job{Stage: read, Target: "doc/0001"}); err == nil {
		t.Error("a job with no id was accepted")
	}
	if _, err := q.Add(Job{ID: "abc", Target: "doc/0001"}); err == nil {
		t.Error("a job with no stage was accepted")
	}
}

// Read in hash order a book comes back scattered and no section is ever whole.
// The order is the order of the targets.
func TestLeaseTakesTargetsInOrder(t *testing.T) {
	q := testQueue(t, nil)
	for _, target := range []string{"doc/0003", "doc/0001", "doc/0002"} {
		add(t, q, target)
	}
	for _, want := range []string{"doc/0001", "doc/0002", "doc/0003"} {
		job, err := q.Lease(read, "host", "doc", time.Minute)
		if err != nil {
			t.Fatalf("Lease: %v", err)
		}
		if job.Target != want {
			t.Errorf("leased %s, want %s", job.Target, want)
		}
	}
}

// A worker that leases across documents reads page 4 of one book against page
// 4 of another and writes a real page over a real page.
func TestLeaseIsScopedToItsGroup(t *testing.T) {
	q := testQueue(t, nil)
	add(t, q, "alpha/0001")
	add(t, q, "beta/0001")

	job, err := q.Lease(read, "host", "beta", time.Minute)
	if err != nil {
		t.Fatalf("Lease: %v", err)
	}
	if job.Target != "beta/0001" {
		t.Errorf("leased %s out of group beta", job.Target)
	}
	if _, err := q.Lease(read, "host", "beta", time.Minute); !errors.Is(err, ErrEmpty) {
		t.Errorf("a second lease in an emptied group = %v", err)
	}
	// An empty group takes anything, for a caller that owns the whole stage.
	if job, err := q.Lease(read, "host", "", time.Minute); err != nil || job.Target != "alpha/0001" {
		t.Errorf("ungrouped lease = %s, %v", job.Target, err)
	}
}

func TestLeaseSpendsAnAttemptAndRecordsTheWorker(t *testing.T) {
	now := time.Date(2026, 9, 11, 12, 0, 0, 0, time.UTC)
	q := testQueue(t, &now)
	add(t, q, "doc/0001")

	job, err := q.Lease(read, "server1", "doc", 10*time.Minute)
	if err != nil {
		t.Fatalf("Lease: %v", err)
	}
	if job.Attempts != 1 {
		t.Errorf("attempts = %d after one lease", job.Attempts)
	}
	if job.Lease == nil {
		t.Fatal("a leased job carries no lease")
	}
	if want := now.Add(10*time.Minute + LeaseSlack); !job.Lease.Until.Equal(want) {
		t.Errorf("deadline = %s, want the estimate plus the slack (%s)", job.Lease.Until, want)
	}
	if job.Lease.Host != "server1" {
		t.Errorf("host = %q", job.Lease.Host)
	}
	// The pid means nothing without the machine that issued it.
	if job.Lease.PID != os.Getpid() || job.Lease.Worker != workerName() {
		t.Errorf("lease = pid %d on %q, want this process on this machine", job.Lease.PID, job.Lease.Worker)
	}
}

// Where the cheap model and the full one are two lanes on one subscription,
// the cheap lane must not pick its own failure back up and spend the job's
// attempts without the full model ever being asked.
func TestLeaseWhereSkipsWhatALaneShouldNotTake(t *testing.T) {
	q := testQueue(t, nil)
	add(t, q, "doc/0001")
	add(t, q, "doc/0002")

	// Fail the first page once so it goes back to pending with an attempt on it.
	job, err := q.Lease(read, "cheap", "doc", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if state, err := q.Fail(job, "wrong"); err != nil || state != Pending {
		t.Fatalf("Fail = %s, %v, want it back in pending", state, err)
	}

	fresh := func(job Job) bool { return job.Attempts == 0 }
	got, err := q.LeaseWhere(read, "cheap", "doc", fresh, time.Minute)
	if err != nil {
		t.Fatalf("LeaseWhere: %v", err)
	}
	if got.Target != "doc/0002" {
		t.Errorf("the cheap lane took %s, want the page it has not already got wrong", got.Target)
	}
	// The skipped job is still pending for a lane that can take it.
	if _, err := q.LeaseWhere(read, "cheap", "doc", fresh, time.Minute); !errors.Is(err, ErrEmpty) {
		t.Errorf("LeaseWhere = %v, want an empty lane rather than an error", err)
	}
	if _, err := q.Lease(read, "full", "doc", time.Minute); err != nil {
		t.Errorf("the full lane could not take the failed page: %v", err)
	}
}

func TestLeasePartTakesARange(t *testing.T) {
	q := testQueue(t, nil)
	for _, target := range []string{"doc/0001", "doc/0050", "doc/0051"} {
		add(t, q, target)
	}
	chapter := func(target string) bool { return target >= "doc/0050" }
	job, err := q.LeasePart(read, "host", "doc", chapter, time.Minute)
	if err != nil || job.Target != "doc/0050" {
		t.Errorf("LeasePart = %s, %v", job.Target, err)
	}
}

// The claim is the rename. Two workers that pick the same job both call it,
// and the loser gets ENOENT.
func TestConcurrentLeaseGivesTheJobToExactlyOneWorker(t *testing.T) {
	q := testQueue(t, nil)
	add(t, q, "doc/0001")

	const workers = 8
	var (
		mu     sync.Mutex
		won    int
		empty  int
		others []error
		wg     sync.WaitGroup
		start  = make(chan struct{})
	)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, err := q.Lease(read, "host", "doc", time.Minute)
			mu.Lock()
			defer mu.Unlock()
			switch {
			case err == nil:
				won++
			case errors.Is(err, ErrEmpty):
				empty++
			default:
				others = append(others, err)
			}
		}()
	}
	close(start)
	wg.Wait()

	if won != 1 {
		t.Errorf("%d workers took one job", won)
	}
	if empty != workers-1 {
		t.Errorf("%d workers came away empty, want %d", empty, workers-1)
	}
	if len(others) > 0 {
		t.Errorf("losing a race was reported as a failure: %v", others)
	}
}

// Two goroutines adding the same job once wrote one temp file over each other
// and both claimed the insert.
func TestConcurrentAddInsertsOnce(t *testing.T) {
	q := testQueue(t, nil)
	job := New(read, "doc/0001", "input", "prompt")

	var (
		mu    sync.Mutex
		added int
		wg    sync.WaitGroup
		start = make(chan struct{})
	)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			ok, err := q.Add(job)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				t.Errorf("Add: %v", err)
			}
			if ok {
				added++
			}
		}()
	}
	close(start)
	wg.Wait()

	if added != 1 {
		t.Errorf("%d of eight adds claimed the insert", added)
	}
	if stats, err := q.Stats(read); err != nil || stats.Counts[Pending] != 1 {
		t.Errorf("pending = %d, %v, want one file", stats.Counts[Pending], err)
	}
}

func TestAttemptsAreBoundedAndTheDeadJobSaysWhy(t *testing.T) {
	q := testQueue(t, nil)
	add(t, q, "doc/0001")

	for attempt := 1; attempt <= 3; attempt++ {
		job, err := q.Lease(read, "host", "doc", time.Minute)
		if err != nil {
			t.Fatalf("attempt %d: %v", attempt, err)
		}
		last := q.LastAttempt(job)
		if want := attempt == 3; last != want {
			t.Errorf("attempt %d: LastAttempt = %t, want %t", attempt, last, want)
		}
		state, err := q.Fail(job, "the page came back empty")
		if err != nil {
			t.Fatalf("Fail: %v", err)
		}
		if want := Pending; attempt < 3 && state != want {
			t.Errorf("attempt %d ended %s, want %s", attempt, state, want)
		}
		if attempt == 3 && state != Dead {
			t.Errorf("the fourth attempt was allowed: state = %s", state)
		}
	}
	if _, err := q.Lease(read, "host", "doc", time.Minute); !errors.Is(err, ErrEmpty) {
		t.Errorf("a dead job was leased again: %v", err)
	}
	dead, err := q.List(read, Dead)
	if err != nil || len(dead) != 1 {
		t.Fatalf("dead = %d, %v", len(dead), err)
	}
	// A dead job that does not say which hosts failed it is one nobody can act
	// on, so the host is read before the lease is cleared.
	if len(dead[0].History) != 3 {
		t.Errorf("history has %d events, want one per attempt", len(dead[0].History))
	}
	for _, event := range dead[0].History {
		if event.Host != "host" {
			t.Errorf("event with no host: %+v", event)
		}
	}
}

// Twenty one pages once went from pending to dead in forty one seconds without
// a single image leaving the laptop. Attempts are for the model's mistakes.
func TestReleaseGivesTheAttemptBack(t *testing.T) {
	q := testQueue(t, nil)
	add(t, q, "doc/0001")

	job, err := q.Lease(read, "host", "doc", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if err := q.Release(job, "ssh would not connect"); err != nil {
		t.Fatalf("Release: %v", err)
	}
	back, state, err := q.Find(read, job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if state != Pending {
		t.Errorf("state = %s, want pending", state)
	}
	if back.Attempts != 0 {
		t.Errorf("attempts = %d, want the attempt given back", back.Attempts)
	}
	if back.Lease != nil {
		t.Error("a released job still holds its lease")
	}
	// Given back quietly it could loop forever with nothing to read afterwards.
	if len(back.History) != 1 || back.History[0].Host != "host" {
		t.Errorf("history = %+v, want the hand back recorded against its host", back.History)
	}
}

// A laptop that sleeps through a lease leaves a job in leased with a deadline
// in the past, and any worker starting up puts it back.
func TestReapPutsAnExpiredLeaseBack(t *testing.T) {
	now := time.Date(2026, 9, 11, 12, 0, 0, 0, time.UTC)
	q := testQueue(t, &now)
	add(t, q, "doc/0001")

	job, err := q.Lease(read, "host", "doc", 10*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if reaped, err := q.Reap(read); err != nil || len(reaped) != 0 {
		t.Fatalf("a live lease was reaped: %v %v", reaped, err)
	}
	now = now.Add(10*time.Minute + LeaseSlack + time.Second)
	reaped, err := q.Reap(read)
	if err != nil {
		t.Fatalf("Reap: %v", err)
	}
	if len(reaped) != 1 || reaped[0] != job.ID {
		t.Fatalf("reaped = %v, want the expired job", reaped)
	}
	back, state, err := q.Find(read, job.ID)
	if err != nil {
		t.Fatal(err)
	}
	if state != Pending {
		t.Errorf("state = %s, want it back in pending", state)
	}
	// Reaping is not a free attempt: the work did go out.
	if back.Attempts != 1 {
		t.Errorf("attempts = %d, want the spent attempt kept", back.Attempts)
	}
}

// deadPID is the pid of a process that has run and been waited for. It is the
// only safe way to name a pid that is not alive, and it can in principle be
// reused, so the test checks before relying on it.
func deadPID(t *testing.T) int {
	t.Helper()
	cmd := exec.Command("/bin/sh", "-c", "exit 0")
	if err := cmd.Run(); err != nil {
		t.Skipf("cannot run a throwaway process here: %v", err)
	}
	pid := cmd.Process.Pid
	if alive(pid) {
		t.Skipf("pid %d was reused between exit and the check", pid)
	}
	return pid
}

// When a disk filled and every run died inside a minute, 25 held leases sat
// there for the better part of an hour with nobody coming back for them. A pid
// ends that, so long as it is only asked about on the machine that issued it.
func TestReapTakesBackALeaseWhoseWorkerIsGone(t *testing.T) {
	now := time.Date(2026, 9, 11, 12, 0, 0, 0, time.UTC)
	q := testQueue(t, &now)
	add(t, q, "doc/0001")

	job, err := q.Lease(read, "host", "doc", 10*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	job.Lease.PID = deadPID(t)
	if err := q.write(Leased, job); err != nil {
		t.Fatal(err)
	}
	// The deadline is still nearly an hour away.
	reaped, err := q.Reap(read)
	if err != nil {
		t.Fatalf("Reap: %v", err)
	}
	if len(reaped) != 1 {
		t.Fatalf("reaped = %v, want the job of a worker that is not there", reaped)
	}
	back, _, err := q.Find(read, job.ID)
	if err != nil {
		t.Fatal(err)
	}
	last := back.History[len(back.History)-1]
	if last.Reason != "the worker is gone, so the lease is not waited out" {
		t.Errorf("reason = %q, want it to say the pid decided and not the clock", last.Reason)
	}
}

// A pid issued by another machine means nothing here, and a lease with no
// worker on it is left to its deadline. Reclaiming another machine's live work
// is the failure this guards.
func TestReapLeavesAnotherMachinesLeaseAlone(t *testing.T) {
	now := time.Date(2026, 9, 11, 12, 0, 0, 0, time.UTC)
	pid := deadPID(t)

	for _, c := range []struct{ name, worker string }{
		{"another machine", "server2"},
		{"no machine named", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			q := testQueue(t, &now)
			add(t, q, "doc/0001")
			job, err := q.Lease(read, "host", "doc", 10*time.Minute)
			if err != nil {
				t.Fatal(err)
			}
			job.Lease.PID, job.Lease.Worker = pid, c.worker
			if err := q.write(Leased, job); err != nil {
				t.Fatal(err)
			}
			if reaped, err := q.Reap(read); err != nil || len(reaped) != 0 {
				t.Errorf("reaped = %v, %v, want the deadline left to decide", reaped, err)
			}
		})
	}
}

// The id is content addressed on the input, and a page image is not immutable:
// re-rendering page 70 at 600 dpi makes work the queue has never seen. Targets
// are what one page means.
func TestOutstandingIsByTarget(t *testing.T) {
	q := testQueue(t, nil)
	first := New(read, "doc/0070", "at-300-dpi", "prompt")
	second := New(read, "doc/0070", "at-600-dpi", "prompt")
	for _, job := range []Job{first, second} {
		if added, err := q.Add(job); err != nil || !added {
			t.Fatalf("Add: %t %v", added, err)
		}
	}
	out, err := q.Outstanding(read)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out["doc/0070"] != Pending {
		t.Errorf("outstanding = %v, want one entry for the page", out)
	}
	// A done job is not outstanding.
	job, err := q.Lease(read, "host", "doc", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Finish(job, true, ""); err != nil {
		t.Fatal(err)
	}
	if out, err = q.Outstanding(read); err != nil || out["doc/0070"] != Pending {
		t.Errorf("outstanding = %v, want the other copy of the page still listed", out)
	}
}

// A section cut into thirteen chunks today was cut into thirty when the
// English was longer, and chunks fourteen to thirty are pending with nothing
// behind them.
func TestSupersedeDropsWorkNobodyWants(t *testing.T) {
	q := testQueue(t, nil)
	stale := New(read, "secA/001", "text", "prompt-v1")
	current := New(read, "secA/001", "text", "prompt-v2")
	orphan := New(read, "secA/014", "text", "prompt-v1")
	elsewhere := New(read, "secB/001", "text", "prompt-v1")
	for _, job := range []Job{stale, current, orphan, elsewhere} {
		if added, err := q.Add(job); err != nil || !added {
			t.Fatalf("Add %s: %t %v", job.Target, added, err)
		}
	}

	dropped, err := q.Supersede(read, map[string]string{"secA/001": current.ID})
	if err != nil {
		t.Fatalf("Supersede: %v", err)
	}
	if dropped != 2 {
		t.Errorf("dropped %d, want the stale chunk and the orphan", dropped)
	}
	left := map[string]bool{}
	jobs, err := q.List(read, Pending)
	if err != nil {
		t.Fatal(err)
	}
	for _, job := range jobs {
		left[job.ID] = true
	}
	if !left[current.ID] {
		t.Error("the chunk that stands today was dropped")
	}
	// keep is built from one section, so another section is not this call's to
	// judge.
	if !left[elsewhere.ID] {
		t.Error("a section the caller said nothing about was emptied")
	}
	if left[stale.ID] || left[orphan.ID] {
		t.Errorf("superseded work is still pending: %v", left)
	}
}

func TestRetryAndReset(t *testing.T) {
	q := testQueue(t, nil)
	add(t, q, "doc/0001")
	for range 3 {
		job, err := q.Lease(read, "host", "doc", time.Minute)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := q.Fail(job, "broken"); err != nil {
			t.Fatal(err)
		}
	}
	moved, err := q.Retry(read)
	if err != nil || moved != 1 {
		t.Fatalf("Retry = %d, %v", moved, err)
	}
	job, state, err := q.Find(read, NewID(read, "doc/0001", "input-doc/0001", "prompt-v1"))
	if err != nil {
		t.Fatal(err)
	}
	if state != Pending || job.Attempts != 0 {
		t.Errorf("retried job is %s with %d attempts", state, job.Attempts)
	}
	// Retrying out of pending or done is a caller mistake worth naming.
	if _, err := q.Retry(read, Done); err == nil {
		t.Error("retry from done was accepted")
	}

	// Reset is --force on one job: the same id, so the history of what the last
	// answer was stays on the file.
	leased, err := q.Lease(read, "host", "doc", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Finish(leased, true, "read"); err != nil {
		t.Fatal(err)
	}
	found, err := q.Reset(read, leased.ID)
	if err != nil || !found {
		t.Fatalf("Reset = %t, %v", found, err)
	}
	after, state, err := q.Find(read, leased.ID)
	if err != nil {
		t.Fatal(err)
	}
	if state != Pending || after.Attempts != 0 {
		t.Errorf("reset job is %s with %d attempts", state, after.Attempts)
	}
	if len(after.History) == 0 {
		t.Error("reset threw the history away")
	}
	if found, err := q.Reset(read, "nosuchjob"); err != nil || found {
		t.Errorf("Reset of an unknown id = %t, %v", found, err)
	}
}

// A leased job belongs to a worker that is holding it.
func TestResetLeavesALeasedJobAlone(t *testing.T) {
	q := testQueue(t, nil)
	add(t, q, "doc/0001")
	job, err := q.Lease(read, "host", "doc", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if found, err := q.Reset(read, job.ID); err != nil || found {
		t.Errorf("Reset = %t, %v, want a held job left where it is", found, err)
	}
	if _, state, _ := q.Find(read, job.ID); state != Leased {
		t.Errorf("state = %s, want leased", state)
	}
}

// A queue that forgets its dead jobs is a queue that reports a clean run.
func TestDrainKeepsTheRecord(t *testing.T) {
	q := testQueue(t, nil)
	add(t, q, "doc/0001")
	add(t, q, "doc/0002")
	job, err := q.Lease(read, "host", "doc", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Finish(job, true, ""); err != nil {
		t.Fatal(err)
	}
	dropped, err := q.Drain(read)
	if err != nil || dropped != 1 {
		t.Fatalf("Drain = %d, %v", dropped, err)
	}
	stats, err := q.Stats(read)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Counts[Pending] != 0 || stats.Counts[Done] != 1 {
		t.Errorf("counts = %v, want the done job kept", stats.Counts)
	}
}

// Expired is the number that says a worker died rather than that work is in
// flight, which is why it is not folded into the leased count.
func TestStatsCountExpiredLeasesSeparately(t *testing.T) {
	now := time.Date(2026, 9, 11, 12, 0, 0, 0, time.UTC)
	q := testQueue(t, &now)
	add(t, q, "doc/0001")
	add(t, q, "doc/0002")
	for range 2 {
		if _, err := q.Lease(read, "host", "doc", time.Minute); err != nil {
			t.Fatal(err)
		}
	}
	stats, err := q.Stats(read)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Counts[Leased] != 2 || stats.Expired != 0 {
		t.Errorf("in flight: leased = %d, expired = %d", stats.Counts[Leased], stats.Expired)
	}
	now = now.Add(time.Minute + LeaseSlack + time.Second)
	if stats, err = q.Stats(read); err != nil || stats.Expired != 2 {
		t.Errorf("expired = %d, %v, want both", stats.Expired, err)
	}
	if stats.Total() != 2 {
		t.Errorf("total = %d", stats.Total())
	}

	all, err := q.StatsAll()
	if err != nil || len(all) != 1 || all[0].Stage != read {
		t.Fatalf("StatsAll = %v, %v", all, err)
	}
	// Naming a stage at Open is what makes an empty one show up in a report.
	board := Table(all)
	for _, want := range []string{"stage", "expired", string(read)} {
		if !strings.Contains(board, want) {
			t.Errorf("table has no %q:\n%s", want, board)
		}
	}
}

func TestStagesAreWhatIsOnDisk(t *testing.T) {
	q, err := Open(t.TempDir(), "translate", "read")
	if err != nil {
		t.Fatal(err)
	}
	stages, err := q.Stages()
	if err != nil {
		t.Fatal(err)
	}
	if len(stages) != 2 || stages[0] != "read" || stages[1] != "translate" {
		t.Errorf("stages = %v, want both in name order", stages)
	}
	// A stage is otherwise created the first time a job lands in it.
	if _, err := q.Add(New("assemble", "doc", "input", "prompt")); err != nil {
		t.Fatal(err)
	}
	if stages, err = q.Stages(); err != nil || len(stages) != 3 {
		t.Errorf("stages = %v, %v", stages, err)
	}
	if _, err := Open(t.TempDir(), "  "); err == nil {
		t.Error("a stage with no name was accepted")
	}
}

func TestGroupOf(t *testing.T) {
	for _, c := range []struct{ target, want string }{
		{"doc/0001", "doc"},
		{"doc/sec/0001", "doc"},
		{"doc", "doc"},
		{"", ""},
	} {
		if got := GroupOf(c.target); got != c.want {
			t.Errorf("GroupOf(%q) = %q, want %q", c.target, got, c.want)
		}
	}
}

func TestParseState(t *testing.T) {
	if state, err := ParseState(" Dead "); err != nil || state != Dead {
		t.Errorf("ParseState = %q, %v", state, err)
	}
	err := func() error { _, err := ParseState("stuck"); return err }()
	if err == nil {
		t.Fatal("an unknown state was accepted")
	}
	// The error is a command line's help, so it lists what would have worked.
	if !strings.Contains(err.Error(), "pending") {
		t.Errorf("error = %q, want the states in it", err)
	}
}
