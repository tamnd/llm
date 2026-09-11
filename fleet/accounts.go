package fleet

import (
	"regexp"
	"strconv"
	"strings"
	"time"
)

// AccountsScript finds the pool tool the way the probe finds it, because the
// path differs per host, and prints its account table. A host without the tool
// prints nothing rather than failing, so one missing box does not lose the
// others.
//
// The table comes out on standard error, not standard output: it is drawn by a
// console library, which writes to the console it opened. Reading only stdout
// gets an empty string and a fleet that looks like it has no accounts at all,
// which is what the first run of this said about every box in it. Hence the
// 2>&1.
const AccountsScript = `
tool=$(command -v chatgpt-tool || ls "$HOME"/chatgpt-tool/.venv/bin/chatgpt-tool 2>/dev/null || true)
if [ -n "$tool" ]; then "$tool" accounts 2>&1; fi
`

// verifiedMark is what the table puts against a slot holding a real session.
const verifiedMark = "✓"

// leftRE reads the cooldown off the "left" column.
//
// Off that column and not off the wall clock time printed beside it, because
// the boxes do not keep this machine's clock: one of them said 09:49 while it
// was 14:49 here, and a board that subtracted those got five hours of cooldown
// out of a profile that was about to come back.
var leftRE = regexp.MustCompile(`(?:(\d+)h)?(\d+)m left`)

// ParseAccounts reads the table into counts and the soonest return.
//
// Anything it does not recognise is a line it skips, which is the right
// failure: a new column on the far side costs a host its counts for one run
// and cannot invent slots that are not there.
//
// The soonest is less than zero when no banned slot said how long it had left,
// which is a host with nothing ready and no time to read off it — a different
// thing from zero, which is a cooldown that has run out. It is only meaningful
// when nothing is free, and the caller decides that.
//
// No email address leaves here. The table prints one per slot and this reads
// the state off the line and drops the rest, so nothing downstream can print
// an address it never received.
func ParseAccounts(out string) (Counts, time.Duration) {
	var counts Counts
	soonest := time.Duration(-1)
	for line := range strings.SplitSeq(out, "\n") {
		if !strings.Contains(line, verifiedMark) {
			continue
		}
		counts.Verified++
		// The state is matched without regard to case, because the host does
		// not keep one: it prints BANNED and LOCKED in capitals and stale-lock
		// in lower case, on the same table. Reading the lock in lower case
		// alone found neither of the two held slots a fleet really had and
		// counted each of them free instead, which was enough to keep a sweep
		// awake: nothing to wait for the moment a host reads ready, so it sent
		// a batch every few minutes at hosts whose only unbanned slot was held
		// by another process, eleven cycles and nothing written.
		state := strings.ToUpper(line)
		switch {
		case strings.Contains(state, "BANNED"):
			counts.Banned++
			if left, ok := parseLeft(line); ok && (soonest < 0 || left < soonest) {
				soonest = left
			}
		case strings.Contains(state, "STALE-LOCK"):
			counts.Stale++
		case strings.Contains(state, "LOCK"):
			counts.Locked++
		default:
			counts.Free++
		}
	}
	return counts, soonest
}

func parseLeft(line string) (time.Duration, bool) {
	match := leftRE.FindStringSubmatch(line)
	if match == nil {
		return 0, false
	}
	hours, _ := strconv.Atoi(match[1])
	minutes, _ := strconv.Atoi(match[2])
	return time.Duration(hours)*time.Hour + time.Duration(minutes)*time.Minute, true
}
