# cyrius 6.7.18: `exec_capture_status` is not safe on two threads at once

**Filed:** 2026-10-29 (found in the ai-hwaccel 2.5.5 audit)
**Severity:** Medium — on real threads (Linux, macOS, Windows) a healthy
capture is reported as a deadline kill (-2) and its output is dropped
whenever another thread's tool hangs. ai-hwaccel's `--threaded` lost a
working GPU that way.
**Owner: CYRIUS (upstream).** `lib/process.cyr` and `lib/process_win.cyr`
are vendored stdlib; ai-hwaccel never edits `lib/`.
**Repos:** cyrius `6.7.18` · ai-hwaccel `2.5.5`
**Status:** OPEN upstream. ai-hwaccel 2.5.5 works around it by serialising
every capture (`cmd_capture` in `src/detect/command.cyr`, its own mutex), so
`registry_detect_threaded`'s detector threads still overlap everything but
the tools. Remove the mutex once the pinned cyrius fixes all four parts below.

## Summary

A capture keeps per-call state where every other thread's capture can reach
it. Four separate problems, one root: the capture path was written for one
caller at a time.

### 1. The idle cut is a process-global flag

`_proc_read_pipe` (lib/process.cyr:675) writes `_proc_read_cut` (a module
global, :673): 0 on entry, 1 when the idle deadline expires. `_proc_capture`
reads it back after its own drain has ended (:931 `var cut = _proc_read_cut;`)
and, when it is 1 and the child exited on its own, rewrites the result into a
deadline kill (:939-942: `reaped = 0`, then `_proc_end_cut_group(pid)`).

So any capture still reading when another thread's capture is cut reads that
cut as its own. The window is not microseconds: it is every capture that
straddles the instant the other tool's idle bound expires.

### 2. The capture pipe is inherited by every concurrently started tool

POSIX: `_proc_capture` makes its pipe with plain `sys_pipe` (:914, `pipe(2)`,
no `O_CLOEXEC`), and the child closes nothing before `execve`. Threads share
one fd table, so a tool another thread forks while this pipe is open inherits
both ends. The capture then sees EOF only when that unrelated tool exits. If
it hangs, the healthy capture idles out and is reported as -2. The child also
inherits the caller's fd 0, so a tool that reads stdin consumes the
embedder's input.

Windows: `CreatePipe` makes the write end inheritable (lib/process_win.cyr:632,
`bInheritHandle=1`) and `CreateProcessW` passes `bInheritHandles = TRUE`
(:358), so a concurrent spawn inherits the write handle; `_win_watchdog` then
fires and the capture returns -2 while `_win_drain` waits on the other tool.

### 3. The kill-tree snapshot is one shared buffer

`_proc_tree` (:202) holds the descendant snapshot, a 4 KiB dirent buffer and
64 KiB of `/proc/<pid>/task/*/children` text. `_proc_tree_ready` allocates it
unsynchronised, and `_proc_tree_add` / `_proc_tree_collect` /
`_proc_snap_sweep` read and rewrite it in place. Two deadline kills at once
(two tools wedged on one driver, started microseconds apart with the same
idle bound) collect into and sweep the same buffer: a getdents record parsed
from bytes the other thread is overwriting, or a pid from interleaved text,
which then gets SIGTERM and SIGKILL.

### 4. The Windows job table is unlocked

`_win_job_put` (lib/process_win.cyr:272) is an unlocked scan-then-store, under
the note at :265 "Not thread-safe ... spawn from one thread". Two concurrent
spawns can tear a slot into (process of one, job of the other), so a deadline
ends the wrong tool's job, or a job handle is lost.

## Reproduction (Linux, cyrius 6.7.18)

Thread A runs `sh -c 'exec sleep 5'`, the main thread
`sh -c 'for i in 1 2 3 4 5 6 7 8; do echo x; sleep 0.1; done'`, both through
`exec_capture_status` with `proc_set_timeout_ms(300)`. The second exits 0 with
all 16 bytes printed and is still reported as -2 (ai-hwaccel's
`test_run_tool_threaded_cut`, tests/tcyr/io_test.tcyr, fails on every run
without the 2.5.5 mutex).

Pipe inheritance: while another thread's `sleep 1` capture is in flight,
`ls -l /proc/self/fd` run through a second capture lists one pipe more than it
does alone (`test_capture_pipe_not_inherited`, same file: "got 2, expected 1"
on every run without the mutex).

## Proposed fix

1. Return the cut per call: an out-parameter of `_proc_read_pipe` (a stack
   slot in `_proc_capture` and `run_capture`), not a global.
2. POSIX: create the pipe close-on-exec (`pipe2(O_CLOEXEC)` on Linux; macOS
   has no `pipe2`, so `pipe` + `fcntl(F_SETFD, FD_CLOEXEC)` under a spawn
   lock). `_proc_redirect` must then clear FD_CLOEXEC when `fd == target`
   (`dup2(fd, fd)` keeps the flag): without that, the 6.7.15 closed-std-fd
   case (fds 0 and 1 closed, so the pipe is (0, 1)) starts the tool with
   stdout closed. Point the child's stdin at `/dev/null`.
3. Make the kill-tree buffers per call (they are about 100 KiB, so a stack
   frame in `_proc_kill_begin`, or a lock around collect and sweep).
4. Windows: hold one lock across `CreatePipe` → `CreateProcessW` →
   `CloseHandle(hWrite)`, or pass an explicit
   `PROC_THREAD_ATTRIBUTE_HANDLE_LIST`; lock `_win_job_tab`.

Once all four land, ai-hwaccel can drop `_cmd_cap_mtx` and let tools overlap
again under `--threaded`.
