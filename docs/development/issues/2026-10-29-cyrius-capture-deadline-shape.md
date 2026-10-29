# cyrius 6.7.18: a capture's deadline costs twice the bound, the reap after KILL is unbounded, and Windows counts wall-clock

**Filed:** 2026-10-29 (found in the ai-hwaccel 2.5.5 audit)
**Severity:** Low to medium — a hung tool costs a capture about twice the
deadline (plus grace); a tool stuck in the kernel is not bounded at all; and
the same deadline means "idle" on POSIX and "total" on Windows.
**Owner: CYRIUS (upstream).** `lib/process.cyr` and `lib/process_win.cyr`
are vendored stdlib; ai-hwaccel never edits `lib/`.
**Repos:** cyrius `6.7.18` · ai-hwaccel `2.5.5`
**Status:** OPEN upstream. ai-hwaccel 2.5.5 documents the real cost
(`src/detect/command.cyr`'s header, `docs/development/threat-model.md`) and
pins it with an elapsed-time assertion (`test_run_tool_deadline`,
tests/tcyr/io_test.tcyr).

## Summary

### 1. Two full deadlines before the TERM (POSIX)

`_proc_capture` drains the pipe under the idle deadline
(`_proc_read_pipe(rfd, ..., proc_timeout_ms())`, lib/process.cyr:930), and
when that cuts, it waits for the child under a second full deadline
(`_proc_wait_deadline(pid, &stbuf, proc_timeout_ms())`, :935) before
`proc_kill_tree` sends TERM, waits up to `_PROC_GRACE_MS` (5 s) and sends
KILL. A silent tool under a 10 s bound therefore costs about 20 s, 25 s if
it ignores TERM. A child that has gone silent past the idle bound has
already been judged hung; the second wait could be short (or the grace
alone).

### 2. The reap after KILL does not time out

`_proc_kill_finish` (:454) does `sys_kill(pid, 9); sys_waitpid(pid, stbuf, 0);`.
A tool in uninterruptible sleep (D state, a wedged GPU driver: the
"nvidia-smi stalls when a GPU is in an error state" case) cannot die until
the kernel lets it, so the caller blocks with no bound, which is the hang the
deadline exists to prevent. A bounded `WNOHANG` poll after the KILL, then
returning "killed, not reaped" (the child is reaped later or by init), would
bound it.

### 3. Windows counts total time

`_win_watchdog` (lib/process_win.cyr:463) loops `while (waited < ms)` and
never resets on output, so on Windows a tool still printing when the bound
passes is killed mid-output, where POSIX would let it finish. Either make it
idle-based (reset when `_win_drain` reads), or document that the bound
differs per target.

## Reproduction (Linux, cyrius 6.7.18)

Under `proc_set_timeout_ms(300)`, `sh -c 'exec sleep 5'` returns -2 after
about 600 ms; under 1000 ms, after about 2 s, and `sh -c "trap '' TERM;
sleep 30"` after about 7 s (the grace). ai-hwaccel's whole detection with a
fake `nvidia-smi` that sleeps took about 40 s under the default 10 s bound
(it ran twice: the CUDA query and the NVLink probe), past its Python
wrapper's 30 s default. ai-hwaccel 2.5.5 skips the NVLink probe after the
CUDA query's deadline and raises the wrapper's default to 60 s; neither
changes what one hung tool costs.
