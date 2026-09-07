# `registry_detect_threaded` logs through sakshi, which is single-threaded by contract

> **RESOLVED in 2.3.21 (2026-09-07)** via option (2): the four parser sites no
> longer log. `warnings_log_parse()` (`src/error.cyr`) emits them on the main
> thread from the merged warnings vec, in both detection paths, filtered to
> `HWA_ERR_PARSE` so the logged set is byte-identical to before. No detector
> reachable from a thread body calls `hwlog_*` any more. Exercising the
> threaded path on real Apple Silicon is **still open**.

**Filed:** 2026-09-07 (found auditing the cyrius `6.5.36 → 6.6.0` bump for 2.3.21)
**Severity:** Medium — a data race on the logger's shared state, reachable only
through the threaded detection API, and only when the log level is raised enough
for a detector to actually emit.
**Repos:** ai-hwaccel `2.3.21` · cyrius `6.6.0` (sakshi 2.4.12)
**Status:** RESOLVED in 2.3.21 — follow-ups noted above. Pre-existing on Linux; **cyrius 6.5.44 extends the exposure to
arm64-macOS**, which is what surfaced it.

## Summary

`src/async_detect.cyr` spawns one thread per CLI-based backend:

```
fn _detect_thread_entry(ta) {
    ...
    if (bid == BACKEND_CUDA)   { detect_cuda(profs, warns); }
    elif (bid == BACKEND_GAUDI) { detect_gaudi(profs, warns); }
    ...
}
```

`detect_cuda` contains 3 `hwlog_*` calls and `detect_gaudi` 1, so up to six
threads can be inside sakshi concurrently.

`lib/sakshi.cyr`'s own header states the contract:

> Single-threaded only, EXCEPT `SK_OUT_ATOMIC_RING` (multi-producer).

`hwlog_init` (`src/log.cyr:40-50`) never selects an output target — it only
calls `sakshi_set_level` and `sakshi_trace_set` — so the process runs on the
default `SK_OUT_STDERR` path, which is *not* the multi-producer one.

## Why it surfaced now

This has been live on Linux for as long as `registry_detect_threaded` has
existed: `thread_create` there is a real `clone()`. What changed is macOS.
Through cyrius 6.5.43, arm64-macOS `thread_create` ran the thread body **inline**
and returned a fake handle, with `thread_join` a no-op — correct answers, one at
a time, and structurally race-free. cyrius **6.5.44** wires real Darwin pthreads
(`_pthread_create` via `__got[5]`, plus a blocking `__ulock` mutex), so from the
2.3.21 toolchain on, Apple Silicon consumers of the threaded API get genuine
concurrency — and inherit the same logger race Linux already had.

Note that the repo's own suite does not exercise this: no test calls
`registry_detect_threaded`, so CI stays green either way.

## Options (pick one, with a benchmark delta)

1. **Select the multi-producer target when threading.** Call
   `sakshi_output_atomic_ring()` from `hwlog_init` — but this changes the
   documented contract in `src/log.cyr` ("All logs go to stderr (fd 2)"), and a
   ring buffer needs a reader. Not a drop-in.
2. **Move logging out of the thread bodies.** Have `_detect_thread_entry`
   accumulate into its private `warns` vec only, and emit every `hwlog_*` in the
   post-join merge on the main thread. Keeps the stderr contract, keeps the
   warnings, costs nothing at runtime. **Recommended.**
3. **Document the limitation** on `registry_detect_threaded` and require callers
   that use it to select their own sakshi target.

Do not ship (1) without deciding what reads the ring.

## Related

The same 6.5.44 change makes arm64-macOS `mutex_lock`/`mutex_unlock` a blocking
`__ulock` park/wake instead of a bare spinlock — that one is a straight
improvement for `cached_registry`'s mutex, and needs no action.
