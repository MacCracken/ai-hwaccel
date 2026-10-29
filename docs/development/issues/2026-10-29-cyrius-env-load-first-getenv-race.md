# cyrius 6.7.18: a first `getenv` on two threads at once answers 0 on one of them

**Filed:** 2026-10-29 (found in the ai-hwaccel 2.5.5 audit)
**Severity:** Medium — a library consumer whose first `getenv` is
`registry_detect_threaded()` lost its vendor tools ("not found") on nearly
every run once its environment passed about 32 KB.
**Owner: CYRIUS (upstream).** `lib/io.cyr` is vendored stdlib; ai-hwaccel
never edits `lib/`.
**Repos:** cyrius `6.7.18` · ai-hwaccel `2.5.5`
**Status:** OPEN upstream. ai-hwaccel 2.5.5 loads the environment on the
calling thread before it starts detector threads (`env_block(0)` in
`registry_detect_threaded_with`, src/async_detect.cyr), and the TTL cache
detects under its lock, so two first misses no longer detect at once. Two
threads of a consumer calling `getenv` (or `registry_detect`) for the first
time together still race.

## Summary

`_env_load` (lib/io.cyr:1650) publishes "loaded, empty" before it has loaded
anything:

```
fn _env_load(): i64 {
    if (_env_len >= 0) { return 0; }
    _env_len = 0;
    ... read /proc/self/environ (macOS: flatten the init-stack envp) ...
    _env_blk = buf;
    _env_len = n;
```

`getenv` calls `_env_load()` and then reads `_env_len`. A second thread that
enters while the first is still reading sees `_env_len == 0`, skips the load
and answers 0: the variable reads as unset. The read starts with a 16 KiB
buffer and doubles with a copy, so the window grows with the environment.

There is also no release/acquire ordering between the `_env_blk` and
`_env_len` stores, so on arm64 macOS a reader can see the new length before
the new block pointer and read through address 0.

Windows and AGNOS are not affected: their `getenv` asks the system one name
at a time and keeps no cache.

## Reproduction (Linux, cyrius 6.7.18)

A program whose first statement after `alloc_init()` is
`registry_detect_threaded()`, run with a fake `vulkaninfo` first on `PATH` and
a 64 KB padding variable (`env -i PATH=<dir>:/usr/bin:/bin PAD=<64 KB>`),
reports `vulkaninfo` not found on every run without the 2.5.5 prime: each
detector thread's `which()` is that thread's first `getenv("PATH")`.
ai-hwaccel's `test_threaded_first_getenv` (tests/tcyr/system_test.tcyr) runs
exactly this as a child process.

## Proposed fix

Build the block into locals and publish it once: either a once-lock around
the load (losers wait for the winner), or a CAS from -1 to a "loading"
sentinel with waiters spinning until the final length is stored, with the
block pointer stored (release) before the length. Keep 6.6.7's rule that a
failed load leaves the cache unset, so a later call retries.
