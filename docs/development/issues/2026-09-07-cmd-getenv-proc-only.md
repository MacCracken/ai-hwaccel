# `cmd_getenv` is /proc-only — `AI_HWACCEL_DATA_DIR` is a silent no-op on macOS *and* Windows

> **RESOLVED in 2.3.21 (2026-09-07).** `cmd_getenv` is now a one-line
> delegation to the stdlib `getenv`, which cyrius 6.6.0 made correct on every
> target ai-hwaccel ships to. `AI_HWACCEL_DATA_DIR`, `which()`'s `$PATH`
> lookup and `NVIDIA_VISIBLE_DEVICES` now work on macOS and Windows.
> Verified on Linux that both resolution channels and the cwd fallback are
> unchanged. Steps 3 and 4 below (real-hardware re-test on ecb/ach/cass, and
> the roadmap 2.3.3/2.3.12 note) are **still open**.

**Filed:** 2026-09-07 (found auditing the cyrius `6.5.36 → 6.6.0` bump for 2.3.21)
**Severity:** Medium — a documented feature is silently inert on two of the three
shipped wheel targets, and the docs said it worked on one of them.
**Repos:** ai-hwaccel `2.3.21` · cyrius `6.6.0`
**Status:** RESOLVED in 2.3.21 — follow-ups noted above. 2.3.21 corrected the comments only; the behaviour change is
deliberately **not** in a toolchain-bump release — it needs its own version and
its own before/after benchmark delta.

## Summary

`src/detect/command.cyr` defines its own environment reader:

```
fn cmd_getenv(name) {
    var ebuf[8192];
    var fd = file_open("/proc/self/environ", 0, 0);
    ...
}
```

It was added to route around an old `io.cyr` `getenv` defect (variable
re-declaration in a loop). It reads `/proc/self/environ` unconditionally, so on
any target without `/proc` it returns `0` for **every** name — not an error, a
silent "unset".

`data_file_path()` is its only consumer of consequence:

```
var dir = cmd_data_dir_arg();                      # --data-dir  (portable: argv)
if (dir == 0) { dir = cmd_getenv("AI_HWACCEL_DATA_DIR"); }
```

so `AI_HWACCEL_DATA_DIR` has never worked on **macOS** (no `/proc`) or
**Windows PE** (no `/proc`). The comment block above `data_file_path` said the
variable was "honoured on Linux/macOS, but a SILENT NO-OP on PE" — the macOS
half of that was wrong from the day it was written, and 2.3.21 corrected it to
"Linux only".

## What changed in cyrius 6.6.0

The stdlib's own `getenv` was the *same* defect class, and it got fixed
upstream (cyrius v6.5.45):

- **macOS** — `_env_load` now derives `envp` from the init stack via
  `_macho_argv_base()` (arm64 `x28` / x86 `r15`) instead of opening `/proc`.
- **Windows** — `getenv` now calls the PE `GetEnvironmentVariableA` reroute
  (`syscall(0xF015, …)`). The reroute had existed since v6.1.x, but nothing in
  the stdlib ever called it.
- **AGNOS** — already had its own branch; that is the shape the other two now
  follow.

The observable consequence for ai-hwaccel today: `src/log.cyr:42` calls the
**stdlib** `getenv`, so **`AI_HWACCEL_LOG` starts working on macOS and Windows
with 2.3.21** (see the CHANGELOG). `AI_HWACCEL_DATA_DIR` does not, because it
goes through the repo-local `cmd_getenv`, which the upstream fix cannot reach.

## Proposed fix (own version, own benchmarks)

1. Confirm the original `io.cyr` re-declaration defect is gone under the pinned
   toolchain (it is not present in the 6.6.0 snapshot).
2. Reimplement `cmd_getenv` as a thin wrapper over the stdlib `getenv`, or
   delete it and call `getenv` directly. Note the ownership difference:
   `cmd_getenv` returns a heap copy the caller owns; the stdlib `getenv` returns
   a pointer into its own cached environment block. `data_file_path` only reads
   the value, but `str_builder_add_cstr` copies, so the lifetime is fine —
   verify each call site rather than assuming.
3. Re-test `--data-dir` **and** `AI_HWACCEL_DATA_DIR` on all three build hosts
   (ecb macOS arm64, ach Intel Mac, cass Windows), not just Linux.
4. Update `docs/development/roadmap.md` 2.3.3 / 2.3.12 and the comment blocks in
   `src/detect/command.cyr`, which currently explain the limitation in terms of
   a missing cyrius reroute rather than ai-hwaccel's own `/proc` read.

`--data-dir` remains the portable channel on every platform either way; this is
about the env var being a documented feature that quietly does nothing.
