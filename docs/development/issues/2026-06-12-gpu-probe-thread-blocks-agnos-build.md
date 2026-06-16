# Threaded GPU backend probe blocks the agnos build (pulls the Linux clone-thread path → `CLONE_VM`)

**Filed:** 2026-06-12
**Severity:** MEDIUM — blocks the agnos build of ai-hwaccel and every downstream agnos consumer; on agnos, `mihi`/`iam`/`chakshu` cannot render GPU sysinfo
**Component:** `src/async_detect.cyr` — parallel backend detection via `thread_create` / `thread_join`
**Reported by:** agnos roadmap (1.44.x slot) + the kii agnos-port session (same stdlib clone-thread class as cyrius issue `2026-06-12-sankoch-locks-not-agnos-compatible.md`)
**Roadmap reference:** agnos `docs/development/roadmap.md`, 1.44.x row — *"`mihi`/`iam`/`chakshu` sysinfo display is blocked because ai-hwaccel's GPU probe pulls `thread`/`atomic`/Linux `CLONE_VM` — so it's either driven on ai-hwaccel's own (agent) roadmap, or made `CYRIUS_TARGET_AGNOS`-compatible once agnos has threads."*

## Symptom

Building any agnos consumer of ai-hwaccel (`mihi` → `iam`/`chakshu` GPU sysinfo) for the agnos target fails in the stdlib clone-thread path:

```
error: lib/mmap.cyr: undefined variable 'CLONE_VM' (missing include or enum?)
```

Host build is clean; the failure is agnos-target-only.

## Root cause

`async_detect.cyr` runs backend GPU detection in parallel threads:

```cyrius
var t = thread_create(&_detect_thread_entry, ta);   # async_detect.cyr:143
...
if (t != 0) { thread_join(t); }                      # async_detect.cyr:166
```

`thread_create` / `thread_join` (stdlib `lib/thread.cyr`) build threads with `clone(CLONE_VM | CLONE_FS | ...)` — the Linux clone model. `CLONE_VM` is a Linux-only constant, undefined under `CYRIUS_TARGET_AGNOS`. agnos userland is single-threaded (no clone-based threads), so the threaded probe cannot compile for the agnos target. ai-hwaccel currently has no `CYRIUS_TARGET_AGNOS` branch anywhere.

## The fix is already half-present

ai-hwaccel **already has a synchronous fallback** — when `thread_create` returns 0 it runs the detection inline:

```cyrius
# async_detect.cyr:149-150
hwlog_warn_n("detect_threaded: thread_create failed, sync fallback for backend ", bid);
_detect_thread_entry(ta);
```

The agnos fix is to take that sync path **unconditionally on agnos**, gating the threaded path behind `#ifndef CYRIUS_TARGET_AGNOS`:

```cyrius
#ifndef CYRIUS_TARGET_AGNOS
    var t = thread_create(&_detect_thread_entry, ta);
    ...
    if (t != 0) { thread_join(t); }
#endif
#ifdef CYRIUS_TARGET_AGNOS
    _detect_thread_entry(ta);   # single-threaded: probe backends sequentially
#endif
```

This is exactly the "made `CYRIUS_TARGET_AGNOS`-compatible" path the agnos roadmap anticipated. agnos exposes few/one GPU backend, so a sequential probe is fine; the host/Linux build keeps the parallel probe byte-identical. No new stdlib primitive is required — it reuses ai-hwaccel's own existing sync entry.

## Scope note

`async_detect.cyr` is the primary site; sweep `cache.cyr` and `lazy.cyr` (also flagged using `thread`/`atomic`) for the same pattern and gate any `thread_create`/`atomic_*` calls behind `#ifndef CYRIUS_TARGET_AGNOS`, falling back to the direct/sequential equivalent (single-threaded agnos has no contention to guard against).

## Related

- cyrius issue `2026-06-12-sankoch-locks-not-agnos-compatible.md` — same Linux-clone-thread / `CLONE_VM` class, surfaced via kii's PNG decoder. Both resolve under option (a) "per-consumer `CYRIUS_TARGET_AGNOS` no-op / sync fallback."
- Unblocks `mihi` / `iam` / `chakshu` GPU sysinfo on agnos once landed.
