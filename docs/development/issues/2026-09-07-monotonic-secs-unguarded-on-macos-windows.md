# `_monotonic_secs()` reads uninitialised stack on macOS and Windows — cache TTL is undefined there

> **RESOLVED in 2.3.22 (2026-09-07)** via option (1): `_monotonic_secs` now
> carries per-target branches mirroring `lib/chrono.cyr` (Darwin id 6 +
> return value, Windows ms + return value, AGNOS `sys_uptime_ms()`, Linux
> `&ts`), rather than adding `chrono` to `[deps].stdlib` — that would have
> moved `dist/ai-hwaccel.deps` from 18 leaves to 19 and made every consumer
> satisfy it. The TTL test that does not assume Linux is **still open**.

**Filed:** 2026-09-07 (found auditing the cyrius `6.5.36 → 6.6.0` bump for 2.3.21)
**Severity:** Medium — `CachedRegistry` / `DiskCachedRegistry` TTL expiry is
undefined on two of the three shipped wheel targets. Detection still returns
correct results; only the *decision to reuse a cached one* is garbage.
**Repos:** ai-hwaccel `2.3.21` · cyrius `6.6.0`
**Status:** RESOLVED in 2.3.22 — follow-ups noted above. Pre-existing (not introduced by the 6.6.0 bump); kept out of a
toolchain-bump release deliberately.

## Summary

`src/cache.cyr:11-15`:

```
# clock_gettime(CLOCK_MONOTONIC) → seconds
fn _monotonic_secs() {
    var ts[16];
    syscall(228, 1, &ts);  # SYS_CLOCK_GETTIME, CLOCK_MONOTONIC
    return load64(&ts);
}
```

There is no target guard. The Linux assumption baked in here is *"228 fills the
`timespec` at `&ts`, and `CLOCK_MONOTONIC` is 1"*. Cyrius's own `lib/chrono.cyr`
documents that neither half holds elsewhere:

| target | what `syscall(228, …)` actually does | id for monotonic |
|---|---|---|
| Linux | fills `&ts` with `{sec, nsec}` | `1` |
| macOS arm64 | routes to libSystem `_clock_gettime_nsec_np(clock_id)` and **returns the ns count in the return register** — `&ts` is untouched | `6` (Darwin), not 1 |
| macOS x86 | 228 is **not routed at all** | — |
| Windows PE | routes to `kernel32!GetTickCount64` and **returns ms since boot in the return register** — `&ts` is untouched | — |

So on macOS and Windows `_monotonic_secs()` discards the value it was given and
returns `load64(&ts)` — a read of an **uninitialised 16-byte stack local**.

## Consequence

`cached_get` / `disk_cached_get` compare `_monotonic_secs() - last_detect_secs`
against the TTL. With a garbage clock the comparison is arbitrary: a cache may
never expire (stale hardware profile served forever) or expire on every call
(TTL silently disabled, every `get` re-shells out to `nvidia-smi`/`vulkaninfo`).
Both are wrong in ways no test catches, because the repo's tests run on Linux.

## Proposed fix

Either

1. mirror `chrono.cyr`'s target branches inside `_monotonic_secs` (three
   `#ifdef`s: Darwin id 6 + return value, Windows ms + return value, Linux
   `&ts`), or
2. add `"chrono"` to `[deps].stdlib` and call `clock_now_ms() / 1000`, which is
   the maintained version of exactly this logic.

(2) is preferable — it stops ai-hwaccel maintaining a private copy of a
per-target reroute table that upstream already keeps correct. It does add a
stdlib module to the vendored set and to `dist/ai-hwaccel.deps`, so it is a
consumer-visible change and belongs in its own version.

Then: add a TTL test that does not assume Linux, and run it on the ecb / ach /
cass build hosts.

## Note

`CHANGELOG.md`'s earlier "Investigated and …" note about the monotonic clock
should be revisited when this is fixed — it predates the `chrono.cyr` reroute
table being documented.
