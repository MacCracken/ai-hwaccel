# `ERR_TIMEOUT` enum constant collides ecosystem-wide — namespace `DetectionError` as `HWA_ERR_*`

> **RESOLVED in ai-hwaccel 2.3.13 (2026-07-13).** The `DetectionError`
> enum was prefixed `ERR_* → HWA_ERR_*` (all six members; values
> unchanged; no bare aliases). Pulled ahead of the filed 2.4.0 target
> because the 6.2.11 → 6.4.62 toolchain bump made 6.4.62's linker *report*
> the collision as an in-tree `duplicate symbol 'ERR_TIMEOUT'` warning on
> our own `src/main.cyr` build (sakshi — the logging lib, present in every
> build — owns bare `ERR_TIMEOUT = 5`). `dist/ai-hwaccel.cyr` regenerated.
> **Sibling rename now also done (2026-07-13, pending version bump):** the
> `registry_new → hw_registry_new` function-namespacing this issue
> cross-references landed on main after 2.3.13 — a separate rename, same
> last-def-wins reasoning, no alias. See the 2026-06-11 issue.

**Filed:** 2026-06-23 (by a hoosh consumer — hoosh 2.4.7 toolchain bump to cyrius 6.2.37)
**Severity:** Medium — `last-definition-wins` build warning today; latent
value-dependent-logic hazard when `dist/ai-hwaccel.cyr` is compiled alongside
another lib that also defines bare `ERR_TIMEOUT` (and `ERR_NONE`/`ERR_PARSE`/
`ERR_TOOL_NOT_FOUND`).
**Component:** `src/error.cyr:9` (`enum DetectionError { … ERR_TIMEOUT = 3; … }`)
→ `dist/ai-hwaccel.cyr:232`.
**ai-hwaccel's role: FIX OWNER for its `DetectionError` enum.** Part of a
coordinated ecosystem-wide error-enum namespacing effort (see Cross-references) —
and a direct follow-on to the `registry_new` → `hw_registry_new` namespacing
already filed here.
**Repos:** ai-hwaccel `2.3.12` (mirrors filed in sigil, yukti, bote, sakshi).

## Summary

Cyrius enum members are **global constants** — `DetectionError` does *not*
namespace them. hoosh observes this directly when it pairs `dist/ai-hwaccel.cyr`
(hardware `/v1/hardware/cost`, `/compatible-models`) with sakshi/yukti:

```
warning:lib/ai-hwaccel.cyr:232: duplicate symbol 'ERR_TIMEOUT' redefined with conflicting value (last definition wins)
```

| Library | Enum | `ERR_TIMEOUT` | Source |
|---|---|---|---|
| **ai-hwaccel 2.3.12** | `DetectionError` | **3** | `src/error.cyr:9` → `dist/ai-hwaccel.cyr:232` |
| sakshi 2.4.1 | `ErrCode` | 5 | `src/error.cyr:26` |
| yukti 2.2.6 | `YuktiErrorKind` | 9 | `src/error.cyr:20` |

Siblings also collide: `ERR_NONE` (sigil 0), `ERR_PARSE` (yukti 15 / bote 4),
`ERR_TOOL_NOT_FOUND` (bote 0 vs ai-hwaccel 1 — **different values**). Cyrius
include semantics are textual paste + **last-definition-wins (with a warning)**.

## Why this is more than a warning

After last-wins one value survives per name. ai-hwaccel stores the code in its
warning struct (`warning_new(code, …)`, `WARNING_SIZE=40`, `+0 code`) and exposes
it via `warning_code(w)`. The `timeout_secs` field at `+32` is keyed off
`ERR_TIMEOUT`; if another lib's `ERR_TIMEOUT` wins, `warning_timeout_*` logic and
any consumer switch on `warning_code` silently compares against the wrong int.
`ERR_TOOL_NOT_FOUND` colliding with bote at a *different* value is the sharpest
case (bote 0 vs ai-hwaccel 1).

## The precedent already exists in-tree

`TLS_ERR_IO`, `PATRA_ERR_IO`, `SANDHI_ERR_TIMEOUT` — and ai-hwaccel's own
in-flight `hw_registry_new` rename — show the `hw`/prefix convention.

## Recommended fix

Prefix the **entire `DetectionError` enum** `ERR_* → HWA_ERR_*` (e.g.
`HWA_ERR_TIMEOUT`, `HWA_ERR_TOOL_NOT_FOUND`) and update `warning_new` /
`warning_*` constructors and every `ERR_*` reference under `src/`. Regenerate
`dist/ai-hwaccel.cyr`. Breaking change to the exported error surface → fold into
the **2.4.0** that already carries `registry_new → hw_registry_new`, optionally
keeping bare aliases for one minor.

## Interim (consumer-side)

hoosh tolerates the warning today (last-wins benign for its reachable hardware
paths). The upstream rename retires it for all ai-hwaccel + sakshi/yukti/bote
consumers (szal, mihi, hoosh).

## Cross-references

- ai-hwaccel `2026-06-11-registry-new-collision.md` (sibling namespacing, 2.4.0).
- sakshi `…2026-06-23-err-timeout-enum-collision-namespace.md`.
- yukti `…2026-06-23-err-enum-collision-namespace.md`.
- sigil / bote `…2026-06-23-err-io-enum-collision-namespace.md`.
