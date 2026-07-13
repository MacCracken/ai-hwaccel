# `registry_new` symbol collision — ai-hwaccel × bote-core

> **RESOLVED on main (2026-07-13), pending version bump.** ai-hwaccel's
> profile-registry constructor was renamed `registry_new → hw_registry_new`
> and all four internal callers updated (`src/registry.cyr:20` def +
> `:270`/`:350`, `src/lazy.cyr:157`, `src/async_detect.cyr:117`);
> `benches/registry.bcyr` + `tests/tcyr/registry_test.tcyr` follow.
> `dist/ai-hwaccel.cyr` regenerated — no bare `registry_new` code symbol
> remains. **No `registry_new` alias was kept:** an alias by that name would
> reintroduce the exact bote-core collision (same reasoning the 2.3.13
> `HWA_ERR_*` enum fix used to drop bare aliases). Verified under pinned
> cyrius 6.4.62: clean build, 12/12 units, 6/6 fuzz, bench-neutral (pure
> symbol rename → codegen-identical). Slated for 2.4.0 in the roadmap;
> pulled forward. `cached_registry_new` (ai-hwaccel-specific, no bote
> collision) intentionally left unchanged.

**Filed:** 2026-06-11 (discovered during the szal Rust→Cyrius port, M2 engine arc)
**Severity:** Medium — blocks any Cyrius consumer that includes BOTH
`dist/ai-hwaccel.cyr` and `dist/bote-core.cyr` (or `dist/bote.cyr`) in one
compile unit. szal is the first such consumer; its `engine_hardware` module is
blocked on this.
**ai-hwaccel's role: RECOMMENDED FIX OWNER.** Renaming ai-hwaccel's
profile-registry constructor is the cleanest resolution (see below). Suggested
for **ai-hwaccel 2.4.0**.
**Repos:** ai-hwaccel `2.3.9` · bote `2.7.3` (mirror filed in bote and szal).

## Summary

Two ecosystem libraries export a **public function with the same name but
different identity**:

| Library | Symbol | Shape | Source |
|---|---|---|---|
| ai-hwaccel 2.3.9 | `fn registry_new()` | **32-byte** profile registry (`REGISTRY_SIZE=32`: `{profiles, warnings, system_io, schema}`) | `src/registry.cyr:20` (→ `dist/ai-hwaccel.cyr:3549`) |
| bote-core 2.7.3 | `fn registry_new()` | **24-byte** tool registry `{entries map@0, versions map@8, names vec@16}` | `src/registry.cyr:148` (→ `dist/bote-core.cyr:554`, `dist/bote.cyr:553`) |

Cyrius include semantics are textual paste + **last-definition-wins (with a
warning)**. A consumer that includes both bundles gets exactly ONE `registry_new`
— whichever is included last — and every caller of the other one silently
allocates/interprets the wrong struct layout (32 vs 24 bytes), corrupting memory.

## Why include order can't fix it (and why this is ai-hwaccel's to own)

ai-hwaccel's own detection API calls `registry_new()` **internally**:

- `src/registry.cyr:270`, `src/registry.cyr:350` (`registry_detect*`)
- `src/lazy.cyr:157` (`cached_registry_new` / lazy registry)
- `src/async_detect.cyr:117`

So if bote-core's 24-byte `registry_new` wins the link, ai-hwaccel's
`registry_detect` / `cached_registry_new` build a 24-byte blob and then write the
profile/warnings/system_io/schema fields past its end — a memory-corruption bug,
not just a duplicate-symbol warning. No include order avoids it: the collision is
internal to ai-hwaccel's own call graph.

## Recommended fix

**Rename ai-hwaccel's `registry_new` → `hw_registry_new`** (and update its
internal callers above) in **2.4.0**. Rationale:

- ai-hwaccel's registry is hardware-profile specific; `hw_registry_new` is the
  natural namespace. The rest of the surface is already `reg_*` / `hw`-flavored —
  only the bare `registry_new` constructor collides.
- bote's `registry_new` is its long-standing ToolRegistry constructor with wide
  consumer reach; renaming there has a much larger blast radius.
- Consumers that pair bote + ai-hwaccel (szal, mihi, hoosh) all benefit from a
  single upstream rename instead of each vendoring + sed-renaming locally.

Public API note: `registry_new` is currently part of ai-hwaccel's exported
surface, so this is a **breaking change** → 2.4.0 (minor pre-1.0-style bump per
the ecosystem's post-1.0 convention). Keep a deprecated `registry_new` alias for
one minor if downstreams call it directly (szal does not — it uses
`cached_registry_new`/`registry_detect`).

## Interim (consumer-side)

Until 2.4.0, szal will vendor ai-hwaccel and apply
`registry_new`→`hw_registry_new` locally (the same pattern as its 9-symbol majra
vendoring rename). This issue is the upstream-fix request that retires that
workaround.

## Cross-references

- szal `docs/development/issues/2026-06-11-registry-new-collision.md` (blocker record).
- bote `docs/development/issues/2026-06-11-registry-new-collision.md`.
- szal port-plan §3.3 (flagged pre-port as "Open Q9").
