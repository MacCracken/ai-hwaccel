# ai-hwaccel — Claude Code Instructions

## Project Identity

**ai-hwaccel** (AI hardware acceleration) — Universal AI hardware accelerator detection — 18 families, quantization, sharding, training memory estimation

- **Type**: Cyrius binary (CLI)
- **License**: GPL-3.0-only
- **Compiler**: Cyrius cc5 5.10.34 (pinned in `cyrius.cyml`)
- **Version**: SemVer — `VERSION` file is the single source of truth; `cyrius.cyml` interpolates via `${file:VERSION}`

## Consumers

hoosh, daimon, Irfan, AgnosAI, murti, tazama

**Note**: Zero dependencies. Ported from Rust (v1.2.0, 847+ crates.io downloads) to Cyrius (v2.0.0).

## Development Process

### P(-1): Scaffold Hardening (before any new features)

1. Test + benchmark sweep of existing code
2. Cleanliness check: `cyrius fmt <file> --check`, `cyrius lint <file>`
3. Get baseline benchmarks (`./scripts/bench-history.sh`)
4. Initial refactor + audit (performance, memory, security, edge cases)
5. Cleanliness check — must be clean after audit
6. Additional tests/benchmarks from observations
7. Post-audit benchmarks — prove the wins
8. Repeat audit if heavy

### Development Loop (continuous)

1. Work phase — new features, roadmap items, bug fixes
2. Cleanliness check: `cyrius fmt <file> --check`, `cyrius lint <file>`
3. Test + benchmark additions for new code
4. Run benchmarks (`./scripts/bench-history.sh`)
5. Audit phase — review performance, memory, security, throughput, correctness
6. Cleanliness check — must be clean after audit
7. Deeper tests/benchmarks from audit observations
8. Run benchmarks again — prove the wins
9. If audit heavy → return to step 5
10. Documentation — update CHANGELOG, roadmap, docs
11. Return to step 1

### Key Principles

- **Never skip benchmarks.** Numbers don't lie. The CSV history is the proof.
- **Tests + benchmarks are the way.** 518 assertions, 6 fuzz harnesses, 20 benchmarks.
- **Own the stack.** Zero external dependencies.
- **No magic.** Every operation is measurable, auditable, traceable.
- **Fixed-point arithmetic** — x1000 multipliers, no floats in the entire codebase.
- **`str_builder` over `format!`** — avoid temporary allocations.
- **Enum constants over global vars** — avoid the 1024 global var limit.
- **Feature-gate optional modules** — `#ifdef` / `-D` flags for conditional compilation.
- **Vendored stdlib in `lib/` is gitignored** — `cyrius deps` repopulates it from the version-pinned `[deps].stdlib` snapshot in `cyrius.cyml`. Run `cyrius deps` after fresh clone or cyrius upgrade.
- **Single version source** — bump only `VERSION`; `cyrius.cyml` reads it via `${file:VERSION}`. Use `./scripts/version-bump.sh <new>` then add the CHANGELOG section, tag, and push.
- **All heap-allocated structs use `#derive(accessors)`** (cc5 v3.7.1+). Pattern: `#derive(accessors) struct <name> { field1; field2; … }` generates `<name>_<field>(p)` getters and `<name>_set_<field>(p, v)` setters under the existing accessor-prefix convention. Constructors stay manual (derive only generates accessors), calling the derived setters internally instead of raw `store64(p + N, v)`. The 16 derived structs at 2.1.7 are: `meta`, `storage`, `ic`, `plan`, `est`, `reg`, `model`, `profile`, `env`, `sio`, `shard`, `cloud_inst`, `rec`, `cached`, `disk_cached`, `lazy`. New structs follow the same pattern.
- **Raw-offset access on derived structs is CI-gated** (`.github/workflows/ci.yml` → `Raw-offset guard`). Cross-file `check_struct <struct> <defining_file> <param>` catches any `load64(<param> + N)` / `store64(<param> + N, …)` outside the defining file (only valid when param is unambiguous across `src/`). Per-file `check_offset_bound <file> <param> <struct> <field_count>` catches offsets past the struct boundary for ambiguous params. New derived struct → add an entry; rename a field → no gate change (derive picks up the new accessor name).

## DO NOT
- **Do not commit or push** — the user handles all git operations (commit, push, tag)
- **NEVER use `gh` CLI** — use `curl` to GitHub API only
- Do not add unnecessary dependencies — keep it lean
- Do not skip benchmarks before claiming performance improvements
- Do not commit `build/` directory
