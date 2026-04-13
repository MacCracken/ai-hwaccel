# ai-hwaccel — Claude Code Instructions

## Project Identity

**ai-hwaccel** (AI hardware acceleration) — Universal AI hardware accelerator detection — 18 families, quantization, sharding, training memory estimation

- **Type**: Cyrius binary (CLI)
- **License**: GPL-3.0-only
- **Compiler**: Cyrius cc3 3.10.0
- **Version**: SemVer 2.0.0

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

## DO NOT
- **Do not commit or push** — the user handles all git operations (commit, push, tag)
- **NEVER use `gh` CLI** — use `curl` to GitHub API only
- Do not add unnecessary dependencies — keep it lean
- Do not skip benchmarks before claiming performance improvements
- Do not commit `build/` directory
