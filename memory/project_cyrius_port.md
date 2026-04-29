---
name: Project state — v2.0.0 release
description: ai-hwaccel v2.0.0 fully ported from Rust to Cyrius, all modules complete, release-ready
type: project
---

ai-hwaccel v2.0.0 — complete Rust→Cyrius port. Released 2026-04-13.

**Binary**: 217 KB, 215ms compile, Cyrius cc3 3.10.0, zero dependencies.

**Modules** (37 total):
- 18 core: types, units, error, quantization, profile, system_io, registry, plan, training, cost, model, model_format, requirement, async_detect, cache, lazy, json_out, main
- 19 detect: cuda, rocm, apple, vulkan, tpu, gaudi, neuron, intel, amd_xdna, cloud_asic, edge, interconnect, bandwidth, pcie, numa, disk, environment, platform, command

**Testing**: 518 assertions (11 phases), 6 fuzz harnesses, 20 benchmarks (3 suites), 0 failures.

**Only 2 Rust modules not ported** (by design): ffi.rs (N/A), detect/windows.rs (no Windows target).

**Docs**: all updated for v2.0.0. Migration guide, benchmarks comparison, architecture, ADRs, guides — all clean.

**Cyrius improvements fed back**: undefined symbol diagnostic (3.10.0, caught 3 bugs in ai-hwaccel), post-4.0 roadmap items (+=, negative literals, jump tables, #derive tooling, struct initializers, dead function warning).

**Next opportunities** (when Cyrius features land):
- `+=` operators — 101 `i = i + 1` patterns to simplify
- `#derive(accessors)` — 274 manual load64/store64 calls to eliminate
- Jump tables — 79 if-chain enum dispatches to optimize
- Struct initializers — ~50 multi-line alloc+store patterns
- Dead-code elimination (v4.0 multi-file linker) — binary carries unused stdlib code
