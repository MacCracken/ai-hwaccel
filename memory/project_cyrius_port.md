---
name: Rust to Cyrius port — complete
description: ai-hwaccel fully ported, rust-old removed, 20 src modules, 19 detectors, 10 test phases, 6 fuzz, 3 bench suites
type: project
---

ai-hwaccel Rust→Cyrius port is complete. Rust source removed (2026-04-12).

Final Cyrius implementation:
- **20 source modules**: types, units, error, quantization, profile, system_io, registry, plan, training, cost, model, model_format, requirement, async_detect, cache, lazy, json_out + 19 detect modules
- **10 test phases**: 491 assertions, 0 failures
- **6 fuzz harnesses**: cuda_parser, model_format, vulkan_parser, neuron_parser, apple_parser, gaudi_parser
- **3 benchmark suites**: core (8 benchmarks), parsing (5), registry (7)
- **Binary**: 197KB, 215ms compile, Cyrius 3.9.0
- **0 dependencies**

Only 2 Rust modules not ported (by design): ffi.rs (N/A), detect/windows.rs (no Windows target).

Performance vs Rust+LLVM: 3-35x slower on micro-ops, but all times sub-microsecond. Detection dominated by 100ms+ CLI tool calls.

**Why:** Cyrius sovereignty — no crates.io, no external governance.
**How to apply:** All code is Cyrius. Benchmarks doc preserves Rust numbers for comparison.
