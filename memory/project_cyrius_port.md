---
name: Rust to Cyrius port status
description: ai-hwaccel fully ported to Cyrius with 18 source modules, 19 detectors, 10 test phases, 6 fuzz harnesses
type: project
---

ai-hwaccel is fully ported from Rust to Cyrius. Original Rust preserved in `rust-old/`.

Current Cyrius implementation (2026-04-12):
- **18 source modules**: types, units, error, quantization, profile, system_io, registry, plan, training, cost, model, model_format, requirement, async_detect, json_out + 19 detect modules
- **10 test phases**: 491 assertions, 0 failures
- **6 fuzz harnesses**: cuda_parser, model_format, vulkan_parser, neuron_parser, apple_parser, gaudi_parser
- **Binary**: 188KB, 191ms compile
- **Vendored stdlib**: str.cyr (with `: Str` annotations), fnptr.cyr, thread.cyr, async.cyr, syscalls.cyr synced from upstream Cyrius v3.7.x

Remaining Rust modules NOT ported (by design):
- ffi.rs — N/A, Cyrius is native
- cache.rs — caching layer (threading available, not yet needed)
- lazy.rs — lazy per-family detection (could use thread.cyr)

**Why:** Cyrius is the AGNOS ecosystem language. Sovereignty — no crates.io, no external governance.

**How to apply:** All new code is Cyrius. Reference `rust-old/` for original logic when debugging.
