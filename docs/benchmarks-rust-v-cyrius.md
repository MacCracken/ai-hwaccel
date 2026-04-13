# Benchmarks: Rust vs Cyrius

Comparison of the ai-hwaccel Rust implementation (v1.2.0, final commit `84dfb0d`)
against the Cyrius port (v1.2.0). All numbers from the same machine.
Rust source has been removed — this document preserves the final numbers.

## Summary

| Metric | Rust | Cyrius | Delta |
|--------|------|--------|-------|
| **Binary size** | 708 KB (release, stripped) | 197 KB | **-72%** |
| **Compile time** | ~1.8s (release, cached) | 215 ms | **-88%** |
| **Source LOC** | 11,278 | 5,271 | **-53%** |
| **Test LOC** | 6,057 | 2,244 | — |
| **Fuzz harnesses** | 0 (fuzz_helpers.rs only) | 6 (.fcyr) | — |
| **Tests** | 460 `#[test]` | 491 assertions (10 phases) | +7% |
| **Benchmarks** | 40 criterion | 3 suites (20 benchmarks) | — |
| **Dependencies** | 131 crates (Cargo.lock) | 0 | **-100%** |
| **Modules (src)** | 20 + 26 detect + 4 hardware = 50 | 20 + 19 detect = 39 | — |

## Runtime Performance Comparison

Cyrius compiles to direct x86_64 machine code without LLVM optimization passes
(no jump tables, no register allocation, no LTO). Absolute times are higher than
Rust+LLVM but well within budget for the detection use case (100ms+ CLI tool calls
dominate wall time).

### Memory Estimation

| Benchmark | Rust (LLVM) | Cyrius | Ratio |
|-----------|-------------|--------|-------|
| estimate_memory 70B FP16 | 257 ps | 9 ns | 35x |
| estimate_memory 70B BF16 | 256 ps | 7 ns | 27x |
| estimate_memory 70B INT4 | 255 ps | 7 ns | 27x |
| bits_per_param (all levels) | 295 ps | 3 ns | 10x |
| reduction_factor (all levels) | 255 ps | 5 ns | 20x |

### Training Memory

| Benchmark | Rust (LLVM) | Cyrius | Ratio |
|-----------|-------------|--------|-------|
| 7B full finetune GPU | 3.29 ns | 34 ns | 10x |
| 7B LoRA GPU | 3.34 ns | 34 ns | 10x |
| 7B QLoRA 4-bit GPU | 3.09 ns | 33 ns | 11x |

### Parsing

| Benchmark | Rust (LLVM) | Cyrius | Ratio |
|-----------|-------------|--------|-------|
| parse_cuda_output 8gpu | 5.80 µs | 18 µs | 3x |
| parse_vulkan_output 2gpu | 1.85 µs | 3 µs | 1.6x |
| detect_safetensors header | — | 953 ns | new |
| detect_gguf header | — | 490 ns | new |
| parse_neuron_json 2dev | — | 1 µs | new |

### Registry Queries

| Benchmark | Rust (LLVM) | Cyrius | Ratio |
|-----------|-------------|--------|-------|
| best_available (13 dev) | 39.56 ns | 858 ns | 22x |
| total_memory (13 dev) | 7.88 ns | 123 ns | 16x |
| has_accelerator (13 dev) | 1.28 ns | 23 ns | 18x |
| count_by_family GPU (13 dev) | 7.88 ns | 272 ns | 35x |
| plan_sharding 70B BF16 4gpu | 97.81 ns | 2 µs | 20x |
| json_serialize (13 dev) | 4.89 µs | 27 µs | 6x |
| json_summary (13 dev) | — | 4 µs | new |

### Analysis

- **Sub-ns Rust ops** (memory estimation, bit lookups) reflect LLVM constant folding
  and branch elimination — these become single instructions. Cyrius emits explicit
  if-chains which are 10-35x slower but still sub-10ns.
- **Parsing** is the closest category (1.6-3x) because both implementations are
  I/O-bound with similar string scanning logic.
- **Registry queries** show 16-35x gaps from Rust's iterator fusion, sorted
  collections, and inlined closures vs Cyrius linear scans.
- **JSON serialization** is 6x slower — str_builder is allocation-heavy vs
  serde's pre-sized buffers.
- **All Cyrius times are fast enough** — detection takes 100ms+ (CLI tool
  execution dominates), so sub-microsecond per-call overhead is irrelevant to
  end-to-end latency.

## Porting Coverage

### Fully Ported

| Rust Module | Cyrius Module | Notes |
|-------------|---------------|-------|
| units.rs | units.cyr | Identical constants |
| error.rs | error.cyr | Enum → integer codes + warning structs |
| quantization.rs | quantization.cyr | Fixed-point x1000 (no floats) |
| profile.rs | profile.cyr | Struct → heap layout with accessors |
| system_io.rs | system_io.cyr | — |
| registry.rs | registry.cyr | DetectBuilder → bitmask, from_json |
| plan.rs + sharding.rs | plan.cyr | Merged sharding into plan |
| training.rs | training.cyr | Fixed-point training memory estimates |
| cost.rs | cost.cyr | JSON parsing, instance recommendation |
| model_compat.rs | model.cyr | load_models, find_model, compatible_models, models_by_family, headroom |
| model_format.rs | model_format.cyr | SafeTensors/GGUF/ONNX/PyTorch header detection |
| requirement.rs | requirement.cyr | AcceleratorRequirement matching |
| async_detect.rs | async_detect.cyr | Threaded concurrent detection (thread.cyr) |
| cache.rs | cache.cyr | CachedRegistry + DiskCachedRegistry with mutex + TTL |
| lazy.rs | lazy.cyr | Per-family lazy detection with mutex |
| fuzz_helpers.rs | fuzz/*.fcyr | 6 fuzz harnesses (cuda, vulkan, neuron, apple, gaudi, model_format) |
| hardware/mod.rs | types.cyr | AcceleratorType/Family enums, classification |
| hardware/gaudi.rs | types.cyr | GaudiGeneration enum, HBM lookup |
| hardware/tpu.rs | types.cyr | TpuVersion enum, HBM lookup |
| hardware/neuron.rs | types.cyr | NeuronChipType enum, HBM lookup |
| main.rs + lib.rs | main.cyr + json_out.cyr | CLI + JSON serialization |
| detect/*.rs (24 files) | detect/*.cyr (19 files) | Consolidated: cerebras+graphcore+groq → cloud_asic, qualcomm+samsung+mediatek → edge, intel_npu+intel_oneapi → intel |

### Not Ported (By Design)

| Rust Module | Lines | Reason |
|-------------|-------|--------|
| ffi.rs | 116 | N/A — Cyrius is native code, no FFI wrapper needed |
| detect/windows.rs | 218 | Windows target — Cyrius doesn't target Windows yet (v4.0.0 roadmap) |

## Rust Benchmarks (Final — commit `84dfb0d`, 2026-04-06)

Last Rust benchmark run before the Cyrius port. These are the numbers to beat.

### Parsing

| Benchmark | Time |
|-----------|------|
| parse_cuda_output_8gpu | 5.80 µs |
| parse_vulkan_output_2gpu | 1.85 µs |
| parse_gaudi_output_8dev | 2.42 µs |
| parse_nvidia_bandwidth_8gpu | 1.06 µs |
| parse_max_dpm_clock | 374.01 ns |
| parse_link_speed | 78.70 ns |
| parse_ib_rate | 30.28 ns |
| parse_nvlink_output_2gpu | 1.03 µs |

### Memory Estimation

| Benchmark | Time |
|-----------|------|
| estimate_memory 70B FP16 | 257.20 ps |
| estimate_memory 70B FP32 | 257.70 ps |
| estimate_memory 70B BF16 | 255.80 ps |
| estimate_memory 70B INT8 | 254.90 ps |
| estimate_memory 70B INT4 | 255.00 ps |
| bits_per_param_all_levels | 294.90 ps |
| memory_reduction_factor_all_levels | 255.10 ps |

### Sharding & Planning

| Benchmark | Time |
|-----------|------|
| plan_sharding 70B BF16 (4 GPU) | 97.81 ns |
| plan_sharding 70B BF16 (13 devices) | 48.61 ns |
| plan_sharding_70B_128gpu | 1.65 µs |
| plan_sharding_405B_128gpu | 1.73 µs |
| plan_sharding_70B_mixed_61dev | 220.85 ns |
| suggest_quantization 70B (4 GPU) | 9.35 ns |
| suggest_quantization_70B_128gpu | 368.24 ns |

### Training Memory

| Benchmark | Time |
|-----------|------|
| 7B_full_finetune_gpu | 3.29 ns |
| 7B_lora_gpu | 3.34 ns |
| 7B_qlora_4bit_gpu | 3.09 ns |
| 7B_qlora_8bit_gpu | 3.08 ns |
| 7B_prefix_gpu | 3.53 ns |
| 7B_dpo_gpu | 3.09 ns |
| 7B_rlhf_gpu | 3.24 ns |
| 7B_distillation_gpu | 2.85 ns |

### Registry Queries

| Benchmark | Time |
|-----------|------|
| best_available (13 devices) | 39.56 ns |
| total_memory (13 devices) | 7.88 ns |
| by_family GPU (13 devices) | 7.88 ns |
| available_count_129dev | 44.71 ns |
| total_memory_129dev | 72.17 ns |
| has_accelerator_129dev | 1.28 ns |

### Cost & Instance Recommendation

| Benchmark | Time |
|-----------|------|
| 7B_bf16_all | 3.36 µs |
| 7B_int8_aws | 1.23 µs |
| 70B_bf16_all | 1.48 µs |
| 70B_int4_gcp | 634.70 ns |
| cheapest_7B_bf16 | 3.44 µs |
| cheapest_70B_bf16 | 1.64 µs |

### Detection

| Benchmark | Time |
|-----------|------|
| detect_all | 145.55 ms |
| detect_none (CPU only) | 144.34 ms |
| concurrent_detect_4_threads | 167.69 ms |
| detect single cuda | 144.74 ms |
| detect single vulkan | 145.80 ms |

### JSON Serialization

| Benchmark | Time |
|-----------|------|
| serialize (13 devices) | 4.89 µs |
| deserialize (13 devices) | 6.12 µs |
| serialize_129dev | 35.70 µs |
| deserialize_129dev | 45.21 µs |
| serialize_registry | 1.62 µs |
| deserialize_registry | 2.11 µs |

### Caching & Lazy

| Benchmark | Time |
|-----------|------|
| cached_registry get_hit | 31.65 ns |
| cached_registry invalidate | 6.80 ns |
| lazy_registry new | 10.84 µs |
| lazy_registry by_family_gpu_warm | 65.85 ns |

---

Generated from Rust `benchmarks.md` (commit `84dfb0d`, 2026-04-06).
Rust source has been removed — this document preserves the final Rust numbers for comparison.
Cyrius benchmarks to be added via `./scripts/bench-history.sh`.
