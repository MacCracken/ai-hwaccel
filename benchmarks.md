# Benchmarks

Latest: **2026-03-27T15:27:35Z** — commit `19a2753`

Tracking: `a281283` (baseline) → `19a2753` (previous) → `19a2753` (current)

## ungrouped

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `load_pricing_table` | 1.70 ns | 2.76 ns +62% | 2.01 ns +18% |
| `detect_all` | 1.00 ms | 1.12 ms +12% | 2.97 ms +196% |
| `detect_none (CPU only)` | 153.33 µs | 156.47 µs | 156.28 µs |
| `concurrent_detect_4_threads` | 4.59 ms | 5.09 ms +11% | 3.86 ms **-16%** |
| `parse_nvidia_bandwidth_8gpu` | 604.63 ns | 678.64 ns +12% | 619.44 ns |
| `nvidia_bus_width_all_ccs` | 19.99 ns | 25.26 ns +26% | 21.58 ns +8% |
| `estimate_bw_from_cc_all` | 22.85 ns | 27.44 ns +20% | 26.90 ns +18% |
| `parse_max_dpm_clock` | 138.52 ns | 161.97 ns +17% | 129.82 ns **-6%** |
| `parse_link_speed` | 42.59 ns | 56.77 ns +33% | 43.42 ns |
| `parse_ib_rate` | 44.47 ns | 62.72 ns +41% | 50.80 ns +14% |
| `parse_nvlink_output_2gpu` | 528.66 ns | 695.90 ns +32% | 620.21 ns +17% |
| `parse_cuda_output_8gpu` | 3.22 µs | 4.26 µs +32% | 3.80 µs +18% |
| `parse_vulkan_output_2gpu` | 923.96 ns | 1.03 µs +11% | 1.07 µs +15% |
| `parse_gaudi_output_8dev` | 1.40 µs | 1.67 µs +19% | 1.58 µs +13% |
| `plan_sharding 70B BF16 (4 GPU)` | 79.87 ns | 104.92 ns +31% | 67.84 ns **-15%** |
| `plan_sharding 70B BF16 (13 devices)` | 34.94 ns | 38.73 ns +11% | 32.52 ns **-7%** |
| `suggest_quantization 70B (4 GPU)` | 10.81 ns | 10.31 ns **-5%** | 10.81 ns |
| `suggest_quantization 70B (13 devices)` | 22.39 ns | 29.20 ns +30% | 25.56 ns +14% |
| `estimate_memory 70B FP16` | 238.50 ps | 281.10 ps +18% | 263.80 ps +11% |
| `estimate_training_memory 7B LoRA GPU` | 2.41 ns | 3.08 ns +28% | 3.23 ns +34% |
| `best_available (13 devices)` | 25.32 ns | 27.94 ns +10% | 22.54 ns **-11%** |
| `total_memory (13 devices)` | 5.20 ns | 6.12 ns +18% | 6.45 ns +24% |
| `by_family GPU (13 devices)` | 55.89 ns | 89.71 ns +60% | 59.60 ns +7% |
| `bits_per_param_all_levels` | 29.00 ps | 53.70 ps +85% | 33.70 ps +16% |
| `memory_reduction_factor_all_levels` | 29.00 ps | 39.70 ps +37% | 34.20 ps +18% |

## recommend_instance

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `7B_bf16_all` | 2.87 µs | 5.23 µs +82% | 2.81 µs |
| `7B_int8_aws` | 1.09 µs | 1.74 µs +60% | 1.03 µs **-6%** |
| `70B_bf16_all` | 1.39 µs | 1.80 µs +29% | 1.23 µs **-12%** |
| `70B_int4_gcp` | 603.34 ns | 688.95 ns +14% | 546.40 ns **-9%** |

## cheapest_instance

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `7B_bf16` | 3.28 µs | 3.16 µs **-4%** | 2.92 µs **-11%** |
| `70B_bf16` | 1.30 µs | 1.39 µs +7% | 1.25 µs **-4%** |

## detect_single

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `cuda` | 173.45 µs | 179.43 µs +3% | 187.68 µs +8% |
| `rocm` | 950.90 µs | 995.00 µs +5% | 994.93 µs +5% |
| `vulkan` | 174.74 µs | 188.28 µs +8% | 180.33 µs +3% |
| `apple` | 186.02 µs | 175.63 µs **-6%** | 174.09 µs **-6%** |
| `tpu` | 213.62 µs | 204.95 µs **-4%** | 210.77 µs |

## system_io

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `full_with_sysio` | 1.00 ms | 1.02 ms | 1.00 ms |
| `query_system_io` | 4.72 ns | 34.30 ps **-99%** | 37.50 ps **-99%** |
| `ingestion_1gb` | 3.60 ns | 2.14 ns **-40%** | 2.32 ns **-36%** |
| `ingestion_100gb` | 3.75 ns | 2.57 ns **-32%** | 2.26 ns **-40%** |
| `ingestion_1tb` | 3.80 ns | 2.54 ns **-33%** | 2.33 ns **-39%** |
| `serialize_registry` | 995.44 ns | 1.20 µs +20% | 984.66 ns |
| `deserialize_registry` | 1.49 µs | 1.66 µs +11% | 1.39 µs **-7%** |

## json_roundtrip

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `serialize (13 devices)` | 2.03 µs | 2.47 µs +22% | 2.22 µs +9% |
| `deserialize (13 devices)` | 3.10 µs | 4.46 µs +44% | 3.54 µs +14% |

## estimate_memory

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `70B_fp32` | 236.40 ps | 287.40 ps +22% | 279.90 ps +18% |
| `70B_fp16` | 237.60 ps | 300.80 ps +27% | 275.30 ps +16% |
| `70B_bf16` | 237.50 ps | 286.30 ps +21% | 272.30 ps +15% |
| `70B_int8` | 239.10 ps | 297.30 ps +24% | 287.80 ps +20% |
| `70B_int4` | 236.90 ps | 307.50 ps +30% | 262.50 ps +11% |

## suggest_quantization

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `1B_1gpu` | 5.46 ns | 7.25 ns +33% | 5.60 ns |
| `7B_1gpu` | 5.50 ns | 7.42 ns +35% | 5.63 ns |
| `13B_1gpu` | 5.71 ns | 7.67 ns +34% | 5.61 ns |
| `70B_1gpu` | 5.72 ns | 7.58 ns +32% | 5.82 ns |
| `405B_1gpu` | 5.80 ns | 8.01 ns +38% | 5.81 ns |

## registry_queries

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `available_4dev` | 53.23 ns | 73.59 ns +38% | 60.88 ns +14% |
| `available_129dev` | 336.90 ns | 465.96 ns +38% | 424.73 ns +26% |
| `available_61dev_mixed` | 175.08 ns | 224.84 ns +28% | 215.21 ns +23% |
| `best_available_129dev` | 134.86 ns | 164.74 ns +22% | 114.14 ns **-15%** |
| `total_memory_129dev` | 41.66 ns | 53.28 ns +28% | 43.71 ns +5% |
| `total_accelerator_memory_129dev` | 49.24 ns | 61.59 ns +25% | 70.68 ns +44% |
| `has_accelerator_129dev` | 1.24 ns | 1.70 ns +37% | 1.85 ns +49% |
| `by_family_gpu_61dev` | 209.69 ns | 336.42 ns +60% | 205.52 ns |
| `by_family_tpu_61dev` | 91.87 ns | 119.05 ns +30% | 88.53 ns **-4%** |
| `satisfying_gpu_61dev` | 214.81 ns | 280.17 ns +30% | 240.19 ns +12% |
| `satisfying_any_accel_61dev` | 213.51 ns | 322.37 ns +51% | 260.85 ns +22% |

## cached_registry

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `get_cached_hit` | 28.92 ns | 37.10 ns +28% | 31.21 ns +8% |
| `invalidate` | 3.67 ns | 6.39 ns +74% | 5.27 ns +44% |

## lazy_registry

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `new` | 10.21 µs | 13.69 µs +34% | 13.11 µs +28% |
| `by_family_gpu_cold` | 1.00 ms | 1.06 ms +5% | 1.20 ms +19% |
| `by_family_gpu_warm` | 44.56 ns | 60.55 ns +36% | 61.89 ns +39% |
| `into_registry` | 2.01 ms | 2.02 ms | 2.01 ms |

## large_registry_sharding

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `plan_sharding_70B_128gpu` | 1.27 µs | 1.71 µs +34% | 1.18 µs **-7%** |
| `plan_sharding_405B_128gpu` | 1.26 µs | 1.88 µs +50% | 1.14 µs **-9%** |
| `suggest_quantization_70B_128gpu` | 214.00 ns | 218.85 ns | 241.30 ns +13% |
| `plan_sharding_70B_mixed_61dev` | 174.58 ns | 154.98 ns **-11%** | 137.14 ns **-21%** |

## large_json

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `serialize_129dev` | 14.46 µs | 17.79 µs +23% | 17.21 µs +19% |
| `deserialize_129dev` | 23.40 µs | 33.69 µs +44% | 26.86 µs +15% |

## training_memory

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `7B_full_finetune_gpu` | 3.15 ns | 5.11 ns +62% | 3.06 ns |
| `7B_lora_gpu` | 2.89 ns | 5.48 ns +89% | 3.06 ns +6% |
| `7B_qlora_4bit_gpu` | 2.91 ns | 4.35 ns +49% | 3.35 ns +15% |
| `7B_qlora_8bit_gpu` | 2.88 ns | 3.69 ns +28% | 3.52 ns +22% |
| `7B_prefix_gpu` | 3.11 ns | 4.55 ns +46% | 4.86 ns +56% |
| `7B_dpo_gpu` | 2.88 ns | 4.08 ns +42% | 5.14 ns +79% |
| `7B_rlhf_gpu` | 2.87 ns | 4.51 ns +57% | 5.17 ns +80% |
| `7B_distillation_gpu` | 2.64 ns | 4.48 ns +70% | 5.15 ns +95% |

## training_targets

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `7B_full_gpu` | 3.11 ns | 3.99 ns +28% | 4.25 ns +37% |
| `7B_full_tpu` | 2.51 ns | 3.60 ns +44% | 3.97 ns +58% |
| `7B_full_gaudi` | 2.49 ns | 4.65 ns +86% | 3.95 ns +58% |
| `7B_full_cpu` | 3.10 ns | 4.98 ns +61% | 3.74 ns +21% |

## training_model_sizes

| Benchmark | Baseline (`a281283`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `1B_lora_gpu` | 3.10 ns | 3.51 ns +13% | 5.37 ns +73% |
| `7B_lora_gpu` | 3.11 ns | 4.14 ns +33% | 5.62 ns +81% |
| `13B_lora_gpu` | 3.12 ns | 4.54 ns +46% | 5.21 ns +67% |
| `70B_lora_gpu` | 3.10 ns | 3.68 ns +19% | 5.38 ns +73% |
| `405B_lora_gpu` | 3.10 ns | 3.74 ns +21% | 2.80 ns **-10%** |

---

Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.
