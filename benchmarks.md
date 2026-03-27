# Benchmarks

Latest: **2026-03-27T16:19:36Z** — commit `19a2753`

Tracking: `19a2753` (baseline) → `19a2753` (previous) → `19a2753` (current)

## ungrouped

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `load_pricing_table` | 2.76 ns | 2.01 ns **-27%** | 1.92 ns **-30%** |
| `detect_all` | 1.12 ms | 2.97 ms +164% | 1.09 ms |
| `detect_none (CPU only)` | 156.47 µs | 156.28 µs | 160.72 µs |
| `concurrent_detect_4_threads` | 5.09 ms | 3.86 ms **-24%** | 5.46 ms +7% |
| `parse_nvidia_bandwidth_8gpu` | 678.64 ns | 619.44 ns **-9%** | 675.51 ns |
| `nvidia_bus_width_all_ccs` | 25.26 ns | 21.58 ns **-15%** | 23.55 ns **-7%** |
| `estimate_bw_from_cc_all` | 27.44 ns | 26.90 ns | 24.40 ns **-11%** |
| `parse_max_dpm_clock` | 161.97 ns | 129.82 ns **-20%** | 146.63 ns **-9%** |
| `parse_link_speed` | 56.77 ns | 43.42 ns **-24%** | 57.03 ns |
| `parse_ib_rate` | 62.72 ns | 50.80 ns **-19%** | 89.87 ns +43% |
| `parse_nvlink_output_2gpu` | 695.90 ns | 620.21 ns **-11%** | 1.67 µs +141% |
| `parse_cuda_output_8gpu` | 4.26 µs | 3.80 µs **-11%** | 7.83 µs +84% |
| `parse_vulkan_output_2gpu` | 1.03 µs | 1.07 µs +3% | 2.01 µs +95% |
| `parse_gaudi_output_8dev` | 1.67 µs | 1.58 µs **-5%** | 2.89 µs +73% |
| `plan_sharding 70B BF16 (4 GPU)` | 104.92 ns | 67.84 ns **-35%** | 75.03 ns **-28%** |
| `plan_sharding 70B BF16 (13 devices)` | 38.73 ns | 32.52 ns **-16%** | 36.91 ns **-5%** |
| `suggest_quantization 70B (4 GPU)` | 10.31 ns | 10.81 ns +5% | 11.03 ns +7% |
| `suggest_quantization 70B (13 devices)` | 29.20 ns | 25.56 ns **-12%** | 23.80 ns **-18%** |
| `estimate_memory 70B FP16` | 281.10 ps | 263.80 ps **-6%** | 260.80 ps **-7%** |
| `estimate_training_memory 7B LoRA GPU` | 3.08 ns | 3.23 ns +5% | 2.79 ns **-9%** |
| `best_available (13 devices)` | 27.94 ns | 22.54 ns **-19%** | 25.58 ns **-8%** |
| `total_memory (13 devices)` | 6.12 ns | 6.45 ns +5% | 5.83 ns **-5%** |
| `by_family GPU (13 devices)` | 89.71 ns | 59.60 ns **-34%** | 60.00 ns **-33%** |
| `bits_per_param_all_levels` | 53.70 ps | 33.70 ps **-37%** | 38.10 ps **-29%** |
| `memory_reduction_factor_all_levels` | 39.70 ps | 34.20 ps **-14%** | 33.70 ps **-15%** |

## recommend_instance

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `7B_bf16_all` | 5.23 µs | 2.81 µs **-46%** | 2.69 µs **-49%** |
| `7B_int8_aws` | 1.74 µs | 1.03 µs **-41%** | 972.12 ns **-44%** |
| `70B_bf16_all` | 1.80 µs | 1.23 µs **-32%** | 1.28 µs **-29%** |
| `70B_int4_gcp` | 688.95 ns | 546.40 ns **-21%** | 502.66 ns **-27%** |

## cheapest_instance

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `7B_bf16` | 3.16 µs | 2.92 µs **-7%** | 2.69 µs **-15%** |
| `70B_bf16` | 1.39 µs | 1.25 µs **-10%** | 1.25 µs **-11%** |

## detect_single

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `cuda` | 179.43 µs | 187.68 µs +5% | 251.48 µs +40% |
| `rocm` | 995.00 µs | 994.93 µs | 1.01 ms |
| `vulkan` | 188.28 µs | 180.33 µs **-4%** | 265.77 µs +41% |
| `apple` | 175.63 µs | 174.09 µs | 183.32 µs +4% |
| `tpu` | 204.95 µs | 210.77 µs | 197.89 µs **-3%** |

## system_io

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `full_with_sysio` | 1.02 ms | 1.00 ms | 1.33 ms +31% |
| `query_system_io` | 34.30 ps | 37.50 ps +9% | 71.20 ps +108% |
| `ingestion_1gb` | 2.14 ns | 2.32 ns +8% | 4.10 ns +91% |
| `ingestion_100gb` | 2.57 ns | 2.26 ns **-12%** | 5.01 ns +95% |
| `ingestion_1tb` | 2.54 ns | 2.33 ns **-8%** | 3.06 ns +21% |
| `serialize_registry` | 1.20 µs | 984.66 ns **-18%** | 1.02 µs **-15%** |
| `deserialize_registry` | 1.66 µs | 1.39 µs **-16%** | 2.36 µs +43% |

## json_roundtrip

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `serialize (13 devices)` | 2.47 µs | 2.22 µs **-10%** | 2.21 µs **-11%** |
| `deserialize (13 devices)` | 4.46 µs | 3.54 µs **-21%** | 3.54 µs **-21%** |

## estimate_memory

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `70B_fp32` | 287.40 ps | 279.90 ps | 260.50 ps **-9%** |
| `70B_fp16` | 300.80 ps | 275.30 ps **-8%** | 266.10 ps **-12%** |
| `70B_bf16` | 286.30 ps | 272.30 ps **-5%** | 263.10 ps **-8%** |
| `70B_int8` | 297.30 ps | 287.80 ps **-3%** | 262.00 ps **-12%** |
| `70B_int4` | 307.50 ps | 262.50 ps **-15%** | 261.00 ps **-15%** |

## suggest_quantization

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `1B_1gpu` | 7.25 ns | 5.60 ns **-23%** | 6.10 ns **-16%** |
| `7B_1gpu` | 7.42 ns | 5.63 ns **-24%** | 6.25 ns **-16%** |
| `13B_1gpu` | 7.67 ns | 5.61 ns **-27%** | 6.69 ns **-13%** |
| `70B_1gpu` | 7.58 ns | 5.82 ns **-23%** | 6.68 ns **-12%** |
| `405B_1gpu` | 8.01 ns | 5.81 ns **-28%** | 9.76 ns +22% |

## registry_queries

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `available_4dev` | 73.59 ns | 60.88 ns **-17%** | 68.44 ns **-7%** |
| `available_129dev` | 465.96 ns | 424.73 ns **-9%** | 412.03 ns **-12%** |
| `available_61dev_mixed` | 224.84 ns | 215.21 ns **-4%** | 227.41 ns |
| `best_available_129dev` | 164.74 ns | 114.14 ns **-31%** | 163.23 ns |
| `total_memory_129dev` | 53.28 ns | 43.71 ns **-18%** | 46.07 ns **-14%** |
| `total_accelerator_memory_129dev` | 61.59 ns | 70.68 ns +15% | 75.32 ns +22% |
| `has_accelerator_129dev` | 1.70 ns | 1.85 ns +9% | 1.44 ns **-15%** |
| `by_family_gpu_61dev` | 336.42 ns | 205.52 ns **-39%** | 627.09 ns +86% |
| `by_family_tpu_61dev` | 119.05 ns | 88.53 ns **-26%** | 125.15 ns +5% |
| `satisfying_gpu_61dev` | 280.17 ns | 240.19 ns **-14%** | 252.99 ns **-10%** |
| `satisfying_any_accel_61dev` | 322.37 ns | 260.85 ns **-19%** | 288.71 ns **-10%** |

## cached_registry

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `get_cached_hit` | 37.10 ns | 31.21 ns **-16%** | 31.97 ns **-14%** |
| `invalidate` | 6.39 ns | 5.27 ns **-18%** | 4.15 ns **-35%** |

## lazy_registry

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `new` | 13.69 µs | 13.11 µs **-4%** | 12.83 µs **-6%** |
| `by_family_gpu_cold` | 1.06 ms | 1.20 ms +13% | 1.63 ms +54% |
| `by_family_gpu_warm` | 60.55 ns | 61.89 ns | 54.30 ns **-10%** |
| `into_registry` | 2.02 ms | 2.01 ms | 2.33 ms +15% |

## large_registry_sharding

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `plan_sharding_70B_128gpu` | 1.71 µs | 1.18 µs **-31%** | 1.30 µs **-24%** |
| `plan_sharding_405B_128gpu` | 1.88 µs | 1.14 µs **-39%** | 1.25 µs **-34%** |
| `suggest_quantization_70B_128gpu` | 218.85 ns | 241.30 ns +10% | 239.10 ns +9% |
| `plan_sharding_70B_mixed_61dev` | 154.98 ns | 137.14 ns **-12%** | 160.47 ns +4% |

## large_json

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `serialize_129dev` | 17.79 µs | 17.21 µs **-3%** | 21.89 µs +23% |
| `deserialize_129dev` | 33.69 µs | 26.86 µs **-20%** | 31.04 µs **-8%** |

## training_memory

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `7B_full_finetune_gpu` | 5.11 ns | 3.06 ns **-40%** | 3.91 ns **-23%** |
| `7B_lora_gpu` | 5.48 ns | 3.06 ns **-44%** | 3.97 ns **-28%** |
| `7B_qlora_4bit_gpu` | 4.35 ns | 3.35 ns **-23%** | 5.33 ns +23% |
| `7B_qlora_8bit_gpu` | 3.69 ns | 3.52 ns **-5%** | 4.32 ns +17% |
| `7B_prefix_gpu` | 4.55 ns | 4.86 ns +7% | 5.95 ns +31% |
| `7B_dpo_gpu` | 4.08 ns | 5.14 ns +26% | 5.28 ns +29% |
| `7B_rlhf_gpu` | 4.51 ns | 5.17 ns +15% | 4.78 ns +6% |
| `7B_distillation_gpu` | 4.48 ns | 5.15 ns +15% | 4.93 ns +10% |

## training_targets

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `7B_full_gpu` | 3.99 ns | 4.25 ns +7% | 6.12 ns +54% |
| `7B_full_tpu` | 3.60 ns | 3.97 ns +10% | 7.42 ns +106% |
| `7B_full_gaudi` | 4.65 ns | 3.95 ns **-15%** | 8.12 ns +75% |
| `7B_full_cpu` | 4.98 ns | 3.74 ns **-25%** | 4.13 ns **-17%** |

## training_model_sizes

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`19a2753`) |
|-----------|------|------|------|
| `1B_lora_gpu` | 3.51 ns | 5.37 ns +53% | 6.16 ns +76% |
| `7B_lora_gpu` | 4.14 ns | 5.62 ns +36% | 3.60 ns **-13%** |
| `13B_lora_gpu` | 4.54 ns | 5.21 ns +15% | 4.18 ns **-8%** |
| `70B_lora_gpu` | 3.68 ns | 5.38 ns +46% | 4.64 ns +26% |
| `405B_lora_gpu` | 3.74 ns | 2.80 ns **-25%** | 5.05 ns +35% |

---

Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.
