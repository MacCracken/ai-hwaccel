# Benchmarks

Latest: **2026-04-03T15:34:28Z** — commit `6fb0ccd`

Tracking: `649834b` (baseline) → `6fb0ccd` (previous) → `6fb0ccd` (current)

## ungrouped

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `load_pricing_table` | 1.21 ns | 1.44 ns +19% | 1.43 ns +18% |
| `detect_all` | 998.24 µs | 1.01 ms | 1.01 ms |
| `detect_none (CPU only)` | 140.31 µs | 141.55 µs | 139.51 µs |
| `concurrent_detect_4_threads` | 2.50 ms | 2.48 ms | 2.51 ms |
| `parse_nvidia_bandwidth_8gpu` | 992.03 ns | 969.76 ns | 977.20 ns |
| `nvidia_bus_width_all_ccs` | 234.40 ps | 238.20 ps | 234.20 ps |
| `estimate_bw_from_cc_all` | 233.60 ps | 237.20 ps | 235.40 ps |
| `parse_max_dpm_clock` | 340.11 ns | 338.10 ns | 334.37 ns |
| `parse_link_speed` | 70.91 ns | 73.38 ns +3% | 72.66 ns |
| `parse_ib_rate` | 75.86 ns | 72.81 ns **-4%** | 72.09 ns **-5%** |
| `parse_nvlink_output_2gpu` | 936.89 ns | 974.87 ns +4% | 944.74 ns |
| `parse_cuda_output_8gpu` | 5.22 µs | 11.51 µs +120% | 5.45 µs +4% |
| `parse_vulkan_output_2gpu` | 1.78 µs | 2.32 µs +30% | 1.81 µs |
| `parse_gaudi_output_8dev` | 2.32 µs | 2.32 µs | 2.33 µs |
| `plan_sharding 70B BF16 (4 GPU)` | 90.09 ns | 95.85 ns +6% | 90.25 ns |
| `plan_sharding 70B BF16 (13 devices)` | 49.32 ns | 47.49 ns **-4%** | 49.36 ns |
| `suggest_quantization 70B (4 GPU)` | 5.41 ns | 8.66 ns +60% | 8.53 ns +58% |
| `suggest_quantization 70B (13 devices)` | 18.76 ns | 20.91 ns +12% | 21.30 ns +14% |
| `estimate_memory 70B FP16` | 241.10 ps | 241.10 ps | 232.90 ps **-3%** |
| `estimate_training_memory 7B LoRA GPU` | 967.00 ps | 499.30 ps **-48%** | 480.60 ps **-50%** |
| `best_available (13 devices)` | 37.36 ns | 37.35 ns | 36.29 ns |
| `total_memory (13 devices)` | 7.75 ns | 10.85 ns +40% | 10.38 ns +34% |
| `by_family GPU (13 devices)` | 11.11 ns | 7.89 ns **-29%** | 7.23 ns **-35%** |
| `bits_per_param_all_levels` | 235.70 ps | 265.10 ps +12% | 229.20 ps |
| `memory_reduction_factor_all_levels` | 235.40 ps | 263.80 ps +12% | 232.40 ps |

## recommend_instance

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `7B_bf16_all` | 3.07 µs | 3.16 µs +3% | 3.24 µs +5% |
| `7B_int8_aws` | 1.10 µs | 1.11 µs | 1.14 µs +4% |
| `70B_bf16_all` | 1.40 µs | 1.36 µs | 1.39 µs |
| `70B_int4_gcp` | 588.81 ns | 593.71 ns | 597.77 ns |

## cheapest_instance

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `7B_bf16` | 3.07 µs | 3.12 µs | 3.15 µs |
| `70B_bf16` | 1.41 µs | 1.40 µs | 1.44 µs |

## detect_single

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `cuda` | 157.93 µs | 160.30 µs | 157.67 µs |
| `rocm` | 1.26 ms | 1.17 ms **-7%** | 1.24 ms |
| `vulkan` | 155.45 µs | 157.74 µs | 155.29 µs |
| `apple` | 161.72 µs | 163.14 µs | 161.20 µs |
| `tpu` | 179.27 µs | 179.74 µs | 179.98 µs |

## system_io

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `full_with_sysio` | 1.01 ms | 1.01 ms | 1.01 ms |
| `query_system_io` | 4.65 ns | 3.69 ns **-21%** | 3.65 ns **-21%** |
| `ingestion_1gb` | 4.71 ns | 3.94 ns **-16%** | 3.91 ns **-17%** |
| `ingestion_100gb` | 4.71 ns | 3.73 ns **-21%** | 3.69 ns **-22%** |
| `ingestion_1tb` | 4.71 ns | 3.96 ns **-16%** | 3.92 ns **-17%** |
| `serialize_registry` | 1.51 µs | 1.61 µs +7% | 1.60 µs +6% |
| `deserialize_registry` | 2.04 µs | 1.98 µs **-3%** | 1.99 µs |

## json_roundtrip

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `serialize (13 devices)` | 4.58 µs | 4.13 µs **-10%** | 4.71 µs |
| `deserialize (13 devices)` | 6.20 µs | 5.81 µs **-6%** | 5.51 µs **-11%** |

## estimate_memory

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `70B_fp32` | 495.50 ps | 520.20 ps +5% | 472.60 ps **-5%** |
| `70B_fp16` | 481.60 ps | 548.50 ps +14% | 472.10 ps |
| `70B_bf16` | 482.40 ps | 546.20 ps +13% | 470.60 ps |
| `70B_int8` | 478.80 ps | 535.80 ps +12% | 471.40 ps |
| `70B_int4` | 496.70 ps | 569.00 ps +15% | 473.60 ps **-5%** |

## suggest_quantization

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `1B_1gpu` | 4.45 ns | 3.67 ns **-18%** | 2.87 ns **-36%** |
| `7B_1gpu` | 4.75 ns | 3.81 ns **-20%** | 2.86 ns **-40%** |
| `13B_1gpu` | 4.68 ns | 3.67 ns **-22%** | 2.90 ns **-38%** |
| `70B_1gpu` | 6.36 ns | 4.73 ns **-26%** | 4.41 ns **-31%** |
| `405B_1gpu` | 4.82 ns | 4.19 ns **-13%** | 4.58 ns **-5%** |

## registry_queries

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `available_4dev` | 1.84 ns | — | — |
| `available_129dev` | 39.53 ns | — | — |
| `available_61dev_mixed` | 15.84 ns | — | — |
| `best_available_129dev` | 372.51 ns | 346.96 ns **-7%** | 366.00 ns |
| `total_memory_129dev` | 68.61 ns | 66.80 ns | 71.72 ns +5% |
| `total_accelerator_memory_129dev` | 70.69 ns | 74.53 ns +5% | 79.32 ns +12% |
| `has_accelerator_129dev` | 1.22 ns | 1.29 ns +5% | 1.20 ns |
| `by_family_gpu_61dev` | 45.50 ns | — | — |
| `by_family_tpu_61dev` | 45.33 ns | — | — |
| `satisfying_gpu_61dev` | 73.87 ns | — | — |
| `satisfying_any_accel_61dev` | 45.68 ns | — | — |
| `available_count_4dev` | — | 1.91 ns | 1.77 ns |
| `available_count_129dev` | — | 41.65 ns | 39.77 ns |
| `available_count_61dev_mixed` | — | 16.12 ns | 15.52 ns |
| `available_collect_4dev` | — | 65.03 ns | 61.07 ns |
| `available_collect_129dev` | — | 533.56 ns | 520.37 ns |
| `available_collect_61dev_mixed` | — | 268.30 ns | 254.83 ns |
| `by_family_gpu_count_61dev` | — | 33.84 ns | 32.27 ns |
| `by_family_tpu_count_61dev` | — | 34.60 ns | 32.65 ns |
| `by_family_gpu_collect_61dev` | — | 305.55 ns | 270.00 ns |
| `by_family_tpu_collect_61dev` | — | 127.00 ns | 124.98 ns |
| `satisfying_gpu_count_61dev` | — | 137.46 ns | 130.63 ns |
| `satisfying_any_accel_count_61dev` | — | 105.55 ns | 102.26 ns |
| `satisfying_gpu_collect_61dev` | — | 392.18 ns | 337.18 ns |
| `satisfying_any_accel_collect_61dev` | — | 384.44 ns | 334.52 ns |

## cached_registry

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `get_cached_hit` | 28.79 ns | 29.87 ns +4% | 29.01 ns |
| `invalidate` | 6.03 ns | 5.97 ns | 5.32 ns **-12%** |

## lazy_registry

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `new` | 9.70 µs | 10.14 µs +5% | 9.78 µs |
| `by_family_gpu_cold` | 1.00 ms | 1.00 ms | 1.00 ms |
| `by_family_gpu_warm` | 62.93 ns | 62.41 ns | 61.20 ns |
| `into_registry` | 2.00 ms | 2.01 ms | 3.22 ms +61% |

## large_registry_sharding

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `plan_sharding_70B_128gpu` | 1.60 µs | 1.55 µs **-3%** | 1.55 µs **-3%** |
| `plan_sharding_405B_128gpu` | 1.60 µs | 1.57 µs | 1.44 µs **-10%** |
| `suggest_quantization_70B_128gpu` | 366.38 ns | 382.33 ns +4% | 363.40 ns |
| `plan_sharding_70B_mixed_61dev` | 193.31 ns | 179.45 ns **-7%** | 173.52 ns **-10%** |

## large_json

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `serialize_129dev` | 35.23 µs | 35.76 µs | 33.00 µs **-6%** |
| `deserialize_129dev` | 42.70 µs | 47.71 µs +12% | 42.35 µs |

## training_memory

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `7B_full_finetune_gpu` | 2.85 ns | 3.56 ns +25% | 3.26 ns +15% |
| `7B_lora_gpu` | 2.85 ns | 3.65 ns +28% | 3.05 ns +7% |
| `7B_qlora_4bit_gpu` | 3.07 ns | 3.21 ns +5% | 3.06 ns |
| `7B_qlora_8bit_gpu` | 3.07 ns | 3.39 ns +11% | 3.06 ns |
| `7B_prefix_gpu` | 3.05 ns | 5.92 ns +94% | 3.28 ns +8% |
| `7B_dpo_gpu` | 3.07 ns | 3.81 ns +24% | 3.05 ns |
| `7B_rlhf_gpu` | 3.06 ns | 3.69 ns +21% | 3.05 ns |
| `7B_distillation_gpu` | 2.84 ns | 3.50 ns +23% | 2.83 ns |

## training_targets

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `7B_full_gpu` | 1.17 ns | 823.70 ps **-30%** | 710.80 ps **-39%** |
| `7B_full_tpu` | 1.17 ns | 746.80 ps **-36%** | 705.80 ps **-40%** |
| `7B_full_gaudi` | 1.17 ns | 745.40 ps **-36%** | 708.10 ps **-39%** |
| `7B_full_cpu` | 1.17 ns | 747.20 ps **-36%** | 706.90 ps **-39%** |

## training_model_sizes

| Benchmark | Baseline (`649834b`) | Previous (`6fb0ccd`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `1B_lora_gpu` | 938.90 ps | 981.10 ps +4% | 920.00 ps |
| `7B_lora_gpu` | 937.90 ps | 981.50 ps +5% | 918.50 ps |
| `13B_lora_gpu` | 939.20 ps | 978.40 ps +4% | 917.30 ps |
| `70B_lora_gpu` | 939.60 ps | 978.00 ps +4% | 932.60 ps |
| `405B_lora_gpu` | 939.00 ps | 990.90 ps +6% | 930.20 ps |

---

Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.
