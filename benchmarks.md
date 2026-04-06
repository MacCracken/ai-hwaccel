# Benchmarks

Latest: **2026-04-05T22:34:16Z** — commit `234419b`

Tracking: `6fb0ccd` (baseline) → `6fb0ccd` (previous) → `234419b` (current)

## ungrouped

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `load_pricing_table` | 1.44 ns | 1.43 ns | 1.64 ns +14% |
| `detect_all` | 1.01 ms | 1.01 ms | 999.30 µs |
| `detect_none (CPU only)` | 141.55 µs | 139.51 µs | 158.51 µs +12% |
| `concurrent_detect_4_threads` | 2.48 ms | 2.51 ms | 3.00 ms +21% |
| `parse_nvidia_bandwidth_8gpu` | 969.76 ns | 977.20 ns | 1.09 µs +13% |
| `nvidia_bus_width_all_ccs` | 238.20 ps | 234.20 ps | 264.50 ps +11% |
| `estimate_bw_from_cc_all` | 237.20 ps | 235.40 ps | 264.30 ps +11% |
| `parse_max_dpm_clock` | 338.10 ns | 334.37 ns | 373.99 ns +11% |
| `parse_link_speed` | 73.38 ns | 72.66 ns | 80.64 ns +10% |
| `parse_ib_rate` | 72.81 ns | 72.09 ns | 80.50 ns +11% |
| `parse_nvlink_output_2gpu` | 974.87 ns | 944.74 ns **-3%** | 1.10 µs +13% |
| `parse_cuda_output_8gpu` | 11.51 µs | 5.45 µs **-53%** | 6.00 µs **-48%** |
| `parse_vulkan_output_2gpu` | 2.32 µs | 1.81 µs **-22%** | 2.09 µs **-10%** |
| `parse_gaudi_output_8dev` | 2.32 µs | 2.33 µs | 2.50 µs +8% |
| `plan_sharding 70B BF16 (4 GPU)` | 95.85 ns | 90.25 ns **-6%** | 106.87 ns +12% |
| `plan_sharding 70B BF16 (13 devices)` | 47.49 ns | 49.36 ns +4% | 54.91 ns +16% |
| `suggest_quantization 70B (4 GPU)` | 8.66 ns | 8.53 ns | 9.65 ns +11% |
| `suggest_quantization 70B (13 devices)` | 20.91 ns | 21.30 ns | 24.17 ns +16% |
| `estimate_memory 70B FP16` | 241.10 ps | 232.90 ps **-3%** | 277.30 ps +15% |
| `estimate_training_memory 7B LoRA GPU` | 499.30 ps | 480.60 ps **-4%** | 550.80 ps +10% |
| `best_available (13 devices)` | 37.35 ns | 36.29 ns | 46.01 ns +23% |
| `total_memory (13 devices)` | 10.85 ns | 10.38 ns **-4%** | 13.33 ns +23% |
| `by_family GPU (13 devices)` | 7.89 ns | 7.23 ns **-8%** | 9.21 ns +17% |
| `bits_per_param_all_levels` | 265.10 ps | 229.20 ps **-14%** | 287.70 ps +9% |
| `memory_reduction_factor_all_levels` | 263.80 ps | 232.40 ps **-12%** | 262.90 ps |

## recommend_instance

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `7B_bf16_all` | 3.16 µs | 3.24 µs | 3.74 µs +18% |
| `7B_int8_aws` | 1.11 µs | 1.14 µs | 1.19 µs +7% |
| `70B_bf16_all` | 1.36 µs | 1.39 µs | 1.57 µs +16% |
| `70B_int4_gcp` | 593.71 ns | 597.77 ns | 677.38 ns +14% |

## cheapest_instance

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `7B_bf16` | 3.12 µs | 3.15 µs | 3.31 µs +6% |
| `70B_bf16` | 1.40 µs | 1.44 µs | 1.59 µs +13% |

## detect_single

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `cuda` | 160.30 µs | 157.67 µs | 177.82 µs +11% |
| `rocm` | 1.17 ms | 1.24 ms +6% | 996.20 µs **-15%** |
| `vulkan` | 157.74 µs | 155.29 µs | 174.37 µs +11% |
| `apple` | 163.14 µs | 161.20 µs | 179.42 µs +10% |
| `tpu` | 179.74 µs | 179.98 µs | 196.09 µs +9% |

## system_io

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `full_with_sysio` | 1.01 ms | 1.01 ms | 1.00 ms |
| `query_system_io` | 3.69 ns | 3.65 ns | 3.94 ns +7% |
| `ingestion_1gb` | 3.94 ns | 3.91 ns | 4.29 ns +9% |
| `ingestion_100gb` | 3.73 ns | 3.69 ns | 4.05 ns +8% |
| `ingestion_1tb` | 3.96 ns | 3.92 ns | 4.31 ns +9% |
| `serialize_registry` | 1.61 µs | 1.60 µs | 1.73 µs +7% |
| `deserialize_registry` | 1.98 µs | 1.99 µs | 2.15 µs +9% |

## json_roundtrip

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `serialize (13 devices)` | 4.13 µs | 4.71 µs +14% | 5.18 µs +25% |
| `deserialize (13 devices)` | 5.81 µs | 5.51 µs **-5%** | 7.12 µs +23% |

## estimate_memory

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `70B_fp32` | 520.20 ps | 472.60 ps **-9%** | 591.10 ps +14% |
| `70B_fp16` | 548.50 ps | 472.10 ps **-14%** | 568.20 ps +4% |
| `70B_bf16` | 546.20 ps | 470.60 ps **-14%** | 585.20 ps +7% |
| `70B_int8` | 535.80 ps | 471.40 ps **-12%** | 591.30 ps +10% |
| `70B_int4` | 569.00 ps | 473.60 ps **-17%** | 578.30 ps |

## suggest_quantization

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `1B_1gpu` | 3.67 ns | 2.87 ns **-22%** | 3.76 ns |
| `7B_1gpu` | 3.81 ns | 2.86 ns **-25%** | 3.73 ns |
| `13B_1gpu` | 3.67 ns | 2.90 ns **-21%** | 3.76 ns |
| `70B_1gpu` | 4.73 ns | 4.41 ns **-7%** | 5.24 ns +11% |
| `405B_1gpu` | 4.19 ns | 4.58 ns +9% | 4.78 ns +14% |

## registry_queries

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `available_count_4dev` | 1.91 ns | 1.77 ns **-7%** | 2.09 ns +9% |
| `available_count_129dev` | 41.65 ns | 39.77 ns **-5%** | 47.67 ns +14% |
| `available_count_61dev_mixed` | 16.12 ns | 15.52 ns **-4%** | 17.74 ns +10% |
| `available_collect_4dev` | 65.03 ns | 61.07 ns **-6%** | 71.97 ns +11% |
| `available_collect_129dev` | 533.56 ns | 520.37 ns | 583.25 ns +9% |
| `available_collect_61dev_mixed` | 268.30 ns | 254.83 ns **-5%** | 300.58 ns +12% |
| `best_available_129dev` | 346.96 ns | 366.00 ns +5% | 388.00 ns +12% |
| `total_memory_129dev` | 66.80 ns | 71.72 ns +7% | 75.00 ns +12% |
| `total_accelerator_memory_129dev` | 74.53 ns | 79.32 ns +6% | 81.34 ns +9% |
| `has_accelerator_129dev` | 1.29 ns | 1.20 ns **-7%** | 1.36 ns +6% |
| `by_family_gpu_count_61dev` | 33.84 ns | 32.27 ns **-5%** | 36.63 ns +8% |
| `by_family_tpu_count_61dev` | 34.60 ns | 32.65 ns **-6%** | 36.82 ns +6% |
| `by_family_gpu_collect_61dev` | 305.55 ns | 270.00 ns **-12%** | 319.35 ns +5% |
| `by_family_tpu_collect_61dev` | 127.00 ns | 124.98 ns | 142.32 ns +12% |
| `satisfying_gpu_count_61dev` | 137.46 ns | 130.63 ns **-5%** | 146.78 ns +7% |
| `satisfying_any_accel_count_61dev` | 105.55 ns | 102.26 ns **-3%** | 115.94 ns +10% |
| `satisfying_gpu_collect_61dev` | 392.18 ns | 337.18 ns **-14%** | 387.83 ns |
| `satisfying_any_accel_collect_61dev` | 384.44 ns | 334.52 ns **-13%** | 410.19 ns +7% |

## cached_registry

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `get_cached_hit` | 29.87 ns | 29.01 ns | 31.47 ns +5% |
| `invalidate` | 5.97 ns | 5.32 ns **-11%** | 5.76 ns **-4%** |

## lazy_registry

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `new` | 10.14 µs | 9.78 µs **-4%** | 10.74 µs +6% |
| `by_family_gpu_cold` | 1.00 ms | 1.00 ms | 1.00 ms |
| `by_family_gpu_warm` | 62.41 ns | 61.20 ns | 66.65 ns +7% |
| `into_registry` | 2.01 ms | 3.22 ms +61% | 3.17 ms +58% |

## large_registry_sharding

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `plan_sharding_70B_128gpu` | 1.55 µs | 1.55 µs | 1.76 µs +14% |
| `plan_sharding_405B_128gpu` | 1.57 µs | 1.44 µs **-8%** | 1.61 µs |
| `suggest_quantization_70B_128gpu` | 382.33 ns | 363.40 ns **-5%** | 400.42 ns +5% |
| `plan_sharding_70B_mixed_61dev` | 179.45 ns | 173.52 ns **-3%** | 204.79 ns +14% |

## large_json

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `serialize_129dev` | 35.76 µs | 33.00 µs **-8%** | 36.07 µs |
| `deserialize_129dev` | 47.71 µs | 42.35 µs **-11%** | 46.82 µs |

## training_memory

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `7B_full_finetune_gpu` | 3.56 ns | 3.26 ns **-8%** | 3.60 ns |
| `7B_lora_gpu` | 3.65 ns | 3.05 ns **-16%** | 3.34 ns **-8%** |
| `7B_qlora_4bit_gpu` | 3.21 ns | 3.06 ns **-5%** | 3.35 ns +4% |
| `7B_qlora_8bit_gpu` | 3.39 ns | 3.06 ns **-10%** | 3.33 ns |
| `7B_prefix_gpu` | 5.92 ns | 3.28 ns **-45%** | 3.60 ns **-39%** |
| `7B_dpo_gpu` | 3.81 ns | 3.05 ns **-20%** | 3.34 ns **-12%** |
| `7B_rlhf_gpu` | 3.69 ns | 3.05 ns **-17%** | 3.55 ns **-4%** |
| `7B_distillation_gpu` | 3.50 ns | 2.83 ns **-19%** | 3.09 ns **-12%** |

## training_targets

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `7B_full_gpu` | 823.70 ps | 710.80 ps **-14%** | 780.00 ps **-5%** |
| `7B_full_tpu` | 746.80 ps | 705.80 ps **-5%** | 773.50 ps +4% |
| `7B_full_gaudi` | 745.40 ps | 708.10 ps **-5%** | 768.50 ps +3% |
| `7B_full_cpu` | 747.20 ps | 706.90 ps **-5%** | 777.10 ps +4% |

## training_model_sizes

| Benchmark | Baseline (`6fb0ccd`) | Previous (`6fb0ccd`) | Current (`234419b`) |
|-----------|------|------|------|
| `1B_lora_gpu` | 981.10 ps | 920.00 ps **-6%** | 1.02 ns +4% |
| `7B_lora_gpu` | 981.50 ps | 918.50 ps **-6%** | 1.02 ns +4% |
| `13B_lora_gpu` | 978.40 ps | 917.30 ps **-6%** | 1.02 ns +5% |
| `70B_lora_gpu` | 978.00 ps | 932.60 ps **-5%** | 1.02 ns +5% |
| `405B_lora_gpu` | 990.90 ps | 930.20 ps **-6%** | 1.02 ns +3% |

---

Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.
