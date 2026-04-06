# Benchmarks

Latest: **2026-04-06T04:10:23Z** — commit `84dfb0d`

Tracking: `9220e97` (baseline) → `84dfb0d` (previous) → `84dfb0d` (current)

## ungrouped

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `load_pricing_table` | 1.58 ns | 1.73 ns +9% | 1.56 ns |
| `detect_all` | 150.40 ms | 159.92 ms +6% | 145.55 ms **-3%** |
| `detect_none (CPU only)` | 150.33 ms | 157.82 ms +5% | 144.34 ms **-4%** |
| `concurrent_detect_4_threads` | 177.10 ms | 183.26 ms +3% | 167.69 ms **-5%** |
| `parse_nvidia_bandwidth_8gpu` | 1.67 µs | 1.10 µs **-34%** | 1.06 µs **-36%** |
| `nvidia_bus_width_all_ccs` | 661.50 ps | 267.60 ps **-60%** | 261.00 ps **-61%** |
| `estimate_bw_from_cc_all` | 272.80 ps | 266.40 ps | 262.10 ps **-4%** |
| `parse_max_dpm_clock` | 389.66 ns | 382.96 ns | 374.01 ns **-4%** |
| `parse_link_speed` | 80.87 ns | 81.56 ns | 78.70 ns |
| `parse_ib_rate` | 85.27 ns | 84.37 ns | 30.28 ns **-64%** |
| `parse_nvlink_output_2gpu` | 1.08 µs | 1.10 µs | 1.03 µs **-4%** |
| `parse_cuda_output_8gpu` | 6.27 µs | 6.01 µs **-4%** | 5.80 µs **-8%** |
| `parse_vulkan_output_2gpu` | 2.04 µs | 2.18 µs +7% | 1.85 µs **-9%** |
| `parse_gaudi_output_8dev` | 2.54 µs | 2.53 µs | 2.42 µs **-5%** |
| `plan_sharding 70B BF16 (4 GPU)` | 102.30 ns | 98.03 ns **-4%** | 97.81 ns **-4%** |
| `plan_sharding 70B BF16 (13 devices)` | 58.87 ns | 58.43 ns | 48.61 ns **-17%** |
| `suggest_quantization 70B (4 GPU)` | 6.25 ns | 6.75 ns +8% | 9.35 ns +50% |
| `suggest_quantization 70B (13 devices)` | 22.42 ns | 17.42 ns **-22%** | 22.38 ns |
| `estimate_memory 70B FP16` | 270.50 ps | 288.10 ps +7% | 257.20 ps **-5%** |
| `estimate_training_memory 7B LoRA GPU` | 614.30 ps | 557.80 ps **-9%** | 519.30 ps **-15%** |
| `best_available (13 devices)` | 48.38 ns | 49.41 ns | 39.56 ns **-18%** |
| `total_memory (13 devices)` | 9.23 ns | 8.95 ns | 7.88 ns **-15%** |
| `by_family GPU (13 devices)` | 11.88 ns | 11.80 ns | 7.88 ns **-34%** |
| `bits_per_param_all_levels` | 292.50 ps | 272.80 ps **-7%** | 294.90 ps |
| `memory_reduction_factor_all_levels` | 264.90 ps | 267.50 ps | 255.10 ps **-4%** |

## recommend_instance

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `7B_bf16_all` | 3.40 µs | 3.75 µs +10% | 3.36 µs |
| `7B_int8_aws` | 1.29 µs | 1.35 µs +5% | 1.23 µs **-5%** |
| `70B_bf16_all` | 1.60 µs | 1.78 µs +11% | 1.48 µs **-8%** |
| `70B_int4_gcp` | 642.33 ns | 712.39 ns +11% | 634.70 ns |

## cheapest_instance

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `7B_bf16` | 3.47 µs | 3.70 µs +7% | 3.44 µs |
| `70B_bf16` | 1.56 µs | 1.69 µs +8% | 1.64 µs +5% |

## detect_single

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `cuda` | 150.04 ms | 155.36 ms +4% | 144.74 ms **-4%** |
| `rocm` | 161.42 ms | 155.85 ms **-3%** | 144.83 ms **-10%** |
| `vulkan` | 150.29 ms | 155.32 ms +3% | 145.80 ms |
| `apple` | 146.34 ms | 151.96 ms +4% | 143.54 ms |
| `tpu` | 146.08 ms | 151.93 ms +4% | 143.78 ms |

## system_io

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `full_with_sysio` | 144.20 ms | 153.58 ms +7% | 174.97 ms +21% |
| `query_system_io` | 4.73 ns | 4.46 ns **-6%** | 4.51 ns **-5%** |
| `ingestion_1gb` | 4.68 ns | 4.51 ns **-4%** | 4.75 ns |
| `ingestion_100gb` | 4.70 ns | 4.54 ns **-3%** | 4.72 ns |
| `ingestion_1tb` | 4.96 ns | 4.76 ns **-4%** | 4.41 ns **-11%** |
| `serialize_registry` | 1.79 µs | 1.81 µs | 1.62 µs **-10%** |
| `deserialize_registry` | 2.20 µs | 2.32 µs +5% | 2.11 µs **-4%** |

## json_roundtrip

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `serialize (13 devices)` | 4.49 µs | 4.63 µs +3% | 4.89 µs +9% |
| `deserialize (13 devices)` | 6.90 µs | 6.30 µs **-9%** | 6.12 µs **-11%** |

## estimate_memory

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `70B_fp32` | 299.60 ps | 269.60 ps **-10%** | 257.70 ps **-14%** |
| `70B_fp16` | 272.70 ps | 266.20 ps | 255.10 ps **-6%** |
| `70B_bf16` | 303.00 ps | 276.50 ps **-9%** | 255.80 ps **-16%** |
| `70B_int8` | 270.30 ps | 266.30 ps | 254.90 ps **-6%** |
| `70B_int4` | 267.90 ps | 279.40 ps +4% | 255.00 ps **-5%** |

## suggest_quantization

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `1B_1gpu` | 5.06 ns | 4.87 ns **-4%** | 5.31 ns +5% |
| `7B_1gpu` | 4.87 ns | 5.26 ns +8% | 5.09 ns +5% |
| `13B_1gpu` | 4.72 ns | 4.85 ns | 6.15 ns +30% |
| `70B_1gpu` | 5.60 ns | 5.29 ns **-6%** | 5.51 ns |
| `405B_1gpu` | 4.96 ns | 4.76 ns **-4%** | 5.52 ns +11% |

## registry_queries

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `available_count_4dev` | 2.01 ns | 2.01 ns | 1.90 ns **-5%** |
| `available_count_129dev` | 51.51 ns | 51.35 ns | 44.71 ns **-13%** |
| `available_count_61dev_mixed` | 17.93 ns | 19.34 ns +8% | 17.19 ns **-4%** |
| `available_collect_4dev` | 73.34 ns | 79.72 ns +9% | 74.67 ns |
| `available_collect_129dev` | 723.16 ns | 614.03 ns **-15%** | 580.74 ns **-20%** |
| `available_collect_61dev_mixed` | 342.12 ns | 351.27 ns | 295.11 ns **-14%** |
| `best_available_129dev` | 383.92 ns | 392.47 ns | 433.34 ns +13% |
| `total_memory_129dev` | 120.17 ns | 114.61 ns **-5%** | 72.17 ns **-40%** |
| `total_accelerator_memory_129dev` | 79.38 ns | 79.68 ns | 74.27 ns **-6%** |
| `has_accelerator_129dev` | 1.36 ns | 1.33 ns | 1.28 ns **-5%** |
| `by_family_gpu_count_61dev` | 41.91 ns | 36.39 ns **-13%** | 34.99 ns **-17%** |
| `by_family_tpu_count_61dev` | 40.06 ns | 35.92 ns **-10%** | 34.84 ns **-13%** |
| `by_family_gpu_collect_61dev` | 356.57 ns | 311.75 ns **-13%** | 293.01 ns **-18%** |
| `by_family_tpu_collect_61dev` | 150.48 ns | 129.52 ns **-14%** | 126.94 ns **-16%** |
| `satisfying_gpu_count_61dev` | 137.26 ns | 136.79 ns | 159.52 ns +16% |
| `satisfying_any_accel_count_61dev` | 105.32 ns | 98.30 ns **-7%** | 127.18 ns +21% |
| `satisfying_gpu_collect_61dev` | 512.95 ns | 462.38 ns **-10%** | 402.60 ns **-22%** |
| `satisfying_any_accel_collect_61dev` | 529.50 ns | 398.08 ns **-25%** | 468.83 ns **-11%** |

## cached_registry

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `get_cached_hit` | 34.23 ns | 31.18 ns **-9%** | 31.65 ns **-8%** |
| `invalidate` | 6.46 ns | 6.05 ns **-6%** | 6.80 ns +5% |

## lazy_registry

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `new` | 11.32 µs | 11.08 µs | 10.84 µs **-4%** |
| `by_family_gpu_cold` | 155.56 ms | 143.09 ms **-8%** | 143.80 ms **-8%** |
| `by_family_gpu_warm` | 77.09 ns | 67.55 ns **-12%** | 65.85 ns **-15%** |
| `into_registry` | 647.65 ms | 602.94 ms **-7%** | 602.80 ms **-7%** |

## large_registry_sharding

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `plan_sharding_70B_128gpu` | 1.87 µs | 1.64 µs **-12%** | 1.65 µs **-12%** |
| `plan_sharding_405B_128gpu` | 1.83 µs | 1.68 µs **-8%** | 1.73 µs **-5%** |
| `suggest_quantization_70B_128gpu` | 405.33 ns | 369.62 ns **-9%** | 368.24 ns **-9%** |
| `plan_sharding_70B_mixed_61dev` | 197.56 ns | 183.34 ns **-7%** | 220.85 ns +12% |

## large_json

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `serialize_129dev` | 38.62 µs | 34.41 µs **-11%** | 35.70 µs **-8%** |
| `deserialize_129dev` | 52.22 µs | 45.95 µs **-12%** | 45.21 µs **-13%** |

## training_memory

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `7B_full_finetune_gpu` | 4.43 ns | 3.10 ns **-30%** | 3.29 ns **-26%** |
| `7B_lora_gpu` | 3.79 ns | 3.34 ns **-12%** | 3.34 ns **-12%** |
| `7B_qlora_4bit_gpu` | 3.42 ns | 3.09 ns **-10%** | 3.09 ns **-10%** |
| `7B_qlora_8bit_gpu` | 4.45 ns | 3.10 ns **-30%** | 3.08 ns **-31%** |
| `7B_prefix_gpu` | 3.90 ns | 3.37 ns **-14%** | 3.53 ns **-9%** |
| `7B_dpo_gpu` | 3.39 ns | 3.13 ns **-8%** | 3.09 ns **-9%** |
| `7B_rlhf_gpu` | 4.49 ns | 3.10 ns **-31%** | 3.24 ns **-28%** |
| `7B_distillation_gpu` | 3.28 ns | 2.85 ns **-13%** | 2.85 ns **-13%** |

## training_targets

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `7B_full_gpu` | 845.70 ps | 777.20 ps **-8%** | 770.00 ps **-9%** |
| `7B_full_tpu` | 838.90 ps | 781.30 ps **-7%** | 773.20 ps **-8%** |
| `7B_full_gaudi` | 856.70 ps | 780.10 ps **-9%** | 769.60 ps **-10%** |
| `7B_full_cpu` | 865.20 ps | 781.80 ps **-10%** | 771.50 ps **-11%** |

## training_model_sizes

| Benchmark | Baseline (`9220e97`) | Previous (`84dfb0d`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `1B_lora_gpu` | 1.09 ns | 1.03 ns **-5%** | 1.02 ns **-6%** |
| `7B_lora_gpu` | 1.08 ns | 1.03 ns **-5%** | 1.02 ns **-6%** |
| `13B_lora_gpu` | 1.09 ns | 1.03 ns **-6%** | 1.02 ns **-6%** |
| `70B_lora_gpu` | 1.11 ns | 1.03 ns **-7%** | 1.02 ns **-8%** |
| `405B_lora_gpu` | 1.16 ns | 1.02 ns **-11%** | 1.02 ns **-12%** |

---

Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.
