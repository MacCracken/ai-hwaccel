# Benchmarks

Latest: **2026-04-06T03:40:41Z** — commit `84dfb0d`

Tracking: `234419b` (baseline) → `9220e97` (previous) → `84dfb0d` (current)

## ungrouped

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `load_pricing_table` | 1.64 ns | 1.58 ns **-3%** | 1.73 ns +5% |
| `detect_all` | 999.30 µs | 150.40 ms +14951% | 159.92 ms +15903% |
| `detect_none (CPU only)` | 158.51 µs | 150.33 ms +94739% | 157.82 ms +99465% |
| `concurrent_detect_4_threads` | 3.00 ms | 177.10 ms +5811% | 183.26 ms +6016% |
| `parse_nvidia_bandwidth_8gpu` | 1.09 µs | 1.67 µs +53% | 1.10 µs |
| `nvidia_bus_width_all_ccs` | 264.50 ps | 661.50 ps +150% | 267.60 ps |
| `estimate_bw_from_cc_all` | 264.30 ps | 272.80 ps +3% | 266.40 ps |
| `parse_max_dpm_clock` | 373.99 ns | 389.66 ns +4% | 382.96 ns |
| `parse_link_speed` | 80.64 ns | 80.87 ns | 81.56 ns |
| `parse_ib_rate` | 80.50 ns | 85.27 ns +6% | 84.37 ns +5% |
| `parse_nvlink_output_2gpu` | 1.10 µs | 1.08 µs | 1.10 µs |
| `parse_cuda_output_8gpu` | 6.00 µs | 6.27 µs +5% | 6.01 µs |
| `parse_vulkan_output_2gpu` | 2.09 µs | 2.04 µs | 2.18 µs +4% |
| `parse_gaudi_output_8dev` | 2.50 µs | 2.54 µs | 2.53 µs |
| `plan_sharding 70B BF16 (4 GPU)` | 106.87 ns | 102.30 ns **-4%** | 98.03 ns **-8%** |
| `plan_sharding 70B BF16 (13 devices)` | 54.91 ns | 58.87 ns +7% | 58.43 ns +6% |
| `suggest_quantization 70B (4 GPU)` | 9.65 ns | 6.25 ns **-35%** | 6.75 ns **-30%** |
| `suggest_quantization 70B (13 devices)` | 24.17 ns | 22.42 ns **-7%** | 17.42 ns **-28%** |
| `estimate_memory 70B FP16` | 277.30 ps | 270.50 ps | 288.10 ps +4% |
| `estimate_training_memory 7B LoRA GPU` | 550.80 ps | 614.30 ps +12% | 557.80 ps |
| `best_available (13 devices)` | 46.01 ns | 48.38 ns +5% | 49.41 ns +7% |
| `total_memory (13 devices)` | 13.33 ns | 9.23 ns **-31%** | 8.95 ns **-33%** |
| `by_family GPU (13 devices)` | 9.21 ns | 11.88 ns +29% | 11.80 ns +28% |
| `bits_per_param_all_levels` | 287.70 ps | 292.50 ps | 272.80 ps **-5%** |
| `memory_reduction_factor_all_levels` | 262.90 ps | 264.90 ps | 267.50 ps |

## recommend_instance

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `7B_bf16_all` | 3.74 µs | 3.40 µs **-9%** | 3.75 µs |
| `7B_int8_aws` | 1.19 µs | 1.29 µs +8% | 1.35 µs +13% |
| `70B_bf16_all` | 1.57 µs | 1.60 µs | 1.78 µs +14% |
| `70B_int4_gcp` | 677.38 ns | 642.33 ns **-5%** | 712.39 ns +5% |

## cheapest_instance

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `7B_bf16` | 3.31 µs | 3.47 µs +5% | 3.70 µs +12% |
| `70B_bf16` | 1.59 µs | 1.56 µs | 1.69 µs +6% |

## detect_single

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `cuda` | 177.82 µs | 150.04 ms +84277% | 155.36 ms +87269% |
| `rocm` | 996.20 µs | 161.42 ms +16104% | 155.85 ms +15544% |
| `vulkan` | 174.37 µs | 150.29 ms +86090% | 155.32 ms +88975% |
| `apple` | 179.42 µs | 146.34 ms +81463% | 151.96 ms +84595% |
| `tpu` | 196.09 µs | 146.08 ms +74396% | 151.93 ms +77380% |

## system_io

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `full_with_sysio` | 1.00 ms | 144.20 ms +14313% | 153.58 ms +15250% |
| `query_system_io` | 3.94 ns | 4.73 ns +20% | 4.46 ns +13% |
| `ingestion_1gb` | 4.29 ns | 4.68 ns +9% | 4.51 ns +5% |
| `ingestion_100gb` | 4.05 ns | 4.70 ns +16% | 4.54 ns +12% |
| `ingestion_1tb` | 4.31 ns | 4.96 ns +15% | 4.76 ns +10% |
| `serialize_registry` | 1.73 µs | 1.79 µs +4% | 1.81 µs +5% |
| `deserialize_registry` | 2.15 µs | 2.20 µs | 2.32 µs +8% |

## json_roundtrip

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `serialize (13 devices)` | 5.18 µs | 4.49 µs **-13%** | 4.63 µs **-11%** |
| `deserialize (13 devices)` | 7.12 µs | 6.90 µs | 6.30 µs **-11%** |

## estimate_memory

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `70B_fp32` | 591.10 ps | 299.60 ps **-49%** | 269.60 ps **-54%** |
| `70B_fp16` | 568.20 ps | 272.70 ps **-52%** | 266.20 ps **-53%** |
| `70B_bf16` | 585.20 ps | 303.00 ps **-48%** | 276.50 ps **-53%** |
| `70B_int8` | 591.30 ps | 270.30 ps **-54%** | 266.30 ps **-55%** |
| `70B_int4` | 578.30 ps | 267.90 ps **-54%** | 279.40 ps **-52%** |

## suggest_quantization

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `1B_1gpu` | 3.76 ns | 5.06 ns +34% | 4.87 ns +29% |
| `7B_1gpu` | 3.73 ns | 4.87 ns +30% | 5.26 ns +41% |
| `13B_1gpu` | 3.76 ns | 4.72 ns +26% | 4.85 ns +29% |
| `70B_1gpu` | 5.24 ns | 5.60 ns +7% | 5.29 ns |
| `405B_1gpu` | 4.78 ns | 4.96 ns +4% | 4.76 ns |

## registry_queries

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `available_count_4dev` | 2.09 ns | 2.01 ns **-4%** | 2.01 ns **-4%** |
| `available_count_129dev` | 47.67 ns | 51.51 ns +8% | 51.35 ns +8% |
| `available_count_61dev_mixed` | 17.74 ns | 17.93 ns | 19.34 ns +9% |
| `available_collect_4dev` | 71.97 ns | 73.34 ns | 79.72 ns +11% |
| `available_collect_129dev` | 583.25 ns | 723.16 ns +24% | 614.03 ns +5% |
| `available_collect_61dev_mixed` | 300.58 ns | 342.12 ns +14% | 351.27 ns +17% |
| `best_available_129dev` | 388.00 ns | 383.92 ns | 392.47 ns |
| `total_memory_129dev` | 75.00 ns | 120.17 ns +60% | 114.61 ns +53% |
| `total_accelerator_memory_129dev` | 81.34 ns | 79.38 ns | 79.68 ns |
| `has_accelerator_129dev` | 1.36 ns | 1.36 ns | 1.33 ns |
| `by_family_gpu_count_61dev` | 36.63 ns | 41.91 ns +14% | 36.39 ns |
| `by_family_tpu_count_61dev` | 36.82 ns | 40.06 ns +9% | 35.92 ns |
| `by_family_gpu_collect_61dev` | 319.35 ns | 356.57 ns +12% | 311.75 ns |
| `by_family_tpu_collect_61dev` | 142.32 ns | 150.48 ns +6% | 129.52 ns **-9%** |
| `satisfying_gpu_count_61dev` | 146.78 ns | 137.26 ns **-6%** | 136.79 ns **-7%** |
| `satisfying_any_accel_count_61dev` | 115.94 ns | 105.32 ns **-9%** | 98.30 ns **-15%** |
| `satisfying_gpu_collect_61dev` | 387.83 ns | 512.95 ns +32% | 462.38 ns +19% |
| `satisfying_any_accel_collect_61dev` | 410.19 ns | 529.50 ns +29% | 398.08 ns |

## cached_registry

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `get_cached_hit` | 31.47 ns | 34.23 ns +9% | 31.18 ns |
| `invalidate` | 5.76 ns | 6.46 ns +12% | 6.05 ns +5% |

## lazy_registry

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `new` | 10.74 µs | 11.32 µs +5% | 11.08 µs +3% |
| `by_family_gpu_cold` | 1.00 ms | 155.56 ms +15399% | 143.09 ms +14156% |
| `by_family_gpu_warm` | 66.65 ns | 77.09 ns +16% | 67.55 ns |
| `into_registry` | 3.17 ms | 647.65 ms +20332% | 602.94 ms +18921% |

## large_registry_sharding

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `plan_sharding_70B_128gpu` | 1.76 µs | 1.87 µs +6% | 1.64 µs **-7%** |
| `plan_sharding_405B_128gpu` | 1.61 µs | 1.83 µs +13% | 1.68 µs +4% |
| `suggest_quantization_70B_128gpu` | 400.42 ns | 405.33 ns | 369.62 ns **-8%** |
| `plan_sharding_70B_mixed_61dev` | 204.79 ns | 197.56 ns **-4%** | 183.34 ns **-10%** |

## large_json

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `serialize_129dev` | 36.07 µs | 38.62 µs +7% | 34.41 µs **-5%** |
| `deserialize_129dev` | 46.82 µs | 52.22 µs +12% | 45.95 µs |

## training_memory

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `7B_full_finetune_gpu` | 3.60 ns | 4.43 ns +23% | 3.10 ns **-14%** |
| `7B_lora_gpu` | 3.34 ns | 3.79 ns +13% | 3.34 ns |
| `7B_qlora_4bit_gpu` | 3.35 ns | 3.42 ns | 3.09 ns **-8%** |
| `7B_qlora_8bit_gpu` | 3.33 ns | 4.45 ns +34% | 3.10 ns **-7%** |
| `7B_prefix_gpu` | 3.60 ns | 3.90 ns +8% | 3.37 ns **-6%** |
| `7B_dpo_gpu` | 3.34 ns | 3.39 ns | 3.13 ns **-6%** |
| `7B_rlhf_gpu` | 3.55 ns | 4.49 ns +26% | 3.10 ns **-13%** |
| `7B_distillation_gpu` | 3.09 ns | 3.28 ns +6% | 2.85 ns **-8%** |

## training_targets

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `7B_full_gpu` | 780.00 ps | 845.70 ps +8% | 777.20 ps |
| `7B_full_tpu` | 773.50 ps | 838.90 ps +8% | 781.30 ps |
| `7B_full_gaudi` | 768.50 ps | 856.70 ps +11% | 780.10 ps |
| `7B_full_cpu` | 777.10 ps | 865.20 ps +11% | 781.80 ps |

## training_model_sizes

| Benchmark | Baseline (`234419b`) | Previous (`9220e97`) | Current (`84dfb0d`) |
|-----------|------|------|------|
| `1B_lora_gpu` | 1.02 ns | 1.09 ns +6% | 1.03 ns |
| `7B_lora_gpu` | 1.02 ns | 1.08 ns +6% | 1.03 ns |
| `13B_lora_gpu` | 1.02 ns | 1.09 ns +7% | 1.03 ns |
| `70B_lora_gpu` | 1.02 ns | 1.11 ns +8% | 1.03 ns |
| `405B_lora_gpu` | 1.02 ns | 1.16 ns +13% | 1.02 ns |

---

Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.
