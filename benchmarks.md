# Benchmarks

Latest: **2026-04-03T15:09:19Z** — commit `6fb0ccd`

Tracking: `649834b` (baseline) → `649834b` (previous) → `6fb0ccd` (current)

## ungrouped

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `load_pricing_table` | 1.24 ns | 1.21 ns | 1.44 ns +16% |
| `detect_all` | 1.01 ms | 998.24 µs | 1.01 ms |
| `detect_none (CPU only)` | 142.68 µs | 140.31 µs | 141.55 µs |
| `concurrent_detect_4_threads` | 4.41 ms | 2.50 ms **-43%** | 2.48 ms **-44%** |
| `parse_nvidia_bandwidth_8gpu` | 1.15 µs | 992.03 ns **-13%** | 969.76 ns **-15%** |
| `nvidia_bus_width_all_ccs` | 300.30 ps | 234.40 ps **-22%** | 238.20 ps **-21%** |
| `estimate_bw_from_cc_all` | 295.80 ps | 233.60 ps **-21%** | 237.20 ps **-20%** |
| `parse_max_dpm_clock` | 470.91 ns | 340.11 ns **-28%** | 338.10 ns **-28%** |
| `parse_link_speed` | 100.22 ns | 70.91 ns **-29%** | 73.38 ns **-27%** |
| `parse_ib_rate` | 91.14 ns | 75.86 ns **-17%** | 72.81 ns **-20%** |
| `parse_nvlink_output_2gpu` | 948.46 ns | 936.89 ns | 974.87 ns |
| `parse_cuda_output_8gpu` | 5.50 µs | 5.22 µs **-5%** | 11.51 µs +109% |
| `parse_vulkan_output_2gpu` | 1.86 µs | 1.78 µs **-4%** | 2.32 µs +24% |
| `parse_gaudi_output_8dev` | 2.33 µs | 2.32 µs | 2.32 µs |
| `plan_sharding 70B BF16 (4 GPU)` | 90.52 ns | 90.09 ns | 95.85 ns +6% |
| `plan_sharding 70B BF16 (13 devices)` | 48.74 ns | 49.32 ns | 47.49 ns |
| `suggest_quantization 70B (4 GPU)` | 9.42 ns | 5.41 ns **-43%** | 8.66 ns **-8%** |
| `suggest_quantization 70B (13 devices)` | 19.89 ns | 18.76 ns **-6%** | 20.91 ns +5% |
| `estimate_memory 70B FP16` | 249.10 ps | 241.10 ps **-3%** | 241.10 ps **-3%** |
| `estimate_training_memory 7B LoRA GPU` | 970.50 ps | 967.00 ps | 499.30 ps **-49%** |
| `best_available (13 devices)` | 42.21 ns | 37.36 ns **-11%** | 37.35 ns **-12%** |
| `total_memory (13 devices)` | 10.69 ns | 7.75 ns **-28%** | 10.85 ns |
| `by_family GPU (13 devices)` | 140.20 ns | 11.11 ns **-92%** | 7.89 ns **-94%** |
| `bits_per_param_all_levels` | 388.30 ps | 235.70 ps **-39%** | 265.10 ps **-32%** |
| `memory_reduction_factor_all_levels` | 354.30 ps | 235.40 ps **-34%** | 263.80 ps **-26%** |

## recommend_instance

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `7B_bf16_all` | 3.30 µs | 3.07 µs **-7%** | 3.16 µs **-4%** |
| `7B_int8_aws` | 1.18 µs | 1.10 µs **-7%** | 1.11 µs **-6%** |
| `70B_bf16_all` | 1.50 µs | 1.40 µs **-7%** | 1.36 µs **-9%** |
| `70B_int4_gcp` | 655.40 ns | 588.81 ns **-10%** | 593.71 ns **-9%** |

## cheapest_instance

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `7B_bf16` | 3.56 µs | 3.07 µs **-14%** | 3.12 µs **-12%** |
| `70B_bf16` | 1.45 µs | 1.41 µs | 1.40 µs **-3%** |

## detect_single

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `cuda` | 238.61 µs | 157.93 µs **-34%** | 160.30 µs **-33%** |
| `rocm` | 1.12 ms | 1.26 ms +13% | 1.17 ms +5% |
| `vulkan` | 161.54 µs | 155.45 µs **-4%** | 157.74 µs |
| `apple` | 168.83 µs | 161.72 µs **-4%** | 163.14 µs **-3%** |
| `tpu` | 186.44 µs | 179.27 µs **-4%** | 179.74 µs **-4%** |

## system_io

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `full_with_sysio` | 1.01 ms | 1.01 ms | 1.01 ms |
| `query_system_io` | 4.82 ns | 4.65 ns **-3%** | 3.69 ns **-23%** |
| `ingestion_1gb` | 5.25 ns | 4.71 ns **-10%** | 3.94 ns **-25%** |
| `ingestion_100gb` | 4.91 ns | 4.71 ns **-4%** | 3.73 ns **-24%** |
| `ingestion_1tb` | 4.93 ns | 4.71 ns **-4%** | 3.96 ns **-20%** |
| `serialize_registry` | 1.63 µs | 1.51 µs **-7%** | 1.61 µs |
| `deserialize_registry` | 3.46 µs | 2.04 µs **-41%** | 1.98 µs **-43%** |

## json_roundtrip

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `serialize (13 devices)` | 4.42 µs | 4.58 µs +4% | 4.13 µs **-6%** |
| `deserialize (13 devices)` | 5.96 µs | 6.20 µs +4% | 5.81 µs |

## estimate_memory

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `70B_fp32` | 716.90 ps | 495.50 ps **-31%** | 520.20 ps **-27%** |
| `70B_fp16` | 824.10 ps | 481.60 ps **-42%** | 548.50 ps **-33%** |
| `70B_bf16` | 594.80 ps | 482.40 ps **-19%** | 546.20 ps **-8%** |
| `70B_int8` | 553.30 ps | 478.80 ps **-13%** | 535.80 ps **-3%** |
| `70B_int4` | 581.10 ps | 496.70 ps **-15%** | 569.00 ps |

## suggest_quantization

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `1B_1gpu` | 4.96 ns | 4.45 ns **-10%** | 3.67 ns **-26%** |
| `7B_1gpu` | 6.81 ns | 4.75 ns **-30%** | 3.81 ns **-44%** |
| `13B_1gpu` | 8.85 ns | 4.68 ns **-47%** | 3.67 ns **-59%** |
| `70B_1gpu` | 9.41 ns | 6.36 ns **-32%** | 4.73 ns **-50%** |
| `405B_1gpu` | 9.61 ns | 4.82 ns **-50%** | 4.19 ns **-56%** |

## registry_queries

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `available_4dev` | 109.52 ns | 1.84 ns **-98%** | — |
| `available_129dev` | 574.42 ns | 39.53 ns **-93%** | — |
| `available_61dev_mixed` | 312.55 ns | 15.84 ns **-95%** | — |
| `best_available_129dev` | 546.14 ns | 372.51 ns **-32%** | 346.96 ns **-36%** |
| `total_memory_129dev` | 245.28 ns | 68.61 ns **-72%** | 66.80 ns **-73%** |
| `total_accelerator_memory_129dev` | 79.85 ns | 70.69 ns **-11%** | 74.53 ns **-7%** |
| `has_accelerator_129dev` | 1.33 ns | 1.22 ns **-8%** | 1.29 ns |
| `by_family_gpu_61dev` | 314.91 ns | 45.50 ns **-86%** | — |
| `by_family_tpu_61dev` | 125.10 ns | 45.33 ns **-64%** | — |
| `satisfying_gpu_61dev` | 345.99 ns | 73.87 ns **-79%** | — |
| `satisfying_any_accel_61dev` | 334.97 ns | 45.68 ns **-86%** | — |
| `available_count_4dev` | — | — | 1.91 ns |
| `available_count_129dev` | — | — | 41.65 ns |
| `available_count_61dev_mixed` | — | — | 16.12 ns |
| `available_collect_4dev` | — | — | 65.03 ns |
| `available_collect_129dev` | — | — | 533.56 ns |
| `available_collect_61dev_mixed` | — | — | 268.30 ns |
| `by_family_gpu_count_61dev` | — | — | 33.84 ns |
| `by_family_tpu_count_61dev` | — | — | 34.60 ns |
| `by_family_gpu_collect_61dev` | — | — | 305.55 ns |
| `by_family_tpu_collect_61dev` | — | — | 127.00 ns |
| `satisfying_gpu_count_61dev` | — | — | 137.46 ns |
| `satisfying_any_accel_count_61dev` | — | — | 105.55 ns |
| `satisfying_gpu_collect_61dev` | — | — | 392.18 ns |
| `satisfying_any_accel_collect_61dev` | — | — | 384.44 ns |

## cached_registry

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `get_cached_hit` | 37.90 ns | 28.79 ns **-24%** | 29.87 ns **-21%** |
| `invalidate` | 11.24 ns | 6.03 ns **-46%** | 5.97 ns **-47%** |

## lazy_registry

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `new` | 11.10 µs | 9.70 µs **-13%** | 10.14 µs **-9%** |
| `by_family_gpu_cold` | 1.00 ms | 1.00 ms | 1.00 ms |
| `by_family_gpu_warm` | 63.67 ns | 62.93 ns | 62.41 ns |
| `into_registry` | 3.23 ms | 2.00 ms **-38%** | 2.01 ms **-38%** |

## large_registry_sharding

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `plan_sharding_70B_128gpu` | 1.75 µs | 1.60 µs **-8%** | 1.55 µs **-11%** |
| `plan_sharding_405B_128gpu` | 1.68 µs | 1.60 µs **-5%** | 1.57 µs **-7%** |
| `suggest_quantization_70B_128gpu` | 410.07 ns | 366.38 ns **-11%** | 382.33 ns **-7%** |
| `plan_sharding_70B_mixed_61dev` | 209.82 ns | 193.31 ns **-8%** | 179.45 ns **-14%** |

## large_json

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `serialize_129dev` | 37.48 µs | 35.23 µs **-6%** | 35.76 µs **-5%** |
| `deserialize_129dev` | 45.25 µs | 42.70 µs **-6%** | 47.71 µs +5% |

## training_memory

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `7B_full_finetune_gpu` | 3.09 ns | 2.85 ns **-8%** | 3.56 ns +15% |
| `7B_lora_gpu` | 3.28 ns | 2.85 ns **-13%** | 3.65 ns +11% |
| `7B_qlora_4bit_gpu` | 3.76 ns | 3.07 ns **-19%** | 3.21 ns **-15%** |
| `7B_qlora_8bit_gpu` | 3.92 ns | 3.07 ns **-22%** | 3.39 ns **-14%** |
| `7B_prefix_gpu` | 3.41 ns | 3.05 ns **-10%** | 5.92 ns +74% |
| `7B_dpo_gpu` | 6.94 ns | 3.07 ns **-56%** | 3.81 ns **-45%** |
| `7B_rlhf_gpu` | 7.55 ns | 3.06 ns **-59%** | 3.69 ns **-51%** |
| `7B_distillation_gpu` | 4.26 ns | 2.84 ns **-33%** | 3.50 ns **-18%** |

## training_targets

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `7B_full_gpu` | 1.29 ns | 1.17 ns **-10%** | 823.70 ps **-36%** |
| `7B_full_tpu` | 1.31 ns | 1.17 ns **-10%** | 746.80 ps **-43%** |
| `7B_full_gaudi` | 1.29 ns | 1.17 ns **-9%** | 745.40 ps **-42%** |
| `7B_full_cpu` | 1.28 ns | 1.17 ns **-9%** | 747.20 ps **-42%** |

## training_model_sizes

| Benchmark | Baseline (`649834b`) | Previous (`649834b`) | Current (`6fb0ccd`) |
|-----------|------|------|------|
| `1B_lora_gpu` | 1.02 ns | 938.90 ps **-8%** | 981.10 ps **-4%** |
| `7B_lora_gpu` | 1.03 ns | 937.90 ps **-9%** | 981.50 ps **-4%** |
| `13B_lora_gpu` | 1.07 ns | 939.20 ps **-12%** | 978.40 ps **-8%** |
| `70B_lora_gpu` | 1.07 ns | 939.60 ps **-13%** | 978.00 ps **-9%** |
| `405B_lora_gpu` | 1.01 ns | 939.00 ps **-7%** | 990.90 ps |

---

Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.
