# Benchmarks

Latest: **2026-04-03T13:44:48Z** — commit `649834b`

Tracking: `ebbb9e1` (baseline) → `649834b` (previous) → `649834b` (current)

## ungrouped

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `load_pricing_table` | 2.28 ns | 1.24 ns **-46%** | 1.21 ns **-47%** |
| `detect_all` | 1.12 ms | 1.01 ms **-10%** | 998.24 µs **-11%** |
| `detect_none (CPU only)` | 151.26 µs | 142.68 µs **-6%** | 140.31 µs **-7%** |
| `concurrent_detect_4_threads` | 5.53 ms | 4.41 ms **-20%** | 2.50 ms **-55%** |
| `parse_nvidia_bandwidth_8gpu` | 1.08 µs | 1.15 µs +6% | 992.03 ns **-8%** |
| `nvidia_bus_width_all_ccs` | 35.12 ns | 300.30 ps **-99%** | 234.40 ps **-99%** |
| `estimate_bw_from_cc_all` | 45.28 ns | 295.80 ps **-99%** | 233.60 ps **-99%** |
| `parse_max_dpm_clock` | 225.93 ns | 470.91 ns +108% | 340.11 ns +51% |
| `parse_link_speed` | 71.56 ns | 100.22 ns +40% | 70.91 ns |
| `parse_ib_rate` | 101.43 ns | 91.14 ns **-10%** | 75.86 ns **-25%** |
| `parse_nvlink_output_2gpu` | 854.78 ns | 948.46 ns +11% | 936.89 ns +10% |
| `parse_cuda_output_8gpu` | 4.70 µs | 5.50 µs +17% | 5.22 µs +11% |
| `parse_vulkan_output_2gpu` | 1.38 µs | 1.86 µs +35% | 1.78 µs +29% |
| `parse_gaudi_output_8dev` | 2.68 µs | 2.33 µs **-13%** | 2.32 µs **-13%** |
| `plan_sharding 70B BF16 (4 GPU)` | 154.73 ns | 90.52 ns **-41%** | 90.09 ns **-42%** |
| `plan_sharding 70B BF16 (13 devices)` | 71.53 ns | 48.74 ns **-32%** | 49.32 ns **-31%** |
| `suggest_quantization 70B (4 GPU)` | 22.05 ns | 9.42 ns **-57%** | 5.41 ns **-75%** |
| `suggest_quantization 70B (13 devices)` | 40.24 ns | 19.89 ns **-51%** | 18.76 ns **-53%** |
| `estimate_memory 70B FP16` | 398.30 ps | 249.10 ps **-37%** | 241.10 ps **-39%** |
| `estimate_training_memory 7B LoRA GPU` | 7.84 ns | 970.50 ps **-88%** | 967.00 ps **-88%** |
| `best_available (13 devices)` | 31.57 ns | 42.21 ns +34% | 37.36 ns +18% |
| `total_memory (13 devices)` | 15.88 ns | 10.69 ns **-33%** | 7.75 ns **-51%** |
| `by_family GPU (13 devices)` | 129.95 ns | 140.20 ns +8% | 11.11 ns **-91%** |
| `bits_per_param_all_levels` | 75.50 ps | 388.30 ps +414% | 235.70 ps +212% |
| `memory_reduction_factor_all_levels` | 79.10 ps | 354.30 ps +348% | 235.40 ps +198% |

## recommend_instance

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `7B_bf16_all` | 3.08 µs | 3.30 µs +7% | 3.07 µs |
| `7B_int8_aws` | 1.35 µs | 1.18 µs **-12%** | 1.10 µs **-18%** |
| `70B_bf16_all` | 1.32 µs | 1.50 µs +13% | 1.40 µs +6% |
| `70B_int4_gcp` | 545.41 ns | 655.40 ns +20% | 588.81 ns +8% |

## cheapest_instance

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `7B_bf16` | 3.08 µs | 3.56 µs +16% | 3.07 µs |
| `70B_bf16` | 1.53 µs | 1.45 µs **-5%** | 1.41 µs **-8%** |

## detect_single

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `cuda` | 169.58 µs | 238.61 µs +41% | 157.93 µs **-7%** |
| `rocm` | 995.56 µs | 1.12 ms +12% | 1.26 ms +27% |
| `vulkan` | 230.58 µs | 161.54 µs **-30%** | 155.45 µs **-33%** |
| `apple` | 219.91 µs | 168.83 µs **-23%** | 161.72 µs **-26%** |
| `tpu` | 202.33 µs | 186.44 µs **-8%** | 179.27 µs **-11%** |

## system_io

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `full_with_sysio` | 1.06 ms | 1.01 ms **-5%** | 1.01 ms **-5%** |
| `query_system_io` | 34.60 ps | 4.82 ns +13821% | 4.65 ns +13338% |
| `ingestion_1gb` | 2.83 ns | 5.25 ns +86% | 4.71 ns +67% |
| `ingestion_100gb` | 2.58 ns | 4.91 ns +90% | 4.71 ns +82% |
| `ingestion_1tb` | 2.86 ns | 4.93 ns +72% | 4.71 ns +65% |
| `serialize_registry` | 1.25 µs | 1.63 µs +30% | 1.51 µs +20% |
| `deserialize_registry` | 2.01 µs | 3.46 µs +73% | 2.04 µs |

## json_roundtrip

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `serialize (13 devices)` | 4.50 µs | 4.42 µs | 4.58 µs |
| `deserialize (13 devices)` | 5.80 µs | 5.96 µs | 6.20 µs +7% |

## estimate_memory

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `70B_fp32` | 357.70 ps | 716.90 ps +100% | 495.50 ps +39% |
| `70B_fp16` | 378.20 ps | 824.10 ps +118% | 481.60 ps +27% |
| `70B_bf16` | 404.40 ps | 594.80 ps +47% | 482.40 ps +19% |
| `70B_int8` | 380.20 ps | 553.30 ps +46% | 478.80 ps +26% |
| `70B_int4` | 361.10 ps | 581.10 ps +61% | 496.70 ps +38% |

## suggest_quantization

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `1B_1gpu` | 11.96 ns | 4.96 ns **-59%** | 4.45 ns **-63%** |
| `7B_1gpu` | 12.26 ns | 6.81 ns **-44%** | 4.75 ns **-61%** |
| `13B_1gpu` | 12.11 ns | 8.85 ns **-27%** | 4.68 ns **-61%** |
| `70B_1gpu` | 13.13 ns | 9.41 ns **-28%** | 6.36 ns **-52%** |
| `405B_1gpu` | 13.16 ns | 9.61 ns **-27%** | 4.82 ns **-63%** |

## registry_queries

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `available_4dev` | 123.51 ns | 109.52 ns **-11%** | 1.84 ns **-99%** |
| `available_129dev` | 994.93 ns | 574.42 ns **-42%** | 39.53 ns **-96%** |
| `available_61dev_mixed` | 456.05 ns | 312.55 ns **-31%** | 15.84 ns **-97%** |
| `best_available_129dev` | 371.62 ns | 546.14 ns +47% | 372.51 ns |
| `total_memory_129dev` | 95.15 ns | 245.28 ns +158% | 68.61 ns **-28%** |
| `total_accelerator_memory_129dev` | 134.23 ns | 79.85 ns **-41%** | 70.69 ns **-47%** |
| `has_accelerator_129dev` | 2.81 ns | 1.33 ns **-53%** | 1.22 ns **-56%** |
| `by_family_gpu_61dev` | 500.39 ns | 314.91 ns **-37%** | 45.50 ns **-91%** |
| `by_family_tpu_61dev` | 197.54 ns | 125.10 ns **-37%** | 45.33 ns **-77%** |
| `satisfying_gpu_61dev` | 545.64 ns | 345.99 ns **-37%** | 73.87 ns **-86%** |
| `satisfying_any_accel_61dev` | 570.78 ns | 334.97 ns **-41%** | 45.68 ns **-92%** |

## cached_registry

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `get_cached_hit` | 56.30 ns | 37.90 ns **-33%** | 28.79 ns **-49%** |
| `invalidate` | 8.57 ns | 11.24 ns +31% | 6.03 ns **-30%** |

## lazy_registry

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `new` | 18.66 µs | 11.10 µs **-41%** | 9.70 µs **-48%** |
| `by_family_gpu_cold` | 1.94 ms | 1.00 ms **-48%** | 1.00 ms **-48%** |
| `by_family_gpu_warm` | 105.38 ns | 63.67 ns **-40%** | 62.93 ns **-40%** |
| `into_registry` | 6.44 ms | 3.23 ms **-50%** | 2.00 ms **-69%** |

## large_registry_sharding

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `plan_sharding_70B_128gpu` | 2.04 µs | 1.75 µs **-14%** | 1.60 µs **-21%** |
| `plan_sharding_405B_128gpu` | 1.21 µs | 1.68 µs +38% | 1.60 µs +32% |
| `suggest_quantization_70B_128gpu` | 178.75 ns | 410.07 ns +129% | 366.38 ns +105% |
| `plan_sharding_70B_mixed_61dev` | 165.47 ns | 209.82 ns +27% | 193.31 ns +17% |

## large_json

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `serialize_129dev` | 24.44 µs | 37.48 µs +53% | 35.23 µs +44% |
| `deserialize_129dev` | 49.37 µs | 45.25 µs **-8%** | 42.70 µs **-13%** |

## training_memory

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `7B_full_finetune_gpu` | 5.41 ns | 3.09 ns **-43%** | 2.85 ns **-47%** |
| `7B_lora_gpu` | 4.86 ns | 3.28 ns **-33%** | 2.85 ns **-41%** |
| `7B_qlora_4bit_gpu` | 6.64 ns | 3.76 ns **-43%** | 3.07 ns **-54%** |
| `7B_qlora_8bit_gpu` | 5.50 ns | 3.92 ns **-29%** | 3.07 ns **-44%** |
| `7B_prefix_gpu` | 6.27 ns | 3.41 ns **-46%** | 3.05 ns **-51%** |
| `7B_dpo_gpu` | 5.24 ns | 6.94 ns +32% | 3.07 ns **-41%** |
| `7B_rlhf_gpu` | 7.27 ns | 7.55 ns +4% | 3.06 ns **-58%** |
| `7B_distillation_gpu` | 7.02 ns | 4.26 ns **-39%** | 2.84 ns **-60%** |

## training_targets

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `7B_full_gpu` | 6.26 ns | 1.29 ns **-79%** | 1.17 ns **-81%** |
| `7B_full_tpu` | 4.92 ns | 1.31 ns **-73%** | 1.17 ns **-76%** |
| `7B_full_gaudi` | 4.35 ns | 1.29 ns **-70%** | 1.17 ns **-73%** |
| `7B_full_cpu` | 3.13 ns | 1.28 ns **-59%** | 1.17 ns **-63%** |

## training_model_sizes

| Benchmark | Baseline (`ebbb9e1`) | Previous (`649834b`) | Current (`649834b`) |
|-----------|------|------|------|
| `1B_lora_gpu` | 2.80 ns | 1.02 ns **-64%** | 938.90 ps **-66%** |
| `7B_lora_gpu` | 2.80 ns | 1.03 ns **-63%** | 937.90 ps **-67%** |
| `13B_lora_gpu` | 2.76 ns | 1.07 ns **-61%** | 939.20 ps **-66%** |
| `70B_lora_gpu` | 2.98 ns | 1.07 ns **-64%** | 939.60 ps **-68%** |
| `405B_lora_gpu` | 2.74 ns | 1.01 ns **-63%** | 939.00 ps **-66%** |

---

Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.
