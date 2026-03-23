# Benchmarks

Latest: **2026-03-23T20:23:27Z** — commit `a281283`

Tracking: `e7525ec` (baseline) → `e7525ec` (previous) → `a281283` (current)

## ungrouped

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `load_pricing_table` | 3.16 ns | 1.59 ns **-50%** | 1.70 ns **-46%** |
| `detect_all` | 2.99 ms | 1.97 ms **-34%** | 1.00 ms **-66%** |
| `detect_none (CPU only)` | 151.07 µs | 155.83 µs +3% | 153.33 µs |
| `concurrent_detect_4_threads` | 3.82 ms | 3.56 ms **-7%** | 4.59 ms +20% |
| `parse_nvidia_bandwidth_8gpu` | 624.59 ns | 560.03 ns **-10%** | 604.63 ns **-3%** |
| `nvidia_bus_width_all_ccs` | 23.20 ns | 20.46 ns **-12%** | 19.99 ns **-14%** |
| `estimate_bw_from_cc_all` | 28.80 ns | 41.33 ns +43% | 22.85 ns **-21%** |
| `parse_max_dpm_clock` | 145.17 ns | 125.73 ns **-13%** | 138.52 ns **-5%** |
| `parse_link_speed` | 51.74 ns | 46.50 ns **-10%** | 42.59 ns **-18%** |
| `parse_ib_rate` | 56.70 ns | 50.41 ns **-11%** | 44.47 ns **-22%** |
| `parse_nvlink_output_2gpu` | 629.96 ns | 564.84 ns **-10%** | 528.66 ns **-16%** |
| `parse_cuda_output_8gpu` | 4.02 µs | 3.61 µs **-10%** | 3.22 µs **-20%** |
| `parse_vulkan_output_2gpu` | 1.10 µs | 1.57 µs +43% | 923.96 ns **-16%** |
| `parse_gaudi_output_8dev` | 1.71 µs | 2.00 µs +17% | 1.40 µs **-18%** |
| `plan_sharding 70B BF16 (4 GPU)` | 95.21 ns | 87.52 ns **-8%** | 79.87 ns **-16%** |
| `plan_sharding 70B BF16 (13 devices)` | 48.80 ns | 36.35 ns **-26%** | 34.94 ns **-28%** |
| `suggest_quantization 70B (4 GPU)` | 11.84 ns | 9.13 ns **-23%** | 10.81 ns **-9%** |
| `suggest_quantization 70B (13 devices)` | 28.18 ns | 25.78 ns **-9%** | 22.39 ns **-21%** |
| `estimate_memory 70B FP16` | 280.70 ps | 254.50 ps **-9%** | 238.50 ps **-15%** |
| `estimate_training_memory 7B LoRA GPU` | 3.92 ns | 3.16 ns **-20%** | 2.41 ns **-38%** |
| `best_available (13 devices)` | 29.77 ns | 28.19 ns **-5%** | 25.32 ns **-15%** |
| `total_memory (13 devices)` | 7.19 ns | 5.75 ns **-20%** | 5.20 ns **-28%** |
| `by_family GPU (13 devices)` | 63.12 ns | 78.52 ns +24% | 55.89 ns **-11%** |
| `bits_per_param_all_levels` | 34.60 ps | 31.10 ps **-10%** | 29.00 ps **-16%** |
| `memory_reduction_factor_all_levels` | 37.40 ps | 30.70 ps **-18%** | 29.00 ps **-22%** |

## recommend_instance

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `7B_bf16_all` | 5.13 µs | 2.46 µs **-52%** | 2.87 µs **-44%** |
| `7B_int8_aws` | 1.89 µs | 1.78 µs **-6%** | 1.09 µs **-42%** |
| `70B_bf16_all` | 1.97 µs | 1.32 µs **-33%** | 1.39 µs **-29%** |
| `70B_int4_gcp` | 562.28 ns | 505.40 ns **-10%** | 603.34 ns +7% |

## cheapest_instance

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `7B_bf16` | 2.80 µs | 2.60 µs **-7%** | 3.28 µs +17% |
| `70B_bf16` | 1.41 µs | 1.22 µs **-13%** | 1.30 µs **-8%** |

## detect_single

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `cuda` | 175.96 µs | 167.43 µs **-5%** | 173.45 µs |
| `rocm` | 974.35 µs | 1.03 ms +6% | 950.90 µs |
| `vulkan` | 171.40 µs | 213.85 µs +25% | 174.74 µs |
| `apple` | 172.00 µs | 175.61 µs | 186.02 µs +8% |
| `tpu` | 194.98 µs | 302.37 µs +55% | 213.62 µs +10% |

## system_io

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `full_with_sysio` | 1.01 ms | 3.41 ms +238% | 1.00 ms |
| `query_system_io` | 4.19 ns | 5.17 ns +23% | 4.72 ns +13% |
| `serialize_registry` | 889.80 ns | 913.36 ns | 995.44 ns +12% |
| `deserialize_registry` | 1.40 µs | 1.39 µs | 1.49 µs +7% |
| `ingestion_1gb` | — | 2.93 ns | 3.60 ns |
| `ingestion_100gb` | — | 3.01 ns | 3.75 ns |
| `ingestion_1tb` | — | 3.51 ns | 3.80 ns |

## json_roundtrip

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `serialize (13 devices)` | 2.65 µs | 2.26 µs **-15%** | 2.03 µs **-23%** |
| `deserialize (13 devices)` | 3.94 µs | 3.27 µs **-17%** | 3.10 µs **-21%** |

## estimate_memory

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `70B_fp32` | 274.90 ps | 247.40 ps **-10%** | 236.40 ps **-14%** |
| `70B_fp16` | 290.30 ps | 247.80 ps **-15%** | 237.60 ps **-18%** |
| `70B_bf16` | 280.80 ps | 250.70 ps **-11%** | 237.50 ps **-15%** |
| `70B_int8` | 278.20 ps | 246.20 ps **-12%** | 239.10 ps **-14%** |
| `70B_int4` | 272.10 ps | 249.80 ps **-8%** | 236.90 ps **-13%** |

## suggest_quantization

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `1B_1gpu` | 6.76 ns | 6.98 ns +3% | 5.46 ns **-19%** |
| `7B_1gpu` | 6.92 ns | 6.21 ns **-10%** | 5.50 ns **-20%** |
| `13B_1gpu` | 6.51 ns | 6.34 ns | 5.71 ns **-12%** |
| `70B_1gpu` | 7.12 ns | 6.70 ns **-6%** | 5.72 ns **-20%** |
| `405B_1gpu` | 7.50 ns | 6.78 ns **-10%** | 5.80 ns **-23%** |

## registry_queries

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `available_4dev` | 65.43 ns | 63.14 ns **-3%** | 53.23 ns **-19%** |
| `available_129dev` | 425.04 ns | 359.73 ns **-15%** | 336.90 ns **-21%** |
| `available_61dev_mixed` | 206.89 ns | 204.93 ns | 175.08 ns **-15%** |
| `best_available_129dev` | 142.34 ns | 177.29 ns +25% | 134.86 ns **-5%** |
| `total_memory_129dev` | 42.41 ns | 45.99 ns +8% | 41.66 ns |
| `total_accelerator_memory_129dev` | 59.78 ns | 62.69 ns +5% | 49.24 ns **-18%** |
| `has_accelerator_129dev` | 1.36 ns | 1.45 ns +7% | 1.24 ns **-9%** |
| `by_family_gpu_61dev` | 229.89 ns | 228.58 ns | 209.69 ns **-9%** |
| `by_family_tpu_61dev` | 100.76 ns | 117.88 ns +17% | 91.87 ns **-9%** |
| `satisfying_gpu_61dev` | 242.32 ns | 260.35 ns +7% | 214.81 ns **-11%** |
| `satisfying_any_accel_61dev` | 232.04 ns | 265.42 ns +14% | 213.51 ns **-8%** |

## cached_registry

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `get_cached_hit` | 29.42 ns | 30.44 ns +3% | 28.92 ns |
| `invalidate` | 3.63 ns | 4.34 ns +20% | 3.67 ns |

## lazy_registry

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `new` | 10.55 µs | 12.03 µs +14% | 10.21 µs **-3%** |
| `by_family_gpu_cold` | 1.00 ms | 1.09 ms +9% | 1.00 ms |
| `by_family_gpu_warm` | 47.69 ns | 55.36 ns +16% | 44.56 ns **-7%** |
| `into_registry` | 2.16 ms | 3.69 ms +71% | 2.01 ms **-7%** |

## large_registry_sharding

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `plan_sharding_70B_128gpu` | 1.28 µs | 1.56 µs +21% | 1.27 µs |
| `plan_sharding_405B_128gpu` | 1.31 µs | 1.56 µs +18% | 1.26 µs **-4%** |
| `suggest_quantization_70B_128gpu` | 228.79 ns | 224.93 ns | 214.00 ns **-6%** |
| `plan_sharding_70B_mixed_61dev` | 144.55 ns | 166.49 ns +15% | 174.58 ns +21% |

## large_json

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `serialize_129dev` | 15.79 µs | 18.49 µs +17% | 14.46 µs **-8%** |
| `deserialize_129dev` | 25.86 µs | 30.98 µs +20% | 23.40 µs **-10%** |

## training_memory

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `7B_full_finetune_gpu` | 2.71 ns | 3.79 ns +40% | 3.15 ns +16% |
| `7B_lora_gpu` | 2.73 ns | 4.11 ns +51% | 2.89 ns +6% |
| `7B_qlora_4bit_gpu` | 2.82 ns | 4.18 ns +48% | 2.91 ns +3% |
| `7B_qlora_8bit_gpu` | 2.83 ns | 4.38 ns +55% | 2.88 ns |
| `7B_prefix_gpu` | 3.23 ns | 4.25 ns +32% | 3.11 ns **-4%** |
| `7B_dpo_gpu` | 2.77 ns | 4.17 ns +51% | 2.88 ns +4% |
| `7B_rlhf_gpu` | 2.78 ns | 4.53 ns +63% | 2.87 ns +3% |
| `7B_distillation_gpu` | 2.84 ns | 3.76 ns +32% | 2.64 ns **-7%** |

## training_targets

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `7B_full_gpu` | 2.80 ns | 3.72 ns +33% | 3.11 ns +11% |
| `7B_full_tpu` | 3.64 ns | 3.22 ns **-11%** | 2.51 ns **-31%** |
| `7B_full_gaudi` | 2.87 ns | 3.70 ns +29% | 2.49 ns **-13%** |
| `7B_full_cpu` | 2.76 ns | 3.43 ns +24% | 3.10 ns +12% |

## training_model_sizes

| Benchmark | Baseline (`e7525ec`) | Previous (`e7525ec`) | Current (`a281283`) |
|-----------|------|------|------|
| `1B_lora_gpu` | 2.81 ns | 3.82 ns +36% | 3.10 ns +10% |
| `7B_lora_gpu` | 2.57 ns | 3.37 ns +31% | 3.11 ns +21% |
| `13B_lora_gpu` | 2.79 ns | 3.43 ns +23% | 3.12 ns +12% |
| `70B_lora_gpu` | 2.70 ns | 3.45 ns +28% | 3.10 ns +15% |
| `405B_lora_gpu` | 2.88 ns | 3.67 ns +28% | 3.10 ns +7% |

---

Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.
