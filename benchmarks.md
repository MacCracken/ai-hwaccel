# Benchmarks

Latest: **2026-03-23T19:46:06Z** — commit `e7525ec`

Tracking: `96cb1cd` (baseline) → `e7525ec` (previous) → `e7525ec` (current)

## ungrouped

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `load_pricing_table` | 2.00 ns | 3.16 ns +58% | 1.59 ns **-20%** |
| `detect_all` | 1.02 ms | 2.99 ms +192% | 1.97 ms +93% |
| `detect_none (CPU only)` | 140.69 µs | 151.07 µs +7% | 155.83 µs +11% |
| `concurrent_detect_4_threads` | 2.96 ms | 3.82 ms +29% | 3.56 ms +20% |
| `parse_nvidia_bandwidth_8gpu` | 514.71 ns | 624.59 ns +21% | 560.03 ns +9% |
| `nvidia_bus_width_all_ccs` | 18.50 ns | 23.20 ns +25% | 20.46 ns +11% |
| `estimate_bw_from_cc_all` | 18.94 ns | 28.80 ns +52% | 41.33 ns +118% |
| `parse_max_dpm_clock` | 115.67 ns | 145.17 ns +26% | 125.73 ns +9% |
| `parse_link_speed` | 46.68 ns | 51.74 ns +11% | 46.50 ns |
| `parse_ib_rate` | 45.87 ns | 56.70 ns +24% | 50.41 ns +10% |
| `parse_nvlink_output_2gpu` | 539.41 ns | 629.96 ns +17% | 564.84 ns +5% |
| `plan_sharding 70B BF16 (4 GPU)` | 76.29 ns | 95.21 ns +25% | 87.52 ns +15% |
| `plan_sharding 70B BF16 (13 devices)` | 36.48 ns | 48.80 ns +34% | 36.35 ns |
| `suggest_quantization 70B (4 GPU)` | 10.60 ns | 11.84 ns +12% | 9.13 ns **-14%** |
| `suggest_quantization 70B (13 devices)` | 24.70 ns | 28.18 ns +14% | 25.78 ns +4% |
| `estimate_memory 70B FP16` | 246.60 ps | 280.70 ps +14% | 254.50 ps +3% |
| `estimate_training_memory 7B LoRA GPU` | 3.20 ns | 3.92 ns +23% | 3.16 ns |
| `best_available (13 devices)` | 23.07 ns | 29.77 ns +29% | 28.19 ns +22% |
| `total_memory (13 devices)` | 5.88 ns | 7.19 ns +22% | 5.75 ns |
| `by_family GPU (13 devices)` | 73.36 ns | 63.12 ns **-14%** | 78.52 ns +7% |
| `bits_per_param_all_levels` | 33.30 ps | 34.60 ps +4% | 31.10 ps **-7%** |
| `memory_reduction_factor_all_levels` | 32.70 ps | 37.40 ps +14% | 30.70 ps **-6%** |
| `parse_cuda_output_8gpu` | — | 4.02 µs | 3.61 µs |
| `parse_vulkan_output_2gpu` | — | 1.10 µs | 1.57 µs |
| `parse_gaudi_output_8dev` | — | 1.71 µs | 2.00 µs |

## recommend_instance

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `7B_bf16_all` | 2.62 µs | 5.13 µs +96% | 2.46 µs **-6%** |
| `7B_int8_aws` | 950.24 ns | 1.89 µs +99% | 1.78 µs +87% |
| `70B_bf16_all` | 1.23 µs | 1.97 µs +60% | 1.32 µs +7% |
| `70B_int4_gcp` | 558.52 ns | 562.28 ns | 505.40 ns **-10%** |

## cheapest_instance

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `7B_bf16` | 2.56 µs | 2.80 µs +10% | 2.60 µs |
| `70B_bf16` | 1.25 µs | 1.41 µs +13% | 1.22 µs |

## detect_single

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `cuda` | 158.56 µs | 175.96 µs +11% | 167.43 µs +6% |
| `rocm` | 1.09 ms | 974.35 µs **-10%** | 1.03 ms **-5%** |
| `vulkan` | 165.24 µs | 171.40 µs +4% | 213.85 µs +29% |
| `apple` | 166.93 µs | 172.00 µs +3% | 175.61 µs +5% |
| `tpu` | 181.44 µs | 194.98 µs +7% | 302.37 µs +67% |

## system_io

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `full_with_sysio` | 1.01 ms | 1.01 ms | 3.41 ms +237% |
| `query_system_io` | 4.32 ns | 4.19 ns **-3%** | 5.17 ns +20% |
| `serialize_registry` | 909.49 ns | 889.80 ns | 913.36 ns |
| `deserialize_registry` | 1.26 µs | 1.40 µs +11% | 1.39 µs +10% |
| `ingestion_1gb` | — | — | 2.93 ns |
| `ingestion_100gb` | — | — | 3.01 ns |
| `ingestion_1tb` | — | — | 3.51 ns |

## json_roundtrip

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `serialize (13 devices)` | 2.25 µs | 2.65 µs +18% | 2.26 µs |
| `deserialize (13 devices)` | 3.65 µs | 3.94 µs +8% | 3.27 µs **-10%** |

## estimate_memory

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `70B_fp32` | 256.40 ps | 274.90 ps +7% | 247.40 ps **-4%** |
| `70B_fp16` | 257.70 ps | 290.30 ps +13% | 247.80 ps **-4%** |
| `70B_bf16` | 259.50 ps | 280.80 ps +8% | 250.70 ps **-3%** |
| `70B_int8` | 260.70 ps | 278.20 ps +7% | 246.20 ps **-6%** |
| `70B_int4` | 263.80 ps | 272.10 ps +3% | 249.80 ps **-5%** |

## suggest_quantization

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `1B_1gpu` | 6.08 ns | 6.76 ns +11% | 6.98 ns +15% |
| `7B_1gpu` | 5.49 ns | 6.92 ns +26% | 6.21 ns +13% |
| `13B_1gpu` | 5.71 ns | 6.51 ns +14% | 6.34 ns +11% |
| `70B_1gpu` | 6.66 ns | 7.12 ns +7% | 6.70 ns |
| `405B_1gpu` | 6.76 ns | 7.50 ns +11% | 6.78 ns |

## registry_queries

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `available_4dev` | 54.57 ns | 65.43 ns +20% | 63.14 ns +16% |
| `available_129dev` | 344.79 ns | 425.04 ns +23% | 359.73 ns +4% |
| `available_61dev_mixed` | 171.33 ns | 206.89 ns +21% | 204.93 ns +20% |
| `best_available_129dev` | 131.63 ns | 142.34 ns +8% | 177.29 ns +35% |
| `total_memory_129dev` | 41.13 ns | 42.41 ns +3% | 45.99 ns +12% |
| `total_accelerator_memory_129dev` | 51.56 ns | 59.78 ns +16% | 62.69 ns +22% |
| `has_accelerator_129dev` | 1.27 ns | 1.36 ns +7% | 1.45 ns +14% |
| `by_family_gpu_61dev` | 188.97 ns | 229.89 ns +22% | 228.58 ns +21% |
| `by_family_tpu_61dev` | 90.33 ns | 100.76 ns +12% | 117.88 ns +31% |
| `satisfying_gpu_61dev` | 231.04 ns | 242.32 ns +5% | 260.35 ns +13% |
| `satisfying_any_accel_61dev` | 219.77 ns | 232.04 ns +6% | 265.42 ns +21% |

## cached_registry

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `get_cached_hit` | 28.97 ns | 29.42 ns | 30.44 ns +5% |
| `invalidate` | 3.64 ns | 3.63 ns | 4.34 ns +19% |

## lazy_registry

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `new` | 10.71 µs | 10.55 µs | 12.03 µs +12% |
| `by_family_gpu_cold` | 1.03 ms | 1.00 ms | 1.09 ms +6% |
| `by_family_gpu_warm` | 48.41 ns | 47.69 ns | 55.36 ns +14% |
| `into_registry` | 2.15 ms | 2.16 ms | 3.69 ms +72% |

## large_registry_sharding

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `plan_sharding_70B_128gpu` | 1.35 µs | 1.28 µs **-5%** | 1.56 µs +16% |
| `plan_sharding_405B_128gpu` | 1.31 µs | 1.31 µs | 1.56 µs +19% |
| `suggest_quantization_70B_128gpu` | 228.78 ns | 228.79 ns | 224.93 ns |
| `plan_sharding_70B_mixed_61dev` | 151.75 ns | 144.55 ns **-5%** | 166.49 ns +10% |

## large_json

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `serialize_129dev` | 14.87 µs | 15.79 µs +6% | 18.49 µs +24% |
| `deserialize_129dev` | 26.12 µs | 25.86 µs | 30.98 µs +19% |

## training_memory

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `7B_full_finetune_gpu` | 3.50 ns | 2.71 ns **-22%** | 3.79 ns +8% |
| `7B_lora_gpu` | 3.03 ns | 2.73 ns **-10%** | 4.11 ns +36% |
| `7B_qlora_4bit_gpu` | 3.17 ns | 2.82 ns **-11%** | 4.18 ns +32% |
| `7B_qlora_8bit_gpu` | 3.16 ns | 2.83 ns **-11%** | 4.38 ns +39% |
| `7B_prefix_gpu` | 3.47 ns | 3.23 ns **-7%** | 4.25 ns +23% |
| `7B_dpo_gpu` | 3.19 ns | 2.77 ns **-13%** | 4.17 ns +31% |
| `7B_rlhf_gpu` | 3.25 ns | 2.78 ns **-14%** | 4.53 ns +39% |
| `7B_distillation_gpu` | 2.77 ns | 2.84 ns | 3.76 ns +36% |

## training_targets

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `7B_full_gpu` | 3.41 ns | 2.80 ns **-18%** | 3.72 ns +9% |
| `7B_full_tpu` | 2.66 ns | 3.64 ns +37% | 3.22 ns +21% |
| `7B_full_gaudi` | 2.65 ns | 2.87 ns +9% | 3.70 ns +40% |
| `7B_full_cpu` | 3.45 ns | 2.76 ns **-20%** | 3.43 ns |

## training_model_sizes

| Benchmark | Baseline (`96cb1cd`) | Previous (`e7525ec`) | Current (`e7525ec`) |
|-----------|------|------|------|
| `1B_lora_gpu` | 3.47 ns | 2.81 ns **-19%** | 3.82 ns +10% |
| `7B_lora_gpu` | 3.66 ns | 2.57 ns **-30%** | 3.37 ns **-8%** |
| `13B_lora_gpu` | 3.66 ns | 2.79 ns **-24%** | 3.43 ns **-6%** |
| `70B_lora_gpu` | 3.92 ns | 2.70 ns **-31%** | 3.45 ns **-12%** |
| `405B_lora_gpu` | 3.63 ns | 2.88 ns **-21%** | 3.67 ns |

---

Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.
