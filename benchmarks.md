# Benchmarks

Latest: **2026-03-27T16:31:06Z** — commit `ebbb9e1`

Tracking: `19a2753` (baseline) → `19a2753` (previous) → `ebbb9e1` (current)

## ungrouped

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `load_pricing_table` | 2.01 ns | 1.92 ns **-5%** | 2.28 ns +14% |
| `detect_all` | 2.97 ms | 1.09 ms **-63%** | 1.12 ms **-62%** |
| `detect_none (CPU only)` | 156.28 µs | 160.72 µs | 151.26 µs **-3%** |
| `concurrent_detect_4_threads` | 3.86 ms | 5.46 ms +42% | 5.53 ms +43% |
| `parse_nvidia_bandwidth_8gpu` | 619.44 ns | 675.51 ns +9% | 1.08 µs +74% |
| `nvidia_bus_width_all_ccs` | 21.58 ns | 23.55 ns +9% | 35.12 ns +63% |
| `estimate_bw_from_cc_all` | 26.90 ns | 24.40 ns **-9%** | 45.28 ns +68% |
| `parse_max_dpm_clock` | 129.82 ns | 146.63 ns +13% | 225.93 ns +74% |
| `parse_link_speed` | 43.42 ns | 57.03 ns +31% | 71.56 ns +65% |
| `parse_ib_rate` | 50.80 ns | 89.87 ns +77% | 101.43 ns +100% |
| `parse_nvlink_output_2gpu` | 620.21 ns | 1.67 µs +170% | 854.78 ns +38% |
| `parse_cuda_output_8gpu` | 3.80 µs | 7.83 µs +106% | 4.70 µs +24% |
| `parse_vulkan_output_2gpu` | 1.07 µs | 2.01 µs +88% | 1.38 µs +30% |
| `parse_gaudi_output_8dev` | 1.58 µs | 2.89 µs +83% | 2.68 µs +70% |
| `plan_sharding 70B BF16 (4 GPU)` | 67.84 ns | 75.03 ns +11% | 154.73 ns +128% |
| `plan_sharding 70B BF16 (13 devices)` | 32.52 ns | 36.91 ns +14% | 71.53 ns +120% |
| `suggest_quantization 70B (4 GPU)` | 10.81 ns | 11.03 ns | 22.05 ns +104% |
| `suggest_quantization 70B (13 devices)` | 25.56 ns | 23.80 ns **-7%** | 40.24 ns +57% |
| `estimate_memory 70B FP16` | 263.80 ps | 260.80 ps | 398.30 ps +51% |
| `estimate_training_memory 7B LoRA GPU` | 3.23 ns | 2.79 ns **-14%** | 7.84 ns +143% |
| `best_available (13 devices)` | 22.54 ns | 25.58 ns +13% | 31.57 ns +40% |
| `total_memory (13 devices)` | 6.45 ns | 5.83 ns **-10%** | 15.88 ns +146% |
| `by_family GPU (13 devices)` | 59.60 ns | 60.00 ns | 129.95 ns +118% |
| `bits_per_param_all_levels` | 33.70 ps | 38.10 ps +13% | 75.50 ps +124% |
| `memory_reduction_factor_all_levels` | 34.20 ps | 33.70 ps | 79.10 ps +131% |

## recommend_instance

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `7B_bf16_all` | 2.81 µs | 2.69 µs **-5%** | 3.08 µs +9% |
| `7B_int8_aws` | 1.03 µs | 972.12 ns **-5%** | 1.35 µs +31% |
| `70B_bf16_all` | 1.23 µs | 1.28 µs +4% | 1.32 µs +7% |
| `70B_int4_gcp` | 546.40 ns | 502.66 ns **-8%** | 545.41 ns |

## cheapest_instance

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `7B_bf16` | 2.92 µs | 2.69 µs **-8%** | 3.08 µs +5% |
| `70B_bf16` | 1.25 µs | 1.25 µs | 1.53 µs +23% |

## detect_single

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `cuda` | 187.68 µs | 251.48 µs +34% | 169.58 µs **-10%** |
| `rocm` | 994.93 µs | 1.01 ms | 995.56 µs |
| `vulkan` | 180.33 µs | 265.77 µs +47% | 230.58 µs +28% |
| `apple` | 174.09 µs | 183.32 µs +5% | 219.91 µs +26% |
| `tpu` | 210.77 µs | 197.89 µs **-6%** | 202.33 µs **-4%** |

## system_io

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `full_with_sysio` | 1.00 ms | 1.33 ms +33% | 1.06 ms +6% |
| `query_system_io` | 37.50 ps | 71.20 ps +90% | 34.60 ps **-8%** |
| `ingestion_1gb` | 2.32 ns | 4.10 ns +77% | 2.83 ns +22% |
| `ingestion_100gb` | 2.26 ns | 5.01 ns +121% | 2.58 ns +14% |
| `ingestion_1tb` | 2.33 ns | 3.06 ns +31% | 2.86 ns +23% |
| `serialize_registry` | 984.66 ns | 1.02 µs +4% | 1.25 µs +27% |
| `deserialize_registry` | 1.39 µs | 2.36 µs +70% | 2.01 µs +45% |

## json_roundtrip

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `serialize (13 devices)` | 2.22 µs | 2.21 µs | 4.50 µs +103% |
| `deserialize (13 devices)` | 3.54 µs | 3.54 µs | 5.80 µs +64% |

## estimate_memory

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `70B_fp32` | 279.90 ps | 260.50 ps **-7%** | 357.70 ps +28% |
| `70B_fp16` | 275.30 ps | 266.10 ps **-3%** | 378.20 ps +37% |
| `70B_bf16` | 272.30 ps | 263.10 ps **-3%** | 404.40 ps +49% |
| `70B_int8` | 287.80 ps | 262.00 ps **-9%** | 380.20 ps +32% |
| `70B_int4` | 262.50 ps | 261.00 ps | 361.10 ps +38% |

## suggest_quantization

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `1B_1gpu` | 5.60 ns | 6.10 ns +9% | 11.96 ns +114% |
| `7B_1gpu` | 5.63 ns | 6.25 ns +11% | 12.26 ns +118% |
| `13B_1gpu` | 5.61 ns | 6.69 ns +19% | 12.11 ns +116% |
| `70B_1gpu` | 5.82 ns | 6.68 ns +15% | 13.13 ns +126% |
| `405B_1gpu` | 5.81 ns | 9.76 ns +68% | 13.16 ns +127% |

## registry_queries

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `available_4dev` | 60.88 ns | 68.44 ns +12% | 123.51 ns +103% |
| `available_129dev` | 424.73 ns | 412.03 ns | 994.93 ns +134% |
| `available_61dev_mixed` | 215.21 ns | 227.41 ns +6% | 456.05 ns +112% |
| `best_available_129dev` | 114.14 ns | 163.23 ns +43% | 371.62 ns +226% |
| `total_memory_129dev` | 43.71 ns | 46.07 ns +5% | 95.15 ns +118% |
| `total_accelerator_memory_129dev` | 70.68 ns | 75.32 ns +7% | 134.23 ns +90% |
| `has_accelerator_129dev` | 1.85 ns | 1.44 ns **-22%** | 2.81 ns +52% |
| `by_family_gpu_61dev` | 205.52 ns | 627.09 ns +205% | 500.39 ns +143% |
| `by_family_tpu_61dev` | 88.53 ns | 125.15 ns +41% | 197.54 ns +123% |
| `satisfying_gpu_61dev` | 240.19 ns | 252.99 ns +5% | 545.64 ns +127% |
| `satisfying_any_accel_61dev` | 260.85 ns | 288.71 ns +11% | 570.78 ns +119% |

## cached_registry

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `get_cached_hit` | 31.21 ns | 31.97 ns | 56.30 ns +80% |
| `invalidate` | 5.27 ns | 4.15 ns **-21%** | 8.57 ns +63% |

## lazy_registry

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `new` | 13.11 µs | 12.83 µs | 18.66 µs +42% |
| `by_family_gpu_cold` | 1.20 ms | 1.63 ms +36% | 1.94 ms +62% |
| `by_family_gpu_warm` | 61.89 ns | 54.30 ns **-12%** | 105.38 ns +70% |
| `into_registry` | 2.01 ms | 2.33 ms +16% | 6.44 ms +220% |

## large_registry_sharding

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `plan_sharding_70B_128gpu` | 1.18 µs | 1.30 µs +10% | 2.04 µs +73% |
| `plan_sharding_405B_128gpu` | 1.14 µs | 1.25 µs +9% | 1.21 µs +6% |
| `suggest_quantization_70B_128gpu` | 241.30 ns | 239.10 ns | 178.75 ns **-26%** |
| `plan_sharding_70B_mixed_61dev` | 137.14 ns | 160.47 ns +17% | 165.47 ns +21% |

## large_json

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `serialize_129dev` | 17.21 µs | 21.89 µs +27% | 24.44 µs +42% |
| `deserialize_129dev` | 26.86 µs | 31.04 µs +16% | 49.37 µs +84% |

## training_memory

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `7B_full_finetune_gpu` | 3.06 ns | 3.91 ns +28% | 5.41 ns +77% |
| `7B_lora_gpu` | 3.06 ns | 3.97 ns +30% | 4.86 ns +59% |
| `7B_qlora_4bit_gpu` | 3.35 ns | 5.33 ns +59% | 6.64 ns +98% |
| `7B_qlora_8bit_gpu` | 3.52 ns | 4.32 ns +23% | 5.50 ns +56% |
| `7B_prefix_gpu` | 4.86 ns | 5.95 ns +22% | 6.27 ns +29% |
| `7B_dpo_gpu` | 5.14 ns | 5.28 ns | 5.24 ns |
| `7B_rlhf_gpu` | 5.17 ns | 4.78 ns **-7%** | 7.27 ns +41% |
| `7B_distillation_gpu` | 5.15 ns | 4.93 ns **-4%** | 7.02 ns +36% |

## training_targets

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `7B_full_gpu` | 4.25 ns | 6.12 ns +44% | 6.26 ns +47% |
| `7B_full_tpu` | 3.97 ns | 7.42 ns +87% | 4.92 ns +24% |
| `7B_full_gaudi` | 3.95 ns | 8.12 ns +106% | 4.35 ns +10% |
| `7B_full_cpu` | 3.74 ns | 4.13 ns +10% | 3.13 ns **-17%** |

## training_model_sizes

| Benchmark | Baseline (`19a2753`) | Previous (`19a2753`) | Current (`ebbb9e1`) |
|-----------|------|------|------|
| `1B_lora_gpu` | 5.37 ns | 6.16 ns +15% | 2.80 ns **-48%** |
| `7B_lora_gpu` | 5.62 ns | 3.60 ns **-36%** | 2.80 ns **-50%** |
| `13B_lora_gpu` | 5.21 ns | 4.18 ns **-20%** | 2.76 ns **-47%** |
| `70B_lora_gpu` | 5.38 ns | 4.64 ns **-14%** | 2.98 ns **-45%** |
| `405B_lora_gpu` | 2.80 ns | 5.05 ns +81% | 2.74 ns |

---

Generated by `./scripts/bench-history.sh`. Full history in `bench-history.csv`.
