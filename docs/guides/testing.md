# Testing Guide

How to run tests locally, set up hardware-dependent testing, and interpret
results.

## Quick start

```sh
make check          # fmt + clippy + test (same as CI)
cargo test          # unit + integration + doc tests
cargo test --test mock_detection   # mock sysfs tests only
cargo test --test integration      # integration tests only
```

## Test categories

| Suite | Count | Location | What it tests |
|---|---|---|---|
| Unit tests | 140 | `src/tests/` | Types, classification, quantization, registry queries, sharding, training, serde, display, builder, command runner, property-based |
| Integration tests | 9 | `tests/integration.rs` | End-to-end detect→query→plan pipeline |
| Mock detection | 11 | `tests/mock_detection.rs` | Fake sysfs trees, serde rejection, schema validation |
| Doc tests | 13 | Inline in source | Code examples in rustdoc compile and run |
| Benchmarks | 6 | `benches/` | Performance of detect, plan, estimate functions |

## Running benchmarks

```sh
cargo bench                    # all benchmarks
cargo bench --bench detect     # detection benchmarks only
cargo bench --bench plan       # planning benchmarks only
```

## Feature flag testing

```sh
# Test with no backends (CPU-only detection)
cargo test --no-default-features

# Test with specific backends
cargo test --no-default-features --features cuda,tpu

# Test with async support
cargo test --features async-detect
```

## Hardware-dependent testing

The crate detects hardware by probing sysfs and running CLI tools. To test
detection accuracy, you need either real hardware or the corresponding tools
installed.

### Required packages by platform

#### Arch Linux

```sh
# AMD GPU (Vulkan + ROCm)
sudo pacman -S vulkan-tools vulkan-radeon rocm-smi-lib

# NVIDIA GPU
sudo pacman -S nvidia-utils    # provides nvidia-smi

# Intel GPU (oneAPI)
# Install via Intel's repo: https://dgpu-docs.intel.com/
```

#### Ubuntu / Debian

```sh
# AMD GPU
sudo apt install vulkan-tools mesa-vulkan-drivers rocm-smi

# NVIDIA GPU
sudo apt install nvidia-utils-535   # or current driver version

# Intel GPU
sudo apt install intel-gpu-tools
```

#### Fedora

```sh
# AMD GPU
sudo dnf install vulkan-tools mesa-vulkan-drivers rocm-smi

# NVIDIA GPU (RPM Fusion)
sudo dnf install xorg-x11-drv-nvidia-cuda
```

### Validating detection

After installing tools, verify detection works:

```sh
# Full detection with debug logging
cargo run -- --table --debug

# Expected output for AMD GPU system:
# ID     Device                        Memory   Family  Status
# ----------------------------------------------------------------
# 0      CPU                           59.8 GB  CPU     ok
# 1      ROCm GPU (device 0)            3.0 GB  GPU     ok
```

### What each tool enables

| Tool | Package | What it tests |
|---|---|---|
| `vulkaninfo` | `vulkan-tools` | Vulkan device name, memory heaps, API version, driver version |
| `nvidia-smi` | `nvidia-utils` | CUDA GPU detection, VRAM, compute capability, driver version |
| `rocm-smi` | `rocm-smi-lib` | AMD GPU clock speeds, firmware, temperature |
| `hl-smi` | Habana SDK | Intel Gaudi HPU detection |
| `neuron-ls` | AWS Neuron SDK | Inferentia/Trainium detection |
| `xpu-smi` | Intel oneAPI | Intel Arc / Data Center GPU Max |
| `system_profiler` | macOS built-in | Apple Silicon chip name, unified memory |

### sysfs paths tested

These paths are read directly (no tools needed):

| Path | Backend |
|---|---|
| `/sys/class/drm/card*/device/driver` | AMD ROCm |
| `/sys/class/drm/card*/device/mem_info_vram_total` | AMD ROCm VRAM |
| `/sys/class/misc/intel_npu` | Intel NPU |
| `/sys/class/accel/accel*/device/driver` | AMD XDNA / Google TPU |
| `/sys/class/qaic` | Qualcomm Cloud AI |
| `/dev/neuron*` | AWS Neuron (fallback) |
| `/dev/accel*` | Google TPU |
| `/proc/meminfo` | CPU system memory |

## CI matrix

CI runs on:

- `ubuntu-latest` (x86_64 Linux)
- `macos-latest` (aarch64 Apple Silicon)
- `windows-latest` (x86_64 Windows)

Plus a dedicated MSRV job with Rust 1.89.

## Supply-chain checks

```sh
make audit          # cargo-audit (known vulnerabilities)
make deny           # cargo-deny (licenses, advisories, sources)
make vet            # cargo-vet (dependency audits)
```
