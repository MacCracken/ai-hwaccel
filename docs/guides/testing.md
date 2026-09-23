# Testing Guide

How to run tests locally, set up hardware-dependent testing, and interpret
results.

## Quick start

```sh
cyrius tests                         # run all 950 assertions (15 test units)
cyrius lint src/main.cyr             # lint (zero warnings)
cyrius fmt src/main.cyr --check      # check formatting
```

## Test categories

Tests are 15 `.tcyr` units under `tests/tcyr/`, with 950 assertions in total.
The README's *Test units* table lists what each unit covers. Real tool output
used as fixtures lives in `tests/fixtures/` (read relative to the repository
root, where CI runs the tests).

| Suite | Location | What it tests |
|---|---|---|
| Unit tests | `tests/tcyr/*.tcyr` | Types, profiles, registry queries and totals, every detection entry point, parsers, planning, JSON output and round-trip, Windows, lazy detection |
| Fuzz harnesses | `fuzz/*.fcyr` | The CUDA, Vulkan, Neuron, Gaudi, Apple and model-format parsers |
| Benchmarks | `benches/*.bcyr` | 18 rows in two suites, `parsing` and `registry` |

## Running benchmarks

```sh
cyrius bench                   # all benchmarks
cyrius bench detect            # detection benchmarks only
cyrius bench plan              # planning benchmarks only
```

## Backend selection at compile time

```sh
# Build with specific backends only
cyrius build src/main.cyr build/ai-hwaccel -DCUDA -DTPU

# Build with no backends (CPU-only detection)
cyrius build src/main.cyr build/ai-hwaccel -DNO_BACKENDS
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
build/ai-hwaccel --table --log-level debug

# Expected output for AMD GPU system:
# ID     Device                        Memory   Family  Status
# ----------------------------------------------------------------
# 0      CPU                           59.8 GB  CPU     ok
# 1      ROCm GPU (device 0)            3.0 GB  GPU     ok
```

### What each tool enables

| Tool | Package | What it tests |
|---|---|---|
| `vulkaninfo` | `vulkan-tools` | Vulkan device name, type and vendor; an integrated GPU's memory heaps |
| `nvidia-smi` | `nvidia-utils` | CUDA GPU detection, VRAM, compute capability, driver version |
| `rocm-smi` | `rocm-smi-lib` | AMD GPU clock speeds, firmware, temperature |
| `hl-smi` | Habana SDK | Intel Gaudi HPU detection |
| `neuron-ls` | AWS Neuron SDK | Inferentia/Trainium detection |
| `xpu-smi` | Intel oneAPI | Intel Arc / Data Center GPU Max |
| `system_profiler` | macOS built-in | Fallback only: Apple chip name + memory when sysctl cannot name the chip |

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

No supply-chain checks needed -- zero external dependencies.
