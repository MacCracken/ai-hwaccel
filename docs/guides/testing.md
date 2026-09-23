# Testing Guide

How to run the tests, check detection on real hardware, and read the results.

## Quick start

Run everything from the repository root (some tests read `tests/fixtures/`):

```sh
cyrius tests tests/tcyr                          # 950 assertions in 15 units
cyrius fuzz                                      # the 6 fuzz harnesses
for f in src/*.cyr src/detect/*.cyr; do cyrius lint --strict "$f"; done   # fails on a warning
for f in src/*.cyr src/detect/*.cyr tests/tcyr/*.tcyr fuzz/*.fcyr benches/*.bcyr; do
    cyrius fmt "$f" --check                      # plain `cyrius fmt <file>` rewrites the file
done
```

CI (`.github/workflows/ci.yml`) runs the same checks, plus `cyrius vet`, the
raw-offset guard, the `dist/` drift check and the `cyrius.lock` hash check. Its
lint step runs without `--strict`, so a lint warning does not fail CI yet.

## Test categories

Tests are 15 `.tcyr` units under `tests/tcyr/`, with 950 assertions in total.
The README's *Test units* table lists what each unit covers. Real tool output
used as fixtures lives in `tests/fixtures/`.

| Suite | Location | What it tests |
|---|---|---|
| Unit tests | `tests/tcyr/*.tcyr` | Types, profiles, registry queries and totals, the duplicate-device pass, every detection entry point, parsers, planning, JSON output and round-trip, Windows, lazy detection |
| Fuzz harnesses | `fuzz/*.fcyr` | The CUDA, Vulkan, Neuron, Gaudi, Apple and model-format parsers |
| Benchmarks | `benches/*.bcyr` | 18 rows in two suites, `parsing` and `registry` |

## Running benchmarks

```sh
./scripts/bench-history.sh             # both suites; appends to bench-history.csv
cyrius bench benches/parsing.bcyr      # one suite
cyrius bench benches/registry.bcyr
```

Run them from the repository root: two parsing rows read
`tests/fixtures/vulkaninfo/`, and skip themselves when they cannot.

## Selecting backends

Every backend is compiled in. To test one backend, select it at run time with
a builder mask (see the production guide's *Selective detection*), for example
`registry_detect_with(builder_with(builder_none(), BACKEND_ROCM))`.

## Hardware-dependent testing

ai-hwaccel detects hardware by reading sysfs and `/dev`, calling OS APIs
(sysctl on macOS, DXGI on Windows) and running vendor tools. To check
detection on a machine, install the tools for its hardware.

### Packages by platform

#### Arch Linux

```sh
# AMD or Intel GPU: vulkaninfo (ROCm itself needs only the amdgpu driver)
sudo pacman -S vulkan-tools vulkan-radeon vulkan-intel

# NVIDIA GPU
sudo pacman -S nvidia-utils    # provides nvidia-smi
```

#### Ubuntu / Debian

```sh
# AMD or Intel GPU (mesa-vulkan-drivers also installs lavapipe, which is ignored)
sudo apt install vulkan-tools mesa-vulkan-drivers

# NVIDIA GPU
sudo apt install nvidia-utils-535   # or the current driver version
```

#### Fedora

```sh
# AMD or Intel GPU
sudo dnf install vulkan-tools mesa-vulkan-drivers

# NVIDIA GPU (RPM Fusion)
sudo dnf install xorg-x11-drv-nvidia-cuda
```

Intel's `xpu-smi` (data-center GPUs) comes from Intel's repository:
<https://dgpu-docs.intel.com/>.

### Validating detection

```sh
build/ai-hwaccel --table --log-level debug
```

On the Linux dev host (Ryzen 7 5800H, its Radeon iGPU reported by ROCm, with
`vulkaninfo`'s name for it):

```
ID     Device                       Memory     Free       Family   Status
-----  ---------------------------  ---------  ---------  -------  ------
0      CPU  54 GiB     -          CPU    OK
0      AMD Radeon Graphics (RADV RENOIR)  8 GiB     7 GiB     GPU    OK
```

The debug log on stderr shows each backend, the tools that were missing, and
a `dedup:` line for each duplicate view of a device that was dropped.

### What each tool enables

| Tool | Package | What it provides |
|---|---|---|
| `vulkaninfo` | `vulkan-tools` | Vulkan devices: name, type, vendor; an integrated GPU's memory heaps |
| `nvidia-smi` | `nvidia-utils` | CUDA GPUs: VRAM, compute capability, driver, temperature, power, PCI id |
| `hl-smi` | Habana SDK | Intel Gaudi |
| `neuron-ls` | AWS Neuron SDK | Inferentia / Trainium |
| `xpu-smi` | Intel oneAPI | Intel Data Center GPU Max / Flex |
| `cerebras_cli` | Cerebras SDK | Cerebras WSE |
| `gc-info` | Graphcore SDK | Graphcore IPU |
| `system_profiler` | macOS built-in | Fallback only: the Apple chip and memory when sysctl cannot name the chip |
| `wmic` | Windows built-in (gone from Windows 11 24H2) | Fallback only: GPUs and RAM when DXGI or `GlobalMemoryStatusEx` fail |

### Paths read directly

These need no tool:

| Path | Backend |
|---|---|
| `/sys/class/drm/card*/device/{driver,vendor,device,mem_info_vram_total}` | AMD ROCm; the Vulkan sysfs scan |
| `/sys/class/accel/accel*/device/{driver,tpu_version,chip_count}` | Google TPU, AMD XDNA |
| `/sys/class/misc/intel_npu` | Intel NPU |
| `/sys/class/qaic`, `/dev/qaic_*` | Qualcomm Cloud AI 100 |
| `/sys/class/misc/samsung_npu` | Samsung NPU |
| `/sys/class/misc/mtk_apu` | MediaTek APU |
| `/dev/neuron*` + `/sys/devices/virtual/dmi/id/product_name` | AWS Neuron (fallback) |
| `/dev/cerebras*`, `/dev/ipu*`, `/dev/groq*` | Cerebras, Graphcore (fallbacks), Groq |
| `/proc/meminfo` | CPU memory (Linux; macOS uses `sysctl hw.memsize`, Windows `GlobalMemoryStatusEx`) |

## CI matrix

- `ubuntu-latest` (x86_64 Linux): the whole test suite, fuzzing, benchmarks
  and the static checks (`ci.yml`).
- `macos-14` (arm64): builds the macOS wheel's binary but does not run it yet
  (the roadmap's `macos-smoke`).
- `windows-latest`: `windows-smoke` runs the cross-built EXE: JSON shape,
  logging, `--version`, and memory and GPUs checked against what Windows
  reports.

The one non-stdlib dependency, bayan's JSON sublib, is pinned by commit in
`cyrius.lock`, and CI checks the vendored files' hashes on every run.
