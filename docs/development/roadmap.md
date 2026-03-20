# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## Detection gaps (hardware-dependent)

Require real hardware or cloud instances. No deadline — addressed as access
becomes available.

### Accuracy improvements

- [x] **AMD ROCm** — sysfs for clock speeds (`pp_dpm_sclk`, `pp_dpm_mclk`),
  VBIOS version, GPU temperature, power draw, utilization. XGMI topology
  deferred (needs multi-GPU system).
- [ ] **Google TPU** — validate on real GCE VMs. Multi-host pod slices where
  chips span `/dev/accel*` nodes across hosts.
- [ ] **Intel Gaudi** — test on Gaudi 3. Parse firmware version from `hl-smi`.
- [ ] **AWS Neuron** — mixed Inferentia + Trainium in same instance. Test
  `neuron-ls` output format across SDK versions.
- [ ] **Intel oneAPI** — Data Center GPU Max (Ponte Vecchio) HBM vs DDR memory
  tiers. Test `xpu-smi` on real hardware.
- [x] **Vulkan** — full `vulkaninfo` parsing for compute queue families,
  queue counts, and subgroup sizes. Exposed in `compute_capability` field.

### New backends

- [x] **AMD MI300X** — CXL-attached memory detection via sysfs
  (`mem_info_vis_vram_total` and `/sys/bus/cxl/devices/`). Unified HBM pool
  reported in `memory_bytes`. Untested — needs MI300X hardware.
- [x] **Cerebras WSE** — `cerebras_cli system-info` + `/dev/cerebras*`
  fallback. 44 GB SRAM default. Untested — needs Cerebras hardware.
- [x] **Graphcore IPU** — `gc-info -d N -i` JSON parsing + `/dev/ipu*`
  fallback. 900 MB SRAM default. Untested — needs Graphcore hardware.
- [x] **Groq LPU** — `/dev/groq*` placeholder. 230 MB SRAM default.
  Untested — Linux driver not yet public.
- [x] **Samsung NPU** — `/sys/class/misc/samsung_npu` + `/dev/samsung_npu*`.
  2 GB shared memory. Untested — needs Exynos device.
- [x] **MediaTek APU** — `/sys/class/misc/mtk_apu` + `/dev/mtk_mdla*`.
  2 GB shared memory. Untested — needs MediaTek device.
- [x] **NVIDIA Grace Hopper** — detects GH200/GH100 from `nvidia-smi` GPU
  name, adds 480 GB unified LPDDR5X to reported HBM for capacity planning.
  Untested — needs Grace Hopper hardware.
- [x] **AMD Instinct MI350** — covered by MI300X CXL detection path; same
  amdgpu sysfs interface. Untested — hardware not yet available.

### Platform gaps

- [ ] **macOS ANE via IOKit** — native ANE core count and performance tier
  detection (current `system_profiler` path covers Metal GPU but ANE is
  estimated).
- [ ] **Windows** — WMI queries for GPU detection, DirectML device
  enumeration, `nvidia-smi.exe` path resolution.
- [ ] **Android** — HAL `hwbinder` for NNAPI accelerator list.
- [ ] **FreeBSD** — DRM sysctl equivalents for GPU detection.

---

## Testing gaps

- [x] **Hardware integration tests** — `tests/hardware_integration.rs` with
  17 tests covering CPU, ROCm, Vulkan, PCIe, bandwidth, storage, interconnects,
  JSON roundtrip, and concurrent detection. Auto-skips when hardware is absent.
- [x] **Load testing** — concurrent 4-thread detection test + benchmark in
  `benches/detect.rs`. Verifies `detect()` and `CachedRegistry::get()` are
  safe under concurrency.
- [x] **System I/O benchmarks** — per-backend detection, serialization,
  deserialization, and system I/O query benchmarks in `benches/detect.rs`.
- [x] **Fuzz testing for parsers** — 9 fuzz targets via `cargo-fuzz` covering
  cuda, gaudi, vulkan, neuron, apple, intel_oneapi, nvlink, bandwidth, and
  small parsers (IB rate, PCIe speed, DPM clock). Found and fixed integer
  overflow in CUDA parser.
- [ ] **Windows integration tests** — current mock tests gate symlinks behind
  `#[cfg(unix)]`. Need Windows-native mock strategies.
- [ ] **Benchmark regression CI** — track benchmark numbers across releases
  to catch performance regressions.

---

## Post-v1 features

Longer-term items that don't block any release.

- [ ] **Topology-aware sharding** — NVLink/NVSwitch/XGMI/ICI topology for
  optimal tensor placement and communication minimization.
- [ ] **Power and thermal monitoring** — current power draw, temperature,
  throttling state via `nvidia-smi`, `rocm-smi`.
- [ ] **Hot-plug support** — `udev`/`inotify` watch for device add/remove
  with dynamic registry updates.
- [ ] **Cost-aware planning** — cloud pricing data for cheapest device
  configuration per workload.
- [ ] **WASM target** — browser-based ML dashboards with stubbed detection.
- [ ] **Full Python bindings** — complete PyO3 package with pip install.
- [ ] **Reduce allocations** — `Cow<str>` for device names, `SmallVec` for
  profiles (most systems have < 8 accelerators).
- [ ] **Lazy detection** — detect only backends the caller queries, not all
  enabled backends upfront.
- [ ] **Multi-node detection** — SSH/gRPC to remote nodes to build a
  cluster-wide registry for distributed training planning.

---

## Non-goals

- **Runtime execution** — detection and planning only, not inference/training.
- **Kernel driver management** — no installing or configuring drivers.
- **Cloud provisioning** — detect what's present, not what could be spun up.
- **Model format parsing** — we estimate from parameter count, not from
  reading `.safetensors` or `.gguf` files.
