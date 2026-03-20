# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## Pre-v1

- [x] Publish `0.19.3` to crates.io.

---

## System I/O and monitoring (0.20)

Hardware bandwidth and topology detection beyond device enumeration.

- [x] **VRAM bandwidth probing** — calculate memory bandwidth from clock speed
  and bus width. NVIDIA via `nvidia-smi --query-gpu=clocks.max.memory`, AMD
  via sysfs `pp_dpm_mclk`. Exposed as `AcceleratorProfile::memory_bandwidth_gbps`.
- [x] **Runtime VRAM usage** — current used/free VRAM (not just total) for
  live capacity planning. Parse `nvidia-smi` and ROCm sysfs for used/free.
- [x] **PCIe link detection** — read sysfs `current_link_width`/`current_link_speed`
  to estimate host-to-device transfer rates. Expose as
  `AcceleratorProfile::pcie_bandwidth_gbps`.
- [x] **NUMA topology** — read `/sys/bus/pci/devices/<addr>/numa_node` to map
  which GPUs are on which NUMA node. Expose as `AcceleratorProfile::numa_node`.
- [x] **Network interconnect detection** — detect InfiniBand, RoCE via
  `/sys/class/infiniband/`, NVLink via `nvidia-smi nvlink -s`. Critical for
  multi-node training planning.
- [x] **Disk I/O throughput** — probe `/sys/block/*/queue/` for NVMe vs HDD.
  Estimate data loading bottlenecks.
- [x] **Network ingestion estimation** — `SystemIo::estimate_ingestion_secs()`
  estimates data loading time given dataset size + detected storage bandwidth.

---

## Detection gaps (hardware-dependent)

Require real hardware or cloud instances. No deadline — addressed as access
becomes available.

### Accuracy improvements

- [ ] **AMD ROCm** — `rocm-smi` for clock speeds, firmware version, XGMI
  topology, power draw.
- [ ] **Google TPU** — validate on real GCE VMs. Multi-host pod slices where
  chips span `/dev/accel*` nodes across hosts.
- [ ] **Intel Gaudi** — test on Gaudi 3. Parse firmware version from `hl-smi`.
- [ ] **AWS Neuron** — mixed Inferentia + Trainium in same instance. Test
  `neuron-ls` output format across SDK versions.
- [ ] **Intel oneAPI** — Data Center GPU Max (Ponte Vecchio) HBM vs DDR memory
  tiers. Test `xpu-smi` on real hardware.
- [ ] **Vulkan** — parse compute queue families and subgroup sizes from
  `vulkaninfo` full output (not just `--summary`).

### New backends

- [ ] **AMD MI300X** — CXL-attached memory, unified HBM pool detection.
- [ ] **Cerebras WSE** — `/dev/cerebras*` or SDK introspection.
- [ ] **Graphcore IPU** — `gc-info` CLI parsing.
- [ ] **Groq LPU** — when Linux driver is publicly available.
- [ ] **Samsung NPU** — Exynos AI accelerator sysfs paths.
- [ ] **MediaTek APU** — Android/Linux APU sysfs detection.
- [ ] **NVIDIA Grace Hopper** — unified memory detection via NVSwitch.
- [ ] **AMD Instinct MI350** — next-gen detection when available.

### Platform gaps

- [ ] **macOS ANE via IOKit** — native ANE core count and performance tier
  detection (current `system_profiler` path covers Metal GPU but ANE is
  estimated).
- [ ] **Windows** — WMI queries for GPU detection, DirectML device
  enumeration, `nvidia-smi.exe` path resolution.
- [ ] **Android** — HAL `hwbinder` for NNAPI accelerator list.
- [ ] **FreeBSD** — DRM sysctl equivalents for GPU detection.

---

## API gaps

- [ ] **Async detection** improvements — current `detect_async` uses
  `spawn_blocking`. True async subprocess I/O via `tokio::process::Command`
  would avoid the blocking thread entirely.
- [x] **Stable JSON schema v2** — bumped `SCHEMA_VERSION` to 2. Includes
  device topology, bandwidth, NUMA, and system I/O fields.
- [x] **`--table` enhancements** — `--columns name,mem,bw` for column
  selection, `--tsv` for machine-readable tab-separated output.
- [x] **`--watch` improvements** — shows delta diffs (memory usage changes
  between refreshes), `--alert mem>90` for threshold alerts.
- [x] **Error recovery** — `DetectionError::Timeout` as distinct variant,
  separate from `ToolFailed`. Enables retry logic for slow tools.

---

## Testing gaps

- [ ] **Hardware-in-the-loop CI** — run detection tests on cloud instances
  with real GPUs (GitHub Actions self-hosted runner or cloud CI).
- [ ] **Fuzz testing for parsers** — fuzz `vulkaninfo`, `nvidia-smi`, `hl-smi`
  output parsers with `cargo-fuzz` or AFL to find crash/panic paths.
- [ ] **Windows integration tests** — current mock tests gate symlinks behind
  `#[cfg(unix)]`. Need Windows-native mock strategies.
- [ ] **Benchmark regression CI** — track benchmark numbers across releases
  to catch performance regressions.
- [ ] **Load testing** — verify detection under high concurrency (many threads
  calling `detect()` or `CachedRegistry::get()` simultaneously).

---

## Security gaps

- [x] **Windows `which()` improvements** — tries `.exe`, `.cmd`, `.bat`
  extensions when name has no extension. Matches standard `PATHEXT` behavior.
- [x] **Subprocess environment sanitization** — `run_tool()` strips
  `LD_PRELOAD`, `LD_LIBRARY_PATH`, `DYLD_INSERT_LIBRARIES`,
  `DYLD_LIBRARY_PATH` from child processes.
- [x] **TOCTOU in `which()`** — documented as accepted risk in `run_tool()`
  and `which()` doc comments. Equivalent to shell behavior.

---

## Documentation gaps

- [x] **Crate-level guide expansion** — added sections on error handling
  patterns, custom backend implementation, serde integration, and system I/O.
- [x] **Troubleshooting guide** — `docs/troubleshooting.md` covers common
  issues with detection, permissions, and configuration.
- [x] **Performance tuning guide** — `docs/performance.md` covers caching,
  selective detection, feature flags, and async usage.
- [x] **Migration guide** — `docs/migration.md` documents all breaking
  changes from v0.19.3 to v0.20.3.

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
