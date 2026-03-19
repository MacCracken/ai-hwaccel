# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## Pre-v1

- [ ] Publish `0.19.3` to crates.io.

---

## System I/O and monitoring (0.20)

Hardware bandwidth and topology detection beyond device enumeration.

- [ ] **VRAM bandwidth probing** — benchmark actual memory throughput per
  device via `nvidia-smi -q -d MEMORY`, `rocm-smi --showmeminfo`. Expose
  as `AcceleratorProfile::memory_bandwidth_gbps`.
- [ ] **Runtime VRAM usage** — current used/free VRAM (not just total) for
  live capacity planning. Parse `nvidia-smi -q -d MEMORY` free field.
- [ ] **PCIe link detection** — read `lspci -vv` for link width/speed to
  estimate host-to-device transfer rates. Expose as
  `AcceleratorProfile::pcie_bandwidth_gbps`.
- [ ] **NUMA topology** — parse `/sys/devices/system/node/` to map which GPUs
  are on which NUMA node. Expose as `AcceleratorProfile::numa_node`.
- [ ] **Network interconnect detection** — detect InfiniBand, RoCE, NVLink
  via `ibstat`, `ibv_devinfo`, `nvidia-smi nvlink -s`. Critical for
  multi-node training planning.
- [ ] **Disk I/O throughput** — probe `/sys/block/*/queue/` for NVMe vs HDD,
  detect RAID. Estimate data loading bottlenecks.
- [ ] **Network ingestion estimation** — given dataset size + detected network
  bandwidth, estimate data loading time for training jobs.

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
- [ ] **Stable JSON schema v2** — add device topology, bandwidth, and NUMA
  fields when system I/O detection ships. Bump `SCHEMA_VERSION` to 2.
- [ ] **`--table` enhancements** — column selection (`--columns name,mem,bw`),
  machine-readable TSV output.
- [ ] **`--watch` improvements** — show delta diffs (memory usage changes),
  alert thresholds.
- [ ] **Error recovery** — `DetectionError::Timeout` as distinct variant
  (currently bundled into `ToolFailed`).

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

- [ ] **Windows `which()` improvements** — handle `.exe`, `.cmd`, `.bat`
  extensions. Current `which()` checks `is_file()` which may miss
  extensionless executables on Windows.
- [ ] **Subprocess environment sanitization** — currently inherits full
  parent environment. Consider clearing `LD_PRELOAD`, `DYLD_INSERT_LIBRARIES`
  before spawning CLI tools.
- [ ] **TOCTOU in `which()`** — tool is resolved to absolute path, then
  invoked. In theory the file could be replaced between resolution and
  execution. Document as accepted risk (same as shell behavior).

---

## Documentation gaps

- [ ] **Crate-level guide expansion** — add sections on error handling
  patterns, custom backend implementation, and serde integration.
- [ ] **Troubleshooting guide** — common issues: "nvidia-smi not found",
  "detection returns CPU only", "Vulkan GPU missing".
- [ ] **Performance tuning guide** — when to use `CachedRegistry`, when to
  use `DetectBuilder::none()`, feature flag impact on compile time.
- [ ] **Migration guide** — document breaking changes between versions for
  downstream users.

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
