# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## 0.21.3 — Platform & Planning

Focus: lazy detection, topology-aware sharding, cloud hardware validation,
Python bindings groundwork.

### Detection performance

Current bottleneck: `vulkaninfo` takes ~5s on AMD Cezanne iGPU. Total
detection is 5.05s, of which 5.0s is vulkaninfo. All other backends
(ROCm sysfs, PCIe, bandwidth, NUMA, storage, disk) complete in <10ms.

- [x] **Lazy detection** — detect only backends the caller queries, not all
  enabled backends upfront. `LazyRegistry::new()` returns a registry
  that probes on first access per family. Avoids spawning nvidia-smi when
  caller only needs TPU info.
- [x] **`vulkaninfo` timeout + caching** — `vulkaninfo` is the single
  slowest probe (~5s). Per-process cache writes parsed results to
  `$XDG_CACHE_HOME/ai-hwaccel/vulkan.json` with a 60s TTL. On subsequent
  calls within the TTL, reads from cache instead of re-running vulkaninfo.
  3s timeout on the subprocess — if vulkaninfo hangs, falls back
  to sysfs-only Vulkan detection (`/sys/class/drm/card*/device/vendor`).
- [x] **Parallel backend probing** — CLI-based backends (nvidia-smi,
  vulkaninfo, rocm-smi, hl-smi) run concurrently via `std::thread::scope`
  (sync) and `tokio::process::Command` (async). Sysfs-only backends already
  complete in <1ms and don't benefit from parallelism. Detection time
  is max(slowest tool) — on systems with both CUDA and Vulkan, this
  halves detection time. *(Already implemented in 0.20.)*
- [x] **Sysfs-only Vulkan fallback** — for systems where `vulkaninfo` is
  slow or absent, detect Vulkan-capable GPUs via
  `/sys/class/drm/card*/device/{vendor,device}` + PCI ID lookup table.
  Provides device name and VRAM estimate without spawning a subprocess.
  Used as automatic fallback when vulkaninfo is missing or times out.
- [x] **Detection result caching** — `DiskCachedRegistry::new(ttl)`
  persists detection results to `$XDG_CACHE_HOME/ai-hwaccel/registry.json`.
  Subsequent calls within TTL return cached data instantly. Also available
  as in-memory `CachedRegistry` for thread-safe caching without disk I/O.
- [x] **Per-backend timing** — `--profile` CLI flag and
  `AcceleratorRegistry::detect_with_timing()` API that returns
  `TimedDetection` with `HashMap<String, Duration>` showing how long
  each backend took. Enables users to identify and disable slow backends.
- [x] **Topology-aware sharding** — uses interconnect data (NVLink, XGMI, ICI)
  from `SystemIo` to generate sharding plans that minimize cross-link
  transfers. Pipeline parallel prefers NUMA-local GPU pairs. Tensor
  parallel prefers NVSwitch-connected groups (>100 GB/s interconnect).
- [x] **Cost-aware planning** — static pricing table for common cloud GPU
  instances (A100, H100, L4, T4, MI300X, TPU v5e). Given model size +
  quantisation, `cost::recommend_instance()` returns cheapest viable config.
  Data in `data/cloud_pricing.json`, updatable without recompiling.
  CLI: `ai-hwaccel --cost 70B --quant bf16`.
- [x] **Container/VM detection** — detects Docker (`/.dockerenv`),
  Kubernetes (`/var/run/secrets/kubernetes.io`), cloud instance type
  (AWS DMI, GCE DMI, Azure DMI — no HTTP metadata calls needed).
  Exposed as `SystemIo::environment` for deployment-aware planning.

### Python bindings (groundwork)

- [x] **PyO3 module scaffold** — `py/` directory with `maturin` build,
  wrapping `AcceleratorRegistry::detect()`, `suggest_quantization()`,
  `plan_sharding()`, `system_io()`. Ship as `ai-hwaccel` on PyPI.
- [x] **Python type stubs** — `.pyi` files for IDE support.
- [x] **Python examples** — basic detection, sharding plan, training memory
  estimation.

### Cloud hardware validation (staged)

Spin up short-lived cloud instances to validate untested backends. Fix any
parser bugs found, add mock test fixtures from captured tool output.

- [ ] **NVIDIA H100 / A100** — validate CUDA parser with real nvidia-smi
  output. Capture fixtures for bandwidth, NVLink, power/thermal. (AWS/GCP)
- [ ] **NVIDIA Grace Hopper GH200** — validate unified memory detection.
  Capture nvidia-smi output. (GCP/Lambda Labs)
- [ ] **AMD MI300X** — validate CXL memory detection, ROCm sysfs enrichment.
  Capture sysfs tree. (Azure)
- [ ] **Google TPU v5e / v5p** — validate TPU detection on GCE. Multi-host
  pod slice testing. Capture sysfs fixtures.
- [ ] **AWS Neuron (trn1/inf2)** — validate neuron-ls parser on mixed
  instances. Capture JSON fixtures for Trainium + Inferentia.
- [ ] **Intel Gaudi 3** — validate hl-smi parser on AWS DL1/DL2. Capture
  CSV fixtures.

### Cross-platform porting

- [ ] **macOS: `system_profiler` for Metal/ANE detection** — use
  `system_profiler SPDisplaysDataType -json` for GPU enumeration and
  Metal feature set, `system_profiler SPHardwareDataType` for ANE core
  count. Replace IOKit-only approach with higher-level API.
- [ ] **macOS: `sysctl` for CPU/memory** — use `sysctl hw.memsize`,
  `hw.ncpu`, `hw.cpufrequency` for system topology instead of sysfs.
- [ ] **Windows: WMI queries for GPU detection** — `Win32_VideoController`
  via WMI for GPU enumeration when nvidia-smi is absent. DirectML device
  listing via `dxdiag` parsing. `nvidia-smi.exe` path resolution
  (`C:\Windows\System32\`).
- [ ] **Windows: DirectX adapter enumeration** — DXGI `EnumAdapters1` via
  `windows-rs` for reliable GPU detection independent of vendor CLI tools.
  Returns adapter LUID, dedicated VRAM, shared memory, driver version.
- [ ] **Cross-platform: abstract sysfs probing behind platform trait** —
  `PlatformProbe` trait with `detect_gpus()`, `detect_memory()`,
  `detect_topology()` methods. Linux impl reads sysfs, macOS impl uses
  `system_profiler`/`sysctl`, Windows impl uses WMI/DXGI. Feature-gated
  backends (`sysfs`, `system-profiler`, `wmi`).

---

## 0.22.3 — Ecosystem & Scale

Focus: full Python package, multi-node detection, hot-plug, WASM, remaining
platforms.

### Python bindings (full)

- [ ] **Complete PyO3 API** — all public types wrapped: `AcceleratorProfile`,
  `SystemIo`, `Interconnect`, `StorageDevice`, `ShardingPlan`, `TrainingMethod`.
- [ ] **`pip install ai-hwaccel`** — wheels for Linux (manylinux), macOS
  (universal2), Windows (x86_64). Published via maturin + GitHub Actions.
- [ ] **Python-native features** — `detect()` returns dict-like objects,
  JSON serialization, pandas DataFrame export for multi-device registries.

### Multi-node detection

- [ ] **SSH probe** — `AcceleratorRegistry::detect_remote(hosts)` connects
  via SSH, runs `ai-hwaccel --json` on each host, merges results into a
  cluster-wide registry. Requires `ai-hwaccel` binary on remote hosts.
- [ ] **Cluster-aware sharding** — extend `plan_sharding()` to distribute
  across nodes. Consider network bandwidth (IB/RoCE) between nodes for
  pipeline vs data parallelism decisions.
- [ ] **Kubernetes integration** — detect GPUs via device plugin labels
  (`nvidia.com/gpu`, `amd.com/gpu`) from node annotations. No SSH needed.

### Hot-plug support

- [ ] **`udev` watcher (Linux)** — `AcceleratorRegistry::watch()` returns
  a stream of `DeviceEvent::Added` / `DeviceEvent::Removed` via `inotify`
  on `/dev/` and sysfs. Requires `tokio` runtime.
- [ ] **Dynamic registry updates** — `CachedRegistry` auto-invalidates on
  hot-plug events. Callbacks for device add/remove.

### WASM target

- [ ] **`wasm32-unknown-unknown` build** — stub all sysfs/subprocess
  detection. Expose `AcceleratorRegistry::from_profiles()`,
  `from_json()`, planning, and sharding for browser-based dashboards.
- [ ] **JS/TS bindings** — `wasm-bindgen` wrapper for npm package.

### Interconnect detection gaps

- [ ] **NVSwitch auto-detection** — probe `/sys/devices/virtual/nvidia-nvswitch/`
  or parse `nvidia-smi topo -m` to detect NVSwitch fabric. Currently NVSwitch
  is only inferred from NVLink link count. Affects tensor-parallel strategy
  selection in `plan_sharding()`.
- [ ] **AMD XGMI / Infinity Fabric** — detect multi-GPU XGMI topology via
  `rocm-smi --showtopo` or sysfs. Currently no XGMI detection exists, so
  AMD multi-GPU sharding underestimates interconnect bandwidth.
- [ ] **Google TPU ICI detection** — detect ICI (Inter-Chip Interconnect)
  mesh topology on TPU pod slices. Currently `InterconnectKind::Ici` is
  defined but never populated by detection code.
- [ ] **RoCE v2 detection** — distinguish RoCEv1 vs RoCEv2 via
  `/sys/class/infiniband/*/ports/*/gid_attrs/types/`. RoCEv2 supports
  routable RDMA which affects multi-node sharding decisions.

### Remaining platform gaps

- [ ] **Android** — HAL `hwbinder` for NNAPI accelerator list. Requires
  Android NDK cross-compilation.
- [ ] **FreeBSD** — DRM sysctl equivalents for GPU detection.
- [ ] **Intel oneAPI** — Data Center GPU Max (Ponte Vecchio) HBM vs DDR
  memory tiers on real hardware.

### Remaining untested backends

- [ ] **Cerebras WSE** — needs Cerebras Cloud access.
- [ ] **Graphcore IPU** — needs Paperspace or IPU cloud access.
- [ ] **Groq LPU** — blocked on public Linux driver.
- [ ] **Samsung NPU** — needs Exynos device (Galaxy S24+).
- [ ] **MediaTek APU** — needs Dimensity device.

---

## 0.23.3 — Fleet & Scale

Focus: fleet-wide inventory, health monitoring, capacity planning at
datacenter scale.

### Fleet inventory

- [ ] **Fleet registry** — `FleetRegistry` aggregates `AcceleratorRegistry`
  from multiple nodes into a unified view. Tracks node hostname, IP, rack,
  zone. Queryable: "show all H100 nodes in us-east-1".
- [ ] **Discovery protocols** — auto-discover nodes via mDNS/DNS-SD on
  local network, Consul service catalog, or Kubernetes node labels.
  Pluggable discovery backends.
- [ ] **Inventory persistence** — serialize fleet state to SQLite or JSON
  lines. Diff against previous scan to detect changes (new nodes, failed
  GPUs, memory degradation).
- [ ] **Fleet CLI** — `ai-hwaccel fleet scan --subnet 10.0.0.0/24` or
  `ai-hwaccel fleet scan --kubeconfig ~/.kube/config`. Table output with
  per-node summary, aggregate stats.

### Health monitoring

- [ ] **Continuous health checks** — periodic re-detection on fleet nodes.
  Track `temperature_c`, `power_watts`, `gpu_utilization_percent`,
  `memory_used_bytes` over time. Detect degradation trends.
- [ ] **Alert rules engine** — configurable thresholds:
  `gpu_temp > 85`, `mem_used > 95%`, `pcie_bandwidth < 50%_of_expected`,
  `device_disappeared`. Notify via webhook, Slack, PagerDuty.
- [ ] **Dead GPU detection** — compare detected device count against
  expected (from previous scan or manifest). Flag nodes with missing or
  failed accelerators.
- [ ] **ECC error tracking** — parse `nvidia-smi -q -d ECC` and ROCm
  `ras_features` for correctable/uncorrectable memory errors. Alert on
  threshold.

### Capacity planning

- [ ] **Fleet-wide sharding** — given a model + quantisation + fleet
  inventory, recommend which nodes to use and how to distribute. Consider
  inter-node bandwidth (IB/RoCE), intra-node topology (NVLink/NVSwitch),
  and power budget.
- [ ] **Bin packing** — pack multiple models onto a fleet. Given N models
  and M nodes, find optimal placement minimizing fragmentation and
  maximizing utilization.
- [ ] **Scaling recommendations** — "you need 3 more H100 nodes to run
  Llama 405B at BF16 with pipeline parallel". Based on current fleet
  capacity vs requested workload.
- [ ] **What-if analysis** — simulate adding/removing nodes before
  provisioning. "If I add 4x MI300X, can I fit two 70B models at FP16?"

### Observability & export

- [ ] **Prometheus metrics** — `/metrics` endpoint exposing per-device
  gauges: `hwaccel_gpu_temperature_celsius`, `hwaccel_gpu_power_watts`,
  `hwaccel_gpu_memory_used_bytes`, `hwaccel_gpu_utilization_percent`.
  Fleet-wide aggregates: `hwaccel_fleet_total_vram_bytes`,
  `hwaccel_fleet_gpu_count`.
- [ ] **OpenTelemetry spans** — instrument detection with OTel traces
  for observability platforms. Span per backend, attributes for device
  count and duration.
- [ ] **Grafana dashboard template** — JSON dashboard for fleet GPU
  utilization, temperature heatmap, memory usage per node.
- [ ] **Structured event log** — JSON lines log of detection events,
  device changes, alerts. Ingestible by ELK/Loki/Datadog.

### Multi-tenancy

- [ ] **Device reservation** — mark devices as reserved/in-use to prevent
  double-allocation across teams. `registry.reserve(device_id, owner)`.
- [ ] **Namespace isolation** — in Kubernetes, scope detection to the
  pod's allocated devices (GPU device plugin, resource limits).
- [ ] **Quota management** — per-team GPU hour budgets. Track usage via
  `gpu_utilization_percent` × time.

---

## Future (post-v1)

Items that don't fit in a specific release yet.

- [ ] **Model compatibility database** — which models run on which hardware
  at which quantisation. Queryable: "can I run Llama 70B on 2x RTX 4090?"
- [x] **Workload profiler** — moved to 0.21.3 as `--profile` / per-backend timing.
- [ ] **Power budget planning** — given power cap (e.g. 1000W), recommend
  device mix. Uses power_watts from detection.
- [ ] **Thermal throttling prediction** — warn when temperature_c approaches
  throttle thresholds (GPU: 83°C, TPU: 95°C).
- [ ] **Model format detection** — scan `.safetensors`, `.gguf`, `.onnx`
  headers for parameter count and dtype, eliminating manual `model_params`
  input.
- [ ] **Plugin system** — allow third-party backends via dynamic loading
  (`.so`/`.dll`) without recompiling the crate.

---

## Engineering backlog

Internal code quality improvements. Not user-facing, but reduce maintenance
burden and improve contributor experience.

### API consistency

- [ ] **Return `impl Iterator` from query methods** — `available()`,
  `by_family()`, `satisfying()` currently return `Vec<&AcceleratorProfile>`.
  Change to `impl Iterator` for zero-alloc queries. Breaking change — requires
  callers to `.collect()` explicitly.
- [ ] **Make `ShardingPlan::shards` private** — add `pub fn shards()` accessor
  to maintain plan invariants. Breaking change for direct field access.
- [ ] **`TryFrom<u32>` for `QuantizationLevel`** — map `32 → None`,
  `16 → Float16`, `8 → Int8`, `4 → Int4` for CLI parsing ergonomics.
- [ ] **Fix `CloudGpuInstance` re-export alias** — standardize to `CloudInstance`
  only. Deprecate or remove the alias in `lib.rs`.

### CLI refactoring

- [ ] **Break `print_table()` into composable helpers** — extract
  `filter_profiles()`, `sort_profiles()`, `render_header()`, `render_row()`,
  `render_footer()` from the 226-line function in `main.rs`.
- [ ] **Extract CLI mode functions** — `handle_cost_mode()`, `run_watch()`
  decomposition, `handle_profile_mode()` as standalone functions.
- [ ] **Reduce watch mode allocations** — cache `format!("{:?}")` keys,
  use `entry()` API for delta tracking HashMap.

### Detection infrastructure

- [ ] **Consolidate `run_backend!`/`spawn_backend!` macros** — merge sync and
  timed macro variants into a single parametric macro in `detect/mod.rs`.
  Currently duplicated across 3 detection paths (sync, timed, async).
- [ ] **Backend registration table** — replace hardcoded macro invocations with
  a `const BACKENDS: &[BackendDef]` table. Adding a new backend should be a
  single line addition.
- [ ] **CSV parsing helper** — extract shared `parse_csv_device_line()` for
  cuda/gaudi/intel_oneapi backends to eliminate ~30 lines of duplicate
  validation boilerplate.

### Feature flags

- [ ] **Add `minimal` and `common` feature sets** — `minimal` enables CPU-only,
  `common` enables `cuda + rocm + apple + vulkan + intel-npu`. Current default
  (`all-backends`) is heavy for embedded or minimal deployments.

---

## Non-goals

- **Runtime execution** — detection and planning only, not inference/training.
- **Kernel driver management** — no installing or configuring drivers.
- **Cloud provisioning** — detect what's present, not what could be spun up.
