# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## 0.24.3 — Platform & Validation

### Cloud hardware validation

Spin up short-lived cloud instances to validate untested backends. Fix any
parser bugs found, add mock test fixtures from captured tool output.

- [x] **NVIDIA H100 / A100** — mock fixtures for nvidia-smi 11-field CSV
  (A100 SXM 8-GPU, H100 SXM 2-GPU). Done in 1.1.0.
- [x] **NVIDIA Grace Hopper GH200** — mock fixture validates unified memory
  detection (96 GB HBM + 480 GB LPDDR5X). Done in 1.1.0.
- [x] **AMD MI300X** — Vulkan parser fixture for 192 GB HBM3. ROCm sysfs
  CXL testing via planning pipeline (8x 192 GB). Done in 1.1.0.
- [x] **Google TPU v5e / v5p** — planning pipeline fixtures for v5e 4-chip
  and v5p 256-chip pod slices. Done in 1.1.0.
- [x] **AWS Neuron (trn1/inf2)** — neuron-ls JSON fixtures for trn1.32xlarge
  (32 NeuronCores) and inf2.48xlarge (24 NeuronCores). Done in 1.1.0.
- [x] **Intel Gaudi 3** — hl-smi CSV fixture for 8-device Gaudi 3 (128 GB
  HBM2e each). Done in 1.1.0.
- [ ] **Live cloud validation** — run fixtures against real hardware to
  capture actual tool output and fix any parser discrepancies.

### Cross-platform porting

- [x] **macOS: `system_profiler -json` for Metal/ANE** — `parse_displays_json()`
  for SPDisplaysDataType JSON (Metal family, GPU cores, discrete VRAM).
  Enriches Metal profiles. `parse_sysctl_output()` for CPU topology.
  Done in 1.1.0.
- [x] **macOS: `sysctl` for CPU/memory** — parser for hw.memsize, hw.ncpu,
  hw.cpufrequency, perflevel cores. Done in 1.1.0.
- [x] **Windows: WMI queries for GPU detection** — `parse_wmic_output()` and
  `parse_powershell_csv()` with `nvidia-smi.exe` path resolution. Done in 1.1.0.
- [ ] **Windows: DirectX adapter enumeration** — DXGI `EnumAdapters1` via
  `windows-rs` for reliable GPU detection independent of vendor CLI tools.
  Returns adapter LUID, dedicated VRAM, shared memory, driver version.
- [x] **Cross-platform: `PlatformProbe` trait** — abstracts file reads,
  commands, device enumeration, memory. `LivePlatform` + `MockPlatform`.
  Done in 1.1.0.

---

## 0.25.3 — Ecosystem & Scale

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

## 0.26.3 — Fleet & Scale

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
- [x] **Make `ShardingPlan::shards` private** — `pub(crate)` with `shards()`
  accessor. Done in 1.0.0.
- [x] **`TryFrom<u32>` for `QuantizationLevel`** — map `32 → None`,
  `16 → Float16`, `8 → Int8`, `4 → Int4`. Done in 1.0.0.
- [x] **`AcceleratorType` is `Copy`** — moved `device_name` from `VulkanGpu`
  into `AcceleratorProfile`. Done in 1.0.0.
- [ ] **Fix `CloudGpuInstance` re-export alias** — blocked: two types named
  `CloudInstance` exist (`cost::CloudInstance` and `system_io::CloudInstance`).
  Needs rename of one type before alias can be removed.

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
