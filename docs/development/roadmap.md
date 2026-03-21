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

- [ ] **Lazy detection** — detect only backends the caller queries, not all
  enabled backends upfront. `AcceleratorRegistry::lazy()` returns a registry
  that probes on first access per family. Avoids spawning nvidia-smi when
  caller only needs TPU info.
- [ ] **`vulkaninfo` timeout + caching** — `vulkaninfo` is the single
  slowest probe (~5s). Add a per-process cache: write parsed results to
  `$XDG_CACHE_HOME/ai-hwaccel/vulkan.json` with a 60s TTL. On subsequent
  calls within the TTL, read from cache instead of re-running vulkaninfo.
  Also add a 3s timeout on the subprocess — if vulkaninfo hangs, fall back
  to sysfs-only Vulkan detection (`/sys/class/drm/card*/device/vendor`).
- [ ] **Parallel backend probing** — run CLI-based backends (nvidia-smi,
  vulkaninfo, rocm-smi, hl-smi) concurrently via `tokio::process::Command`
  instead of sequentially. Sysfs-only backends already complete in <1ms
  and don't benefit from parallelism. Expected improvement: detection time
  drops from max(all tools) to max(slowest tool) — on systems with both
  CUDA and Vulkan, this halves detection time.
- [ ] **Sysfs-only Vulkan fallback** — for systems where `vulkaninfo` is
  slow or absent, detect Vulkan-capable GPUs via
  `/sys/class/drm/card*/device/{vendor,device}` + PCI ID lookup table.
  Provides device name and VRAM estimate without spawning a subprocess.
  Use as fast path; full `vulkaninfo` becomes opt-in enrichment.
- [ ] **Detection result caching** — `AcceleratorRegistry::cached(ttl)`
  persists detection results to disk. Subsequent calls within TTL return
  cached data instantly. Useful for gateway servers (like hoosh) that
  call `detect()` at startup and don't need real-time hardware changes.
- [ ] **Per-backend timing** — `--profile` CLI flag and
  `AcceleratorRegistry::detect_with_timing()` API that returns
  `HashMap<&str, Duration>` showing how long each backend took. Enables
  users to identify and disable slow backends.
- [ ] **Topology-aware sharding** — use interconnect data (NVLink, XGMI, ICI)
  already collected in `SystemIo` to generate sharding plans that minimize
  cross-link transfers. Pipeline parallel should prefer directly-connected
  GPU pairs. Tensor parallel should prefer NVSwitch-connected groups.
- [ ] **Cost-aware planning** — static pricing table for common cloud GPU
  instances (A100, H100, L4, T4, MI300X, TPU v5e). Given model size +
  quantisation, recommend cheapest viable config. Data in
  `data/cloud_pricing.json`, updatable without recompiling.
- [ ] **Container/VM detection** — detect Docker (`/.dockerenv`),
  Kubernetes (`/var/run/secrets/kubernetes.io`), cloud instance type
  (AWS `instance-identity`, GCE `metadata.google.internal`, Azure IMDS).
  Expose as `SystemIo::environment` for deployment-aware planning.

### Python bindings (groundwork)

- [ ] **PyO3 module scaffold** — `py/` directory with `maturin` build,
  wrapping `AcceleratorRegistry::detect()`, `suggest_quantization()`,
  `plan_sharding()`, `system_io()`. Ship as `ai-hwaccel` on PyPI.
- [ ] **Python type stubs** — `.pyi` files for IDE support.
- [ ] **Python examples** — basic detection, sharding plan, training memory
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

### Platform gaps

- [ ] **Windows GPU detection** — WMI queries via `wmic` for GPU enumeration
  when nvidia-smi is absent. DirectML device listing via `dxdiag` parsing.
  `nvidia-smi.exe` path resolution (`C:\Windows\System32\`).
- [ ] **macOS ANE via IOKit** — native ANE core count and performance tier
  via `IORegistryEntryCreateCFProperties`. Replace estimated ANE memory
  with actual hardware values.

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

### Remaining platform gaps

- [ ] **Android** — HAL `hwbinder` for NNAPI accelerator list. Requires
  Android NDK cross-compilation.
- [ ] **FreeBSD** — DRM sysctl equivalents for GPU detection.
- [ ] **Intel oneAPI** — Data Center GPU Max (Ponte Vecchio) HBM vs DDR
  memory tiers on real hardware.
- [ ] **AMD ROCm XGMI** — multi-GPU XGMI/Infinity Fabric topology detection
  on multi-GPU AMD systems.

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

## Non-goals

- **Runtime execution** — detection and planning only, not inference/training.
- **Kernel driver management** — no installing or configuring drivers.
- **Cloud provisioning** — detect what's present, not what could be spun up.
