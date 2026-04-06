# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## 1.3.0 — Platform Validation

Live cloud hardware validation and remaining cross-platform gaps.

### Live cloud validation

- [ ] **NVIDIA H100 / A100** — capture real nvidia-smi output (AWS/GCP)
- [ ] **NVIDIA Grace Hopper GH200** — validate unified memory on real GH200
- [ ] **AMD MI300X** — validate CXL memory, ROCm sysfs (Azure)
- [ ] **Google TPU v5e / v5p** — multi-host pod slice testing (GCE)
- [ ] **AWS Neuron (trn1/inf2)** — validate on mixed instances
- [ ] **Intel Gaudi 3** — validate on AWS DL1/DL2

### Cross-platform

- [ ] **Windows: DirectX adapter enumeration** — DXGI `EnumAdapters1` via
  `windows-rs`. Returns adapter LUID, dedicated VRAM, shared memory,
  driver version.

### Untested backends

- [ ] **Cerebras WSE** — needs Cerebras Cloud access
- [ ] **Graphcore IPU** — needs Paperspace or IPU cloud access
- [ ] **Groq LPU** — blocked on public Linux driver
- [ ] **Samsung NPU** — needs Exynos device (Galaxy S24+)
- [ ] **MediaTek APU** — needs Dimensity device

---

## 1.4.0 — Ecosystem

Python bindings, WASM, JS/TS.

### Python bindings

- [ ] **Complete PyO3 API** — all public types: `AcceleratorProfile`,
  `SystemIo`, `Interconnect`, `StorageDevice`, `ShardingPlan`,
  `TrainingMethod`
- [ ] **`pip install ai-hwaccel`** — wheels for Linux (manylinux), macOS
  (universal2), Windows (x86_64) via maturin + GitHub Actions
- [ ] **Python-native features** — dict-like objects, JSON serialization,
  pandas DataFrame export

### WASM / JS

- [ ] **JS/TS bindings** — `wasm-bindgen` wrapper for npm package

---

## 1.5.0 — Multi-Node & Hot-Plug

### Multi-node detection

- [ ] **SSH probe** — `AcceleratorRegistry::detect_remote(hosts)`,
  merge into cluster-wide registry
- [ ] **Cluster-aware sharding** — distribute across nodes, consider
  IB/RoCE bandwidth for pipeline vs data parallelism

### Hot-plug support

- [ ] **`udev` watcher (Linux)** — `AcceleratorRegistry::watch()` returns
  `DeviceEvent::Added` / `DeviceEvent::Removed` stream
- [ ] **Dynamic registry updates** — `CachedRegistry` auto-invalidates
  on hot-plug events

### Remaining platforms

- [ ] **Android** — HAL `hwbinder` for NNAPI accelerator list
- [ ] **FreeBSD** — DRM sysctl equivalents for GPU detection
- [ ] **Intel oneAPI** — DC GPU Max HBM vs DDR tiers on real hardware

---

## 2.0.0 — Fleet & Scale

Fleet-wide inventory, health monitoring, capacity planning at datacenter
scale.

### Fleet inventory

- [ ] **Fleet registry** — `FleetRegistry` aggregates registries from
  multiple nodes. Tracks hostname, IP, rack, zone
- [ ] **Discovery protocols** — mDNS/DNS-SD, Consul, Kubernetes node labels
- [ ] **Inventory persistence** — SQLite or JSON lines, diff against
  previous scan
- [ ] **Fleet CLI** — `ai-hwaccel fleet scan --subnet 10.0.0.0/24`

### Health monitoring

- [ ] **Continuous health checks** — periodic re-detection, trend tracking
- [ ] **Alert rules engine** — configurable thresholds, webhook/Slack/PagerDuty
- [ ] **Dead GPU detection** — flag nodes with missing accelerators
- [ ] **ECC error tracking** — nvidia-smi ECC, ROCm ras_features

### Capacity planning

- [ ] **Fleet-wide sharding** — recommend nodes and distribution strategy
- [ ] **Bin packing** — optimal multi-model placement across fleet
- [ ] **Scaling recommendations** — "you need 3 more H100 nodes for 405B"

### Observability & export

- [ ] **Prometheus metrics** — per-device gauges, fleet aggregates
- [ ] **OpenTelemetry spans** — instrument detection with OTel traces
- [ ] **Grafana dashboard template** — fleet GPU utilization heatmap
- [ ] **Structured event log** — JSON lines for ELK/Loki/Datadog

### Multi-tenancy

- [ ] **Device reservation** — `registry.reserve(device_id, owner)`
- [ ] **Namespace isolation** — Kubernetes pod-scoped detection
- [ ] **Quota management** — per-team GPU hour budgets

---

## Future

- [ ] **Power budget planning** — recommend device mix for power cap
- [ ] **Thermal throttling prediction** — warn on approaching thresholds
- [ ] **Plugin system** — third-party backends via dynamic loading

---

## Non-goals

- **Runtime execution** — detection and planning only, not inference/training
- **Kernel driver management** — no installing or configuring drivers
- **Cloud provisioning** — detect what's present, not what could be spun up
