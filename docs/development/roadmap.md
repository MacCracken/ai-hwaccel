# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## 0.21.3 — Platform & Planning

Focus: lazy detection, topology-aware sharding, cloud hardware validation,
Python bindings groundwork.

### API improvements

- [ ] **Lazy detection** — detect only backends the caller queries, not all
  enabled backends upfront. `AcceleratorRegistry::lazy()` returns a registry
  that probes on first access per family. Avoids spawning nvidia-smi when
  caller only needs TPU info.
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

## Future (post-v1)

Items that don't fit in a specific release yet.

- [ ] **Model compatibility database** — which models run on which hardware
  at which quantisation. Queryable: "can I run Llama 70B on 2x RTX 4090?"
- [ ] **Workload profiler** — measure actual detection latency per backend,
  report slow tools. `--profile` flag on CLI.
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
