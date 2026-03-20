# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## Detection gaps (hardware-dependent)

Require real hardware or cloud instances. No deadline — addressed as access
becomes available.

### Accuracy improvements

- [ ] **Google TPU** — validate on real GCE VMs. Multi-host pod slices where
  chips span `/dev/accel*` nodes across hosts.
- [ ] **Intel Gaudi** — test on Gaudi 3. Parse firmware version from `hl-smi`.
- [ ] **AWS Neuron** — mixed Inferentia + Trainium in same instance. Test
  `neuron-ls` output format across SDK versions.
- [ ] **Intel oneAPI** — Data Center GPU Max (Ponte Vecchio) HBM vs DDR memory
  tiers. Test `xpu-smi` on real hardware.
- [ ] **AMD ROCm XGMI topology** — multi-GPU XGMI/Infinity Fabric detection.
  Needs multi-GPU AMD system.

### Untested backends (code written, needs hardware validation)

- [ ] **AMD MI300X / MI350** — CXL memory detection. Needs MI300X instance.
- [ ] **Cerebras WSE** — `cerebras_cli` + `/dev/cerebras*`. Needs hardware.
- [ ] **Graphcore IPU** — `gc-info` JSON parsing. Needs hardware.
- [ ] **Groq LPU** — `/dev/groq*`. Linux driver not yet public.
- [ ] **Samsung NPU** — sysfs detection. Needs Exynos device.
- [ ] **MediaTek APU** — sysfs detection. Needs MediaTek device.
- [ ] **NVIDIA Grace Hopper** — unified memory reporting. Needs GH200.

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

- [x] **Windows CI** — `windows-latest` runner in CI matrix runs unit tests,
  integration tests, doc tests, and feature-gated checks.
- [x] **Benchmark regression CI** — `github-action-benchmark` tracks numbers
  on main, alerts at 120% threshold, comments on PRs.
- [x] **Fuzz CI** — all 9 fuzz targets run for 30s each on every push/PR.
- [x] **Minimal feature testing** — CI tests `--no-default-features` and
  single-backend builds to catch feature gate issues.
- [x] **Release builds** — Windows (.zip), macOS ARM, Linux AMD64/ARM64
  binaries built and published on tag.

---

## Post-v1 features

Longer-term items that don't block any release.

- [ ] **Topology-aware sharding** — NVLink/NVSwitch/XGMI/ICI topology for
  optimal tensor placement and communication minimization.
- [ ] **Hot-plug support** — `udev`/`inotify` watch for device add/remove
  with dynamic registry updates.
- [ ] **Cost-aware planning** — cloud pricing data for cheapest device
  configuration per workload.
- [ ] **WASM target** — browser-based ML dashboards with stubbed detection.
- [ ] **Full Python bindings** — complete PyO3 package with pip install.
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
