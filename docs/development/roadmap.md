# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## Phase 4: Ecosystem (2026-03-20)

### Integration guides

- [ ] Document usage with `candle`, `burn`, `tch-rs`, and `ort`.
- [ ] Expand examples with framework-specific code.

### Async API

- [ ] `tokio` feature flag: async `detect()` via `tokio::process`.
- [ ] Async builder: `AcceleratorRegistry::builder().detect_async().await`.

### CLI polish

- [ ] `--table` column sorting and filtering flags.
- [ ] `--watch` mode: re-detect on interval, show diffs.

### Language bindings

- [ ] C FFI: `ai_hwaccel_detect()`, `ai_hwaccel_best_available()`, etc.
- [ ] Python bindings scaffold via PyO3.

---

## Remaining pre-v1

- [ ] Switch from CalVer to SemVer (`1.0.0`).
- [ ] Publish to crates.io.
- [ ] `cargo-vet`: supply-chain audits for all dependencies.

---

## Hardware-dependent (no deadline -- based on access)

These items require real hardware or cloud instances. They will be addressed
as hardware becomes available, in any order.

### Detection accuracy

- [ ] **AMD ROCm**: query `rocm-smi` for clock speeds, firmware, XGMI topology.
- [ ] **Google TPU**: validate on real GCE VMs; multi-host pod slices.
- [ ] **Intel Gaudi**: test on Gaudi 3 hardware; parse firmware version.
- [ ] **AWS Neuron**: mixed Inferentia + Trainium in the same instance.
- [ ] **Intel oneAPI**: Data Center GPU Max HBM vs DDR memory tiers.
- [ ] **Vulkan**: compute queue family parsing from `vulkaninfo`.

### New backends

- [ ] **AMD MI300X** -- CXL-attached memory, unified HBM pool.
- [ ] **Cerebras WSE** -- `/dev/cerebras*` or SDK introspection.
- [ ] **Graphcore IPU** -- `gc-info` CLI parsing.
- [ ] **Groq LPU** -- when Linux driver is publicly available.
- [ ] **Samsung NPU** -- Exynos AI accelerator sysfs.
- [ ] **MediaTek APU** -- Android/Linux APU sysfs.

### Platform support

- [ ] **macOS ANE via IOKit** -- native ANE detection beyond `system_profiler`.
- [ ] **Windows** -- WMI queries, DirectML enumeration.
- [ ] **Android** -- HAL `hwbinder` for NNAPI accelerator list.
- [ ] **FreeBSD** -- DRM sysctl equivalents.

---

## Post-v1 features (no deadline)

These are longer-term items that don't block any release.

- [ ] **Topology-aware sharding** -- NVLink/NVSwitch/XGMI/ICI topology for
  optimal tensor placement.
- [ ] **NVLink/NVSwitch detection** -- parse `nvidia-smi topo -m`.
- [ ] **Power and thermal monitoring** -- current power draw, temperature,
  throttling state.
- [ ] **Hot-plug support** -- `udev`/`inotify` watch for device add/remove.
- [ ] **Cost-aware planning** -- cloud pricing data for cheapest config.
- [ ] **WASM target** -- browser-based ML dashboards with stubbed detection.
- [ ] **Python bindings** -- full PyO3 package for the ML ecosystem.
- [ ] **Reduce allocations** -- `Cow<str>` for device names, `SmallVec` for
  profiles.
- [ ] **Lazy detection** -- detect only backends the caller queries.

---

## Non-goals

- **Runtime execution** -- detection and planning only, not inference/training.
- **Kernel driver management** -- no installing or configuring drivers.
- **Cloud provisioning** -- detect what's present, not what could be spun up.
