# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## Phase 1: v1.0-rc (2026-03-20)

Final pre-release. Everything needed for a stable public API.

- [ ] Stable serde format: document the JSON schema, freeze `SCHEMA_VERSION`.
- [ ] `cargo publish --dry-run` passes (verified).
- [ ] Review all public API signatures for stability.
- [ ] Final CHANGELOG entry for v1.0-rc.

## Phase 2: v1.0 (2026-03-21)

First stable release. Published to crates.io.

- [ ] Publish to crates.io.
- [ ] Tag `1.0.0` (switch from CalVer to SemVer).
- [ ] Announce on relevant channels.

---

## Phase 3: Hardening (2026-03-22 -- 2026-03-24)

Post-v1 stabilization, one release per day.

### 2026-03-22: Testing depth

- [ ] Mock detection harness: simulated sysfs trees and CLI outputs via
  `tempdir` + symlinks for hardware-independent backend testing.
- [ ] Windows CI: add `x86_64-pc-windows-msvc` to the matrix.

### 2026-03-23: Supply chain + audit

- [ ] `cargo-vet`: add supply-chain audits for all dependencies.
- [ ] Minimal dependencies audit: evaluate whether `serde_json` can be
  feature-gated (only needed by `aws-neuron` backend and CLI binary).

### 2026-03-24: Performance tuning

- [ ] Caching: cache detection results for a configurable TTL so repeated
  `detect()` calls don't re-shell-out.
- [ ] Lazy detection: detect only the backends the caller queries.
- [ ] Reduce allocations: `Cow<str>` for device names, `SmallVec` for profiles.

---

## Phase 4: Ecosystem (2026-03-25 -- 2026-03-28)

### 2026-03-25: Integration guides

- [ ] Document usage with `candle`, `burn`, `tch-rs`, and `ort`.
- [ ] Expand examples with framework-specific code.

### 2026-03-26: Async API

- [ ] `tokio` feature flag: async `detect()` via `tokio::process`.
- [ ] Async builder: `AcceleratorRegistry::builder().detect_async().await`.

### 2026-03-27: CLI polish

- [ ] `--table` column sorting and filtering flags.
- [ ] `--watch` mode: re-detect on interval, show diffs.

### 2026-03-28: Language bindings

- [ ] C FFI: `ai_hwaccel_detect()`, `ai_hwaccel_best_available()`, etc.
- [ ] Python bindings scaffold via PyO3.

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

---

## Non-goals

- **Runtime execution** -- detection and planning only, not inference/training.
- **Kernel driver management** -- no installing or configuring drivers.
- **Cloud provisioning** -- detect what's present, not what could be spun up.
