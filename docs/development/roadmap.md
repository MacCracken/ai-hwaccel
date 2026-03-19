# Roadmap to v1.0

This document tracks remaining work to take `ai-hwaccel` from a working
prototype to a production-grade public crate. Completed items have been removed;
see [CHANGELOG.md](../../CHANGELOG.md) for what shipped.

---

## Detection accuracy

- [ ] **AMD ROCm**: query `rocm-smi` for clock speeds, firmware version, and
  XGMI topology when available.
- [ ] **Google TPU**: validate detection on real GCE VMs; handle multi-host pod
  slices where chips span multiple `/dev/accel*` nodes.
- [ ] **Intel Gaudi**: test on real Gaudi 3 hardware; parse firmware version.
- [ ] **AWS Neuron**: handle mixed Inferentia + Trainium in the same instance.
- [ ] **Intel oneAPI**: detect Intel Data Center GPU Max (Ponte Vecchio) memory
  tiers (HBM vs DDR).
- [ ] **Vulkan**: parse compute queue families from `vulkaninfo`.

## New backends

- [ ] **AMD MI300X** -- detect CXL-attached memory and unified HBM pool.
- [ ] **Cerebras WSE** -- `/dev/cerebras*` or Cerebras SDK introspection.
- [ ] **Graphcore IPU** -- `gc-info` CLI parsing.
- [ ] **Groq LPU** -- when Linux driver becomes publicly available.
- [ ] **Samsung NPU** -- Exynos AI accelerator sysfs paths.
- [ ] **MediaTek APU** -- Android/Linux APU sysfs detection.

## Platform support

- [ ] **macOS ANE via IOKit** -- native ANE detection (current `system_profiler`
  path covers Metal GPU and unified memory, but not ANE specifically).
- [ ] **Windows** -- WMI queries for GPU detection, DirectML device enumeration.
- [ ] **Android** -- HAL `hwbinder` queries for NNAPI accelerator list.
- [ ] **FreeBSD** -- DRM sysctl equivalents.

---

## API stability

- [ ] **Async detection** -- offer an async `detect()` variant that runs CLI
  tools concurrently via `tokio::process` behind a feature flag.
- [ ] **Feature flags** -- gate each backend behind a cargo feature so
  downstream crates can slim their dependency tree
  (e.g. `features = ["cuda", "tpu"]`).
- [ ] **Stable serde format** -- define and document the JSON schema so external
  tools can consume serialized registries. Consider a schema version field.
- [ ] **`--table` CLI output** -- human-readable tabular device listing.

---

## Performance

- [ ] **Caching** -- cache detection results for a configurable TTL so repeated
  calls to `detect()` don't re-shell-out. Invalidate on hot-plug events if
  a filesystem watch is available.
- [ ] **Lazy detection** -- detect only the backends the caller queries (e.g.
  don't run `nvidia-smi` if the caller only asks for TPUs).
- [ ] **Reduce allocations** -- return `&str` or `Cow<str>` for device names
  instead of owned `String` where possible. Use `SmallVec` for profiles
  (most systems have < 8 accelerators).

---

## Security

- [ ] **`cargo-vet`** -- add supply-chain audits for all dependencies.
- [ ] **Minimal dependencies audit** -- review whether `serde_json` can be
  dev-only (move CLI JSON output behind a feature flag).

---

## Testing

- [ ] **Mock detection** -- create a test harness that simulates sysfs trees and
  CLI tool outputs so detectors can be unit-tested without real hardware.
  Use `tempdir` + symlinks to build fake `/sys/class/drm/card0/...` trees.
- [ ] **Property-based testing** -- use `proptest` or `quickcheck` to fuzz
  `estimate_memory`, `plan_sharding`, and `suggest_quantization` across
  random parameter counts and device configurations.
- [ ] **Windows CI** -- add `x86_64-pc-windows-msvc` to the CI matrix.

---

## Documentation

- [ ] **Crate-level guide** -- expand `lib.rs` docs with a guided walkthrough
  covering detection -> querying -> planning -> training estimation.
- [ ] **Architecture decision records (ADRs)** -- document key design choices
  (e.g. "why sysfs over vendor SDKs", "why calendar versioning").
- [ ] **crates.io publishing** -- publish to crates.io once API is stable.
  Ensure `cargo publish --dry-run` passes.
- [ ] **Ecosystem integration guides** -- document how to use `ai-hwaccel` with
  `candle`, `burn`, `tch-rs`, and `ort` (ONNX Runtime).

---

## Advanced features (post-v1)

- [ ] **Topology-aware sharding** -- use NVLink/NVSwitch/XGMI/ICI topology to
  minimize inter-device communication in sharding plans.
- [ ] **NVLink/NVSwitch detection** -- parse `nvidia-smi topo -m` for GPU
  interconnect topology.
- [ ] **Power and thermal monitoring** -- expose current power draw,
  temperature, and throttling state via `nvidia-smi`, `rocm-smi`, etc.
- [ ] **Hot-plug support** -- watch for device addition/removal via `udev` or
  `inotify` and update the registry dynamically.
- [ ] **Cost-aware planning** -- integrate cloud pricing data to suggest the
  cheapest device configuration for a given workload.
- [ ] **WASM target** -- compile detection logic to WASM for use in
  browser-based ML dashboards (with stubbed-out sysfs/CLI paths).
- [ ] **C FFI** -- expose a C-compatible API (`ai_hwaccel_detect()`, etc.) for
  use from Python, Go, and other languages.
- [ ] **Python bindings** -- `PyO3`-based Python package for the ML ecosystem.

---

## Non-goals

These are explicitly out of scope for this crate:

- **Runtime execution** -- this crate detects and plans, it does not execute
  inference or training. That is the job of frameworks like `candle`, `burn`,
  or vendor SDKs.
- **Kernel driver management** -- installing, loading, or configuring drivers is
  outside scope.
- **Cloud instance provisioning** -- detecting what is on the machine, not what
  could be provisioned.

---

## How to help

Pick any item above, open an issue to signal intent, and submit a PR.
See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the process.
