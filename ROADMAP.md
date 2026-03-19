# Roadmap to v1.0

This document outlines the gaps, improvements, and milestones needed to take
`ai-hwaccel` from a working prototype to a production-grade public crate.

---

## Milestone 1: Hardened detection (pre-v1 prerequisite)

### Detection accuracy

- [ ] **NVIDIA**: parse driver version from `nvidia-smi`, detect multi-GPU
  NVLink/NVSwitch topology for smarter sharding decisions.
- [ ] **AMD ROCm**: query `rocm-smi` for clock speeds, firmware version, and
  XGMI topology when available.
- [ ] **Apple Metal/ANE**: use `system_profiler` or `IOKit` on macOS instead of
  `/proc/device-tree` (Linux-only today). Detect unified memory size per
  chip (M1 8 GB vs M4 Max 128 GB).
- [ ] **Vulkan**: actually parse `vulkaninfo` output for device name, memory
  heaps, and compute queue families instead of registering a generic device.
- [ ] **Google TPU**: validate detection on real GCE VMs; handle multi-host pod
  slices where chips span multiple `/dev/accel*` nodes.
- [ ] **Intel Gaudi**: test on real Gaudi 3 hardware; parse firmware version.
- [ ] **AWS Neuron**: handle mixed Inferentia + Trainium in the same instance.
- [ ] **Intel oneAPI**: detect Intel Data Center GPU Max (Ponte Vecchio) memory
  tiers (HBM vs DDR).

### New backends

- [ ] **AMD MI300X** -- detect CXL-attached memory and unified HBM pool.
- [ ] **Cerebras WSE** -- `/dev/cerebras*` or Cerebras SDK introspection.
- [ ] **Graphcore IPU** -- `gc-info` CLI parsing.
- [ ] **Groq LPU** -- when Linux driver becomes publicly available.
- [ ] **Samsung NPU** -- Exynos AI accelerator sysfs paths.
- [ ] **MediaTek APU** -- Android/Linux APU sysfs detection.

### Platform support

- [ ] **macOS native**: Metal detection via `objc` / `metal-rs` bindings rather
  than device-tree. ANE detection via IOKit.
- [ ] **Windows**: WMI queries for GPU detection, DirectML device enumeration.
- [ ] **Android**: HAL `hwbinder` queries for NNAPI accelerator list.
- [ ] **FreeBSD**: DRM sysctl equivalents.

---

## Milestone 2: API stability and ergonomics

### API design

- [ ] **Error types**: replace silent fallbacks with a proper `DetectionError`
  enum. Callers should be able to distinguish "tool not found" from "tool
  crashed" from "parse failure". Detection remains best-effort but errors
  are queryable.
- [ ] **Builder pattern for registry**: allow callers to opt in/out of specific
  backends (`AcceleratorRegistry::builder().with_cuda().without_vulkan().detect()`).
- [ ] **Async detection**: some CLI tools (`nvidia-smi`, `neuron-ls`) take
  hundreds of milliseconds. Offer an async `detect()` variant that runs
  them concurrently via `tokio::process` behind a feature flag.
- [ ] **Feature flags**: gate each backend behind a cargo feature so
  downstream crates can slim their dependency tree
  (e.g. `features = ["cuda", "tpu"]`).
- [ ] **`#[non_exhaustive]`**: mark `AcceleratorType`, `AcceleratorFamily`,
  `QuantizationLevel`, and `AcceleratorRequirement` as `#[non_exhaustive]`
  so new variants can be added without breaking downstream `match` arms.
- [ ] **Stable serde format**: define and document the JSON schema so external
  tools can consume serialized registries. Consider a schema version field.

### Ergonomics

- [ ] **Pretty-print CLI output**: `--pretty` flag for human-readable formatted
  JSON, `--table` for tabular output.
- [ ] **`Display` for `ShardingPlan`**: human-readable multi-line plan summary.
- [ ] **Convenience constructors**: `AcceleratorProfile::cuda(device_id, vram)`
  etc. for test/mock setups.

---

## Milestone 3: Performance

- [ ] **Parallel detection**: run all detectors concurrently (they are
  independent I/O operations). Use `std::thread::scope` for the sync API
  and `tokio::join!` for the async variant.
- [ ] **Caching**: cache detection results for a configurable TTL so repeated
  calls to `detect()` don't re-shell-out. Invalidate on hot-plug events
  if a filesystem watch is available.
- [ ] **Lazy detection**: detect only the backends the caller queries
  (e.g. don't run `nvidia-smi` if the caller only asks for TPUs).
- [ ] **Reduce allocations**: return `&str` or `Cow<str>` for device names
  instead of owned `String` where possible. Use `SmallVec` for profiles
  (most systems have < 8 accelerators).

---

## Milestone 4: Security and robustness

### Command execution

- [ ] **Absolute paths**: resolve tool paths at detection time and invoke them
  by absolute path to prevent `$PATH` hijacking.
- [ ] **Timeout enforcement**: enforce a configurable timeout (default 5 s) on
  all subprocess calls. A hung `nvidia-smi` should not block the caller
  indefinitely.
- [ ] **Output size limits**: cap the bytes read from subprocess stdout to
  prevent memory exhaustion from malicious or misbehaving tools.
- [ ] **Input validation**: validate and sanitize all parsed values from CLI
  output and sysfs (e.g. memory sizes, device IDs) before using them.
- [ ] **Sandboxing**: document threat model. Consider `seccomp` or
  `landlock` hints for callers running in restricted environments.

### Serde safety

- [ ] **Deserialization limits**: use `serde`'s `#[serde(deny_unknown_fields)]`
  and add size limits when deserializing untrusted `AcceleratorRegistry`
  JSON to prevent DoS via deeply nested or excessively large payloads.

### Supply chain

- [ ] **`cargo-vet`**: add supply-chain audits for all dependencies.
- [ ] **`cargo-deny`**: configure license and advisory checks in CI.
- [ ] **Minimal dependencies audit**: review whether `serde_json` can be
  dev-only (move CLI JSON output behind a feature flag).

---

## Milestone 5: Testing and CI

- [ ] **Integration tests**: add a `tests/` directory (crate-level) with
  integration tests that exercise the public API end-to-end.
- [ ] **Mock detection**: create a test harness that simulates sysfs trees and
  CLI tool outputs so detectors can be unit-tested without real hardware.
  Use `tempdir` + symlinks to build fake `/sys/class/drm/card0/...` trees.
- [ ] **Property-based testing**: use `proptest` or `quickcheck` to fuzz
  `estimate_memory`, `plan_sharding`, and `suggest_quantization` across
  random parameter counts and device configurations.
- [ ] **Benchmark suite**: add `criterion` benchmarks for `detect()`,
  `plan_sharding()`, and `estimate_training_memory()`.
- [ ] **CI matrix**: test on `x86_64-unknown-linux-gnu`,
  `aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`,
  `aarch64-apple-darwin`, and `x86_64-pc-windows-msvc`.
- [ ] **MSRV CI job**: dedicated CI job that builds with the declared MSRV
  to prevent accidental breakage.
- [ ] **Coverage**: add `cargo-llvm-cov` or `tarpaulin` to CI and track
  coverage over time.

---

## Milestone 6: Documentation and ecosystem

- [ ] **Rustdoc examples**: add `# Examples` sections to every public function
  and type. Ensure all examples compile as doc-tests.
- [ ] **Crate-level guide**: expand `lib.rs` docs with a guided walkthrough
  covering detection → querying → planning → training estimation.
- [ ] **Architecture decision records (ADRs)**: document key design choices
  (e.g. "why sysfs over vendor SDKs", "why calendar versioning").
- [ ] **`examples/` directory**: add runnable examples:
  - `detect.rs` -- print detected hardware.
  - `plan.rs` -- plan sharding for a user-specified model.
  - `training.rs` -- estimate training memory.
  - `json_output.rs` -- serialize and deserialize a registry.
- [ ] **crates.io publishing**: publish to crates.io once API is stable.
  Ensure `cargo publish --dry-run` passes.
- [ ] **Ecosystem integration guides**: document how to use `ai-hwaccel` with
  `candle`, `burn`, `tch-rs`, and `ort` (ONNX Runtime).

---

## Milestone 7: Advanced features (post-v1)

- [ ] **Topology-aware sharding**: use NVLink/NVSwitch/XGMI/ICI topology to
  minimize inter-device communication in sharding plans.
- [ ] **Power and thermal monitoring**: expose current power draw, temperature,
  and throttling state via `nvidia-smi`, `rocm-smi`, etc.
- [ ] **Hot-plug support**: watch for device addition/removal via `udev` or
  `inotify` and update the registry dynamically.
- [ ] **Cost-aware planning**: integrate cloud pricing data to suggest the
  cheapest device configuration for a given workload.
- [ ] **WASM target**: compile detection logic to WASM for use in browser-based
  ML dashboards (with stubbed-out sysfs/CLI paths).
- [ ] **C FFI**: expose a C-compatible API (`ai_hwaccel_detect()`, etc.) for
  use from Python, Go, and other languages.
- [ ] **Python bindings**: `PyO3`-based Python package for the ML ecosystem.

---

## Non-goals

These are explicitly out of scope for this crate:

- **Runtime execution**: this crate detects and plans, it does not execute
  inference or training. That is the job of frameworks like `candle`, `burn`,
  or vendor SDKs.
- **Kernel driver management**: installing, loading, or configuring drivers is
  outside scope.
- **Cloud instance provisioning**: detecting what is on the machine, not what
  could be provisioned.

---

## How to help

Pick any unchecked item above, open an issue to signal intent, and submit a PR.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the process. Items in Milestones 1--2
are the highest priority for reaching v1.0.
