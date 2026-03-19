# Roadmap to v1.0

This document outlines the gaps, improvements, and milestones needed to take
`ai-hwaccel` from a working prototype to a production-grade public crate.

---

## Milestone 1: Hardened detection (pre-v1 prerequisite)

### Detection accuracy

- [x] **NVIDIA**: parse driver version from `nvidia-smi`. Structured error
  reporting on tool failure. ~~detect multi-GPU NVLink/NVSwitch topology for
  smarter sharding decisions~~ (deferred to Milestone 7).
- [ ] **AMD ROCm**: query `rocm-smi` for clock speeds, firmware version, and
  XGMI topology when available.
- [x] **Apple Metal/ANE**: macOS detection via `system_profiler
  SPHardwareDataType` for chip name and unified memory. ANE memory estimate
  varies by generation. Linux Asahi device-tree preserved as fallback.
- [x] **Vulkan**: parse `vulkaninfo --summary` for device names, memory heap
  sizes, API version, and driver version. ~~compute queue families~~ (deferred).
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

- [x] **macOS native**: Metal detection via `system_profiler` (chip name,
  unified memory). ~~`objc` / `metal-rs` bindings~~ deferred to post-v1.
  ANE detection via IOKit still TODO.
- [ ] **Windows**: WMI queries for GPU detection, DirectML device enumeration.
- [ ] **Android**: HAL `hwbinder` queries for NNAPI accelerator list.
- [ ] **FreeBSD**: DRM sysctl equivalents.

---

## Milestone 2: API stability and ergonomics

### API design

- [x] **Error types**: `DetectionError` enum with `ToolNotFound`, `ToolFailed`,
  `ParseError`, `SysfsReadError` variants. Collected as non-fatal warnings in
  `AcceleratorRegistry::warnings()`.
- [x] **Builder pattern for registry**: `AcceleratorRegistry::builder()` with
  `with_*()` / `without_*()` methods per backend, plus `DetectBuilder::none()`
  for opt-in-only detection.
- [ ] **Async detection**: some CLI tools (`nvidia-smi`, `neuron-ls`) take
  hundreds of milliseconds. Offer an async `detect()` variant that runs
  them concurrently via `tokio::process` behind a feature flag.
- [ ] **Feature flags**: gate each backend behind a cargo feature so
  downstream crates can slim their dependency tree
  (e.g. `features = ["cuda", "tpu"]`).
- [x] **`#[non_exhaustive]`**: applied to `AcceleratorType`, `AcceleratorFamily`,
  `QuantizationLevel`, `AcceleratorRequirement`, and `DetectionError`.
- [ ] **Stable serde format**: define and document the JSON schema so external
  tools can consume serialized registries. Consider a schema version field.

### Ergonomics

- [x] **Pretty-print CLI output**: `--pretty` / `-p` flag for formatted JSON.
  `--table` for tabular output still TODO.
- [x] **`Display` for `ShardingPlan`**: multi-line summary with strategy,
  memory, throughput, and per-shard device assignments.
- [x] **Convenience constructors**: `AcceleratorProfile::cuda()`, `rocm()`,
  `tpu()`, `gaudi()`, `cpu()` for test/mock setups.

---

## Milestone 3: Performance

- [x] **Parallel detection**: all backends run concurrently via
  `std::thread::scope`. Vulkan deduplication moved to a post-pass so
  ordering dependencies don't block parallelism.
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

- [x] **Absolute paths**: all CLI tools resolved to absolute path via `which()`
  before invocation. Implemented in `detect/command.rs`.
- [x] **Timeout enforcement**: 5-second default timeout on all subprocess calls
  via `try_wait()` loop + `child.kill()`. Configurable via `DEFAULT_TIMEOUT`.
- [x] **Output size limits**: stdout capped at 1 MiB (`MAX_STDOUT_BYTES`),
  stderr at 4 KiB. Reads beyond the limit are silently discarded.
- [x] **Input validation**: `validate_device_id()` (0--1024) and
  `validate_memory_mb()` (0--16 TiB) applied to all parsed CLI output.
- [x] **Sandboxing**: threat model documented in
  `docs/development/threat-model.md`. `seccomp`/`landlock` guidance for
  callers in restricted environments.

### Serde safety

- [x] **Deserialization limits**: `#[serde(deny_unknown_fields)]` on
  `AcceleratorRegistry`, `AcceleratorProfile`, `ModelShard`, `ShardingPlan`.
  Size limits are caller responsibility (documented in threat model).

### Supply chain

- [ ] **`cargo-vet`**: add supply-chain audits for all dependencies.
- [x] **`cargo-deny`**: `deny.toml` configured with license allowlist,
  advisory checks, source restrictions. Added `make deny` target.
- [ ] **Minimal dependencies audit**: review whether `serde_json` can be
  dev-only (move CLI JSON output behind a feature flag).

---

## Milestone 5: Testing and CI

- [x] **Integration tests**: `tests/integration.rs` with 9 tests covering
  detect→query→plan pipeline, builder, JSON roundtrip, manual registry,
  training estimation, Display impls, and warnings.
- [ ] **Mock detection**: create a test harness that simulates sysfs trees and
  CLI tool outputs so detectors can be unit-tested without real hardware.
  Use `tempdir` + symlinks to build fake `/sys/class/drm/card0/...` trees.
- [ ] **Property-based testing**: use `proptest` or `quickcheck` to fuzz
  `estimate_memory`, `plan_sharding`, and `suggest_quantization` across
  random parameter counts and device configurations.
- [x] **Benchmark suite**: `criterion` benchmarks in `benches/` for
  `detect()`, `detect_none()`, `plan_sharding()`, `suggest_quantization()`,
  `estimate_memory()`, and `estimate_training_memory()`.
- [x] **CI matrix**: test on `ubuntu-latest` (x86_64-linux) and
  `macos-latest` (aarch64-apple-darwin). Windows deferred.
- [x] **MSRV CI job**: dedicated job building and testing with Rust 1.89.
- [x] **Coverage**: `cargo-llvm-cov` in CI with Codecov upload on main.

---

## Milestone 6: Documentation and ecosystem

- [x] **Rustdoc examples**: `# Examples` sections on `AcceleratorRegistry`,
  `AcceleratorProfile`, `QuantizationLevel`, `DetectionError`, and
  `estimate_training_memory`. All compile as doc-tests (7 total).
- [ ] **Crate-level guide**: expand `lib.rs` docs with a guided walkthrough
  covering detection → querying → planning → training estimation.
- [ ] **Architecture decision records (ADRs)**: document key design choices
  (e.g. "why sysfs over vendor SDKs", "why calendar versioning").
- [x] **`examples/` directory**: four runnable examples:
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
