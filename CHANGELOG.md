# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [semantic versioning](https://semver.org/) as of v0.19.3.

## [0.20.3] - Unreleased

### Added

#### System I/O and monitoring

- **VRAM bandwidth probing**: `AcceleratorProfile::memory_bandwidth_gbps`
  calculates theoretical memory throughput from clock speed and bus width.
  NVIDIA via `nvidia-smi --query-gpu=clocks.max.memory` + compute capability
  lookup; AMD via sysfs `pp_dpm_mclk` + PCI device ID lookup. Includes
  fallback tables for known GPU specs.
- **Runtime VRAM usage**: `memory_used_bytes` and `memory_free_bytes` for
  CUDA (via `nvidia-smi`) and ROCm (via sysfs).
- **PCIe link detection**: `pcie_bandwidth_gbps` reads sysfs
  `current_link_width`/`current_link_speed` for CUDA and ROCm GPUs.
- **NUMA topology**: `numa_node` maps GPUs to their NUMA node via sysfs PCI
  device info.
- **Power and thermal monitoring**: `temperature_c`, `power_watts`,
  `gpu_utilization_percent` on `AcceleratorProfile`. CUDA via `nvidia-smi`
  (`temperature.gpu`, `power.draw`, `utilization.gpu`). ROCm via sysfs hwmon
  (`temp1_input`, `power1_average`, `gpu_busy_percent`).
- **Network interconnect detection**: `SystemIo::interconnects` detects
  InfiniBand and RoCE via `/sys/class/infiniband/`, NVLink via `nvidia-smi
  nvlink -s`. Exposes bandwidth and link state.
- **Disk I/O detection**: `SystemIo::storage` probes `/sys/block/*/queue/`
  to classify NVMe, SATA SSD, and HDD with estimated bandwidth.
- **Ingestion estimation**: `SystemIo::estimate_ingestion_secs()` estimates
  data loading time given dataset size and detected storage throughput.
- **New types**: `SystemIo`, `Interconnect`, `InterconnectKind`,
  `StorageDevice`, `StorageKind` — all serializable.

#### Detection improvements

- **AMD ROCm enrichment**: clock speeds (`pp_dpm_sclk`/`pp_dpm_mclk`),
  VBIOS version, GPU temperature, power draw, and utilization from sysfs.
  CXL-attached memory detection for MI300X/MI350.
- **Vulkan full parsing**: compute queue families, queue counts, and subgroup
  sizes from full `vulkaninfo` output (not just `--summary`).
- **NVIDIA Grace Hopper**: detects GH200/GH100 from GPU name, adds 480 GB
  unified LPDDR5X to reported HBM for capacity planning.

#### New backends (untested — written from documentation)

- **Cerebras WSE**: `cerebras_cli system-info` + `/dev/cerebras*` fallback.
- **Graphcore IPU**: `gc-info` JSON parsing + `/dev/ipu*` fallback.
- **Groq LPU**: `/dev/groq*` placeholder (driver not yet public).
- **Samsung NPU**: `/sys/class/misc/samsung_npu` + `/dev/samsung_npu*`.
- **MediaTek APU**: `/sys/class/misc/mtk_apu` + `/dev/mtk_mdla*`.

#### API and CLI

- **Schema v2**: `SCHEMA_VERSION` bumped to 2, formalizing all system I/O
  fields, power/thermal fields, and the `Timeout` error variant.
- **`DetectionError::Timeout`**: new error variant for timed-out tools,
  separate from `ToolFailed`. Enables programmatic retry logic.
- **True async detection**: `detect_async()` now uses
  `tokio::process::Command` for non-blocking subprocess I/O. CLI backends
  run as concurrent tokio tasks, sysfs-only backends in a single
  `spawn_blocking`.
- **`--columns`**: select specific table columns (`--columns name,mem,bw`).
- **`--tsv`**: tab-separated output for machine-readable table data.
- **`--watch` deltas**: memory usage changes shown between refreshes.
- **`--alert`**: threshold alerts during watch mode (`--alert mem>90`).
- **CLI table**: now shows Free VRAM, BW, PCIe, NUMA, plus Interconnects
  and Storage sections.

#### Testing

- **Hardware integration tests**: `tests/hardware_integration.rs` with 17
  tests covering CPU, ROCm, Vulkan, PCIe, bandwidth, storage, interconnects,
  JSON roundtrip, and concurrent detection. Auto-skips when hardware absent.
- **Fuzz testing**: 9 `cargo-fuzz` targets covering all CLI output parsers.
  Found and fixed integer overflow in CUDA memory parser.
- **Load testing**: concurrent 4-thread detection test + benchmark.
- **System I/O benchmarks**: per-backend, serialization, deserialization,
  and query benchmarks in `benches/detect.rs`.

#### Documentation

- **Troubleshooting guide**: `docs/troubleshooting.md`.
- **Performance tuning guide**: `docs/performance.md`.
- **Migration guide**: `docs/migration.md` (v0.19.3 → v0.20.3).
- **Crate-level docs**: expanded with error handling, custom backends, serde
  integration, and system I/O examples.

### Security

- **Subprocess environment sanitization**: `run_tool()` strips `LD_PRELOAD`,
  `LD_LIBRARY_PATH`, `DYLD_INSERT_LIBRARIES`, `DYLD_LIBRARY_PATH` from child
  processes to prevent library injection.
- **Windows `which()` improvements**: tries `.exe`, `.cmd`, `.bat` extensions
  when the tool name has no extension.
- **TOCTOU documentation**: the inherent time-of-check-time-of-use gap
  between path resolution and execution is documented as an accepted risk.

### Fixed

- **Integer overflow in CUDA parser**: `memory.used`/`memory.free` values
  exceeding u64 range on multiply now use `saturating_mul` with range filter.
  Found via fuzz testing.
- **Unbounded CSV field parsing**: CUDA parser now caps CSV splits to 20
  fields to prevent memory exhaustion from malicious `nvidia-smi` output.
- **Path traversal in PCI address handling**: PCI addresses in `pcie.rs`
  and `numa.rs` are now validated (hex+colon+dot only) and paths are
  canonicalized to prevent symlink-based information disclosure.
- **Grace Hopper memory validation**: unified memory is only added when
  reported HBM is in the realistic 80–100 GB range (prevents miscalculation
  from malformed nvidia-smi output).
- **Silent device ID fallback**: TPU and Neuron `/dev` parsers now skip
  malformed device names instead of silently mapping them to device 0.
- **Unbounded Vulkan device name**: `vulkaninfo` device names capped at
  256 characters to prevent memory exhaustion.
- **Defensive CSV bounds**: CUDA parser uses `.get()` for all field access
  instead of direct indexing.
- **Neuron JSON defaults removed**: malformed `neuron-ls` JSON devices are
  now skipped instead of using fabricated defaults (2 cores, 8192 MB).
- **Sysfs file size cap**: new `read_sysfs_string()` helper checks file
  size before reading (default 4 KiB cap) to prevent DoS via oversized
  sysfs files. Applied to ROCm VBIOS, revision, and DPM clock reads.
- **Subprocess zombie prevention**: `child.kill()` in timeout handler now
  polls `try_wait()` for up to 100ms instead of blocking `wait()` to avoid
  hanging on zombie processes.

### Performance

- **Pre-allocated profile collection**: `Vec::with_capacity(8)` avoids
  reallocation for typical systems with fewer than 8 accelerators.

## [0.19.3] - 2026-03-19

### Performance

- **Detection 3.5x faster**: eliminated per-subprocess reader threads in the
  command runner. Pipes are now read after the child exits (no deadlock risk
  since output is capped at 1 MiB). Poll interval reduced from 50ms to 10ms.
- **Single-pass `suggest_quantization`**: replaced 5 separate profile scans
  (`best_memory_for` per family) with one loop collecting all family maxima.
  Reduces O(5n) → O(n) on the profile list.
- **Sequential path for ≤1 backend**: `DetectBuilder::none().with_cuda()`
  skips `std::thread::scope` entirely, avoiding thread spawn/join overhead
  for selective single-backend detection.
- **CachedRegistry zero-copy**: `get()` now returns `Arc<AcceleratorRegistry>`
  instead of cloning the entire profile list on every call.
- **Reduced allocations**: `/proc/meminfo` parsing uses `nth()` iterator
  instead of collecting into a `Vec`. `read_limited` pre-allocates with
  `Vec::with_capacity`. `String::from_utf8_lossy().into_owned()` avoids
  double allocation. `#[inline]` on hot accessors (`all_profiles`,
  `warnings`, `estimate_memory`).

### Added

- **Async detection**: `AcceleratorRegistry::detect_async()` and
  `DetectBuilder::detect_async()` behind the `async-detect` cargo feature.
  Uses `tokio::task::spawn_blocking` to avoid blocking the async runtime.
- **CLI `--watch <secs>` mode**: re-detects on interval with screen clear and
  device-count change notifications.
- **CLI `--sort` flag**: sort `--table` output by `mem`, `name`, or `family`.
- **CLI `--family` flag**: filter `--table` output to a specific family
  (`gpu`, `tpu`, `npu`, `asic`, `cpu`).
- **C FFI** (`src/ffi.rs` + `include/ai_hwaccel.h`): `extern "C"` API with
  `ai_hwaccel_detect()`, `ai_hwaccel_device_count()`,
  `ai_hwaccel_has_accelerator()`, `ai_hwaccel_accelerator_memory()`,
  `ai_hwaccel_json()` and corresponding free functions.
- **Framework integration guide**: `docs/guides/framework-integration.md` with
  code examples for `candle`, `burn`, `tch-rs`, `ort`, multi-device sharding,
  and training memory budgeting.
- `tokio` optional dependency (behind `async-detect` feature).
- **Feature flags**: each of the 11 hardware backends is gated behind a cargo
  feature (`cuda`, `rocm`, `apple`, `vulkan`, `intel-npu`, `amd-xdna`, `tpu`,
  `gaudi`, `aws-neuron`, `intel-oneapi`, `qualcomm`). All enabled by default
  via `all-backends`. Disabled backends are not compiled.
- **CLI `--table` / `-t` flag**: human-readable tabular device listing with
  device ID, name, memory, family, and status columns.
- **CLI `--debug` / `-d` flag**: sets `RUST_LOG=debug` for verbose detection
  diagnostics without manually setting the environment variable.
- **Serde schema version**: `AcceleratorRegistry` now serializes a
  `schema_version` field (currently `1`) for forward-compatibility. Accessible
  via `registry.schema_version()` and `SCHEMA_VERSION` constant.
- **Property-based tests**: `proptest` fuzzing for `estimate_memory`,
  `plan_sharding`, `suggest_quantization`, and `estimate_training_memory`
  across random parameter counts and device configurations.
- **Architecture decision records**: `docs/decisions/` with 4 ADRs:
  sysfs-over-vendor-SDKs, calendar versioning, parallel detection, and
  feature flags per backend.
- **Crate-level guide**: expanded `lib.rs` documentation with a 4-step
  walkthrough (detect → query → plan → train) and cargo feature reference
  table.
- **JSON schema**: `docs/schema.json` documenting the serialized registry
  format (JSON Schema draft 2020-12).
- **`CachedRegistry`**: thread-safe detection cache with configurable TTL.
  Avoids redundant CLI tool invocations on repeated `detect()` calls.
- **Mock detection tests**: `tests/mock_detection.rs` with 11 tests using
  `tempfile` to build fake sysfs trees for hardware-independent backend
  testing, plus serde `deny_unknown_fields` rejection tests and schema
  version validation.
- **Windows CI**: added `x86_64-pc-windows-msvc` to the CI test matrix.
- `proptest` and `tempfile` dev-dependencies.
- Test suite expanded to 173 tests (140 unit + 9 integration + 11 mock +
  13 doc-tests).
- **Modular architecture**: refactored 3 monolithic source files into 23
  focused modules with single responsibilities.
  - `types.rs` (714 lines) split into `hardware/` (with `tpu.rs`, `gaudi.rs`,
    `neuron.rs`), `profile.rs`, `quantization.rs`, `requirement.rs`,
    `sharding.rs`, and `training.rs`.
  - `detect.rs` (693 lines) split into `registry.rs` (struct + query methods)
    and `detect/` module with one file per hardware backend.
  - `plan.rs` split into `plan.rs` (sharding logic) and `training.rs`
    (training types and memory estimation).
  - `tests.rs` (849 lines) split into `tests/` module with 10 files by concern.
- **`DetectionError` type** (`src/error.rs`): non-fatal detection errors
  captured as structured warnings (`ToolNotFound`, `ToolFailed`, `ParseError`,
  `SysfsReadError`) and accessible via `AcceleratorRegistry::warnings()`.
- **`DetectBuilder`**: selective backend detection via builder pattern —
  `AcceleratorRegistry::builder().with_cuda().without_vulkan().detect()`.
  Includes `Backend` enum with `ALL` constant.
- **`#[non_exhaustive]`** on `AcceleratorType`, `AcceleratorFamily`,
  `QuantizationLevel`, `AcceleratorRequirement`, and `DetectionError` for
  semver-safe enum extension.
- **Convenience constructors** on `AcceleratorProfile`: `cuda()`, `rocm()`,
  `tpu()`, `gaudi()`, `cpu()` for test and manual-config ergonomics.
- **`Display` for `ShardingPlan`**: human-readable multi-line plan summary
  showing strategy, memory, throughput, and per-shard device assignments.
- **CLI `--pretty` / `-p` flag**: pretty-printed JSON output.
- **CLI warnings**: detection warnings appear in `--summary` JSON output and
  are logged at `warn` level.
- **Structured logging**: CLI binary uses `tracing-subscriber` with `RUST_LOG`
  environment variable support and `--json-log` flag for structured JSON
  output to stderr.
- **Parallel detection**: all backends run concurrently via
  `std::thread::scope`, reducing wall-clock latency on multi-tool systems.
  Vulkan deduplication moved to a post-pass.
- **Safe command runner** (`detect/command.rs`): all CLI-based detectors use
  `run_tool()` which enforces:
  - Absolute path resolution via `which()` to prevent `$PATH` hijacking.
  - 5-second timeout with `child.kill()` on expiry.
  - Output size limits: stdout capped at 1 MiB, stderr at 4 KiB.
- **Input validation**: `validate_device_id()` (0--1024) and
  `validate_memory_mb()` (0--16 TiB) reject out-of-range parsed values from
  CLI tool output.
- **`#[serde(deny_unknown_fields)]`** on `AcceleratorRegistry`,
  `AcceleratorProfile`, `ModelShard`, `ShardingPlan` to reject unexpected
  JSON fields during deserialization.
- **`deny.toml`**: `cargo-deny` configuration for license allowlist, advisory
  checks, and crate source restrictions. New `make deny` target.
- **Threat model**: `docs/development/threat-model.md` documenting attack
  surface, trust assumptions, and mitigations.
- **Integration tests**: `tests/integration.rs` with 9 end-to-end tests
  covering the detect-query-plan pipeline, builder, JSON roundtrip, manual
  registry, training estimation, Display impls, and warnings.
- **Benchmark suite**: `criterion` benchmarks in `benches/` for `detect()`,
  `plan_sharding()`, `suggest_quantization()`, `estimate_memory()`, and
  `estimate_training_memory()`.
- **`examples/` directory**: four runnable examples — `detect.rs`, `plan.rs`,
  `training.rs`, `json_output.rs`.
- **Rustdoc examples**: `# Examples` sections on `AcceleratorRegistry`,
  `AcceleratorProfile`, `QuantizationLevel`, `DetectionError`, and
  `estimate_training_memory()`. All compile as doc-tests.
- **CI improvements**: cross-platform matrix (Linux + macOS), MSRV job
  (Rust 1.89), coverage via `cargo-llvm-cov` + Codecov, `cargo-deny`
  supply-chain checks.
- `tracing-subscriber` dependency (with `env-filter` and `json` features).
- `criterion` dev-dependency for benchmarks.
- `docs/development/roadmap.md` documenting the path to v1.0.
- Test suite expanded from 46 to 149 tests (133 unit + 9 integration +
  7 doc-tests).

### Changed

- **Switched from CalVer to SemVer**: version is now `0.19.3` (pre-1.0). The
  `0.x` series may contain breaking changes between minor versions.
- **NVIDIA detection**: now parses `driver_version` from `nvidia-smi` and
  reports structured `DetectionError` on tool failure or parse errors.
- **Vulkan detection**: parses `vulkaninfo --summary` for real device names,
  memory heap sizes, API version, and driver version instead of registering a
  generic placeholder device.
- **Apple detection**: macOS support via `system_profiler SPHardwareDataType`
  for chip name and unified memory size. ANE memory estimate varies by chip
  generation (M1: 4 GB, M2: 6 GB, M3/M4: 8 GB). Linux Asahi detection
  preserved as fallback.
- **CPU memory detection**: macOS fallback via `sysctl hw.memsize` when
  `/proc/meminfo` is unavailable.
- All detection backends now report structured warnings and use the safe
  command runner for CLI tool invocations.

### Fixed

- **`suggest_quantization` semantic bugs**: no longer returns BF16 for
  Qualcomm/Neuron AI ASICs (only Gaudi). Falls back through FP16→INT8→INT4
  on CPU instead of unconditionally returning FP16 for models that don't fit.
- **CPU memory detection**: macOS `sysctl` fallback now uses the safe command
  runner (`run_tool`) with absolute path resolution and timeout, matching all
  other CLI tool invocations.
- **Integer overflow safety**: all memory multiplications (TPU HBM × chip
  count, Neuron cores × memory, KB→bytes) use `saturating_mul` to prevent
  panics on extreme values.
- **Pipeline parallel layer assignment**: last shard now captures all remaining
  layers instead of potentially leaving a gap when layer count doesn't divide
  evenly across devices.
- **Per-chip memory precision**: TPU tensor-parallel uses ceiling division
  (`div_ceil`) so no bytes are lost to rounding.
- **Cache mutex poisoning**: `CachedRegistry` recovers from poisoned locks
  instead of panicking.
- **UTF-8 safe truncation**: CLI table column truncation uses `chars().take()`
  instead of byte slicing.
- **Windows compatibility**: mock detection tests gate Unix symlinks behind
  `#[cfg(unix)]` so the test file compiles on Windows.
- **Gaudi detection**: malformed CSV lines now produce `ParseError` warnings
  instead of being silently skipped. Device IDs and memory values are
  validated with `validate_device_id` / `validate_memory_mb`.
- `Cargo.toml` license field corrected from `AGPL-3.0` to the SPDX-correct
  `AGPL-3.0-only`.
- Added missing `homepage` and `readme` fields to `Cargo.toml` for crates.io
  compliance.

## [2026.3.19] - 2026-03-19 (CalVer, pre-SemVer switch)

### Added

- Initial public release.
- Hardware detection for 13 accelerator families: NVIDIA CUDA, AMD ROCm,
  Apple Metal, Apple ANE, Intel NPU, AMD XDNA, Google TPU (v4/v5e/v5p),
  Intel Gaudi (2/3), AWS Inferentia, AWS Trainium, Qualcomm Cloud AI 100,
  Vulkan Compute, and CPU fallback.
- `AcceleratorRegistry` with `detect()`, querying, and planning APIs.
- Quantization-aware memory estimation (`FP32`, `FP16`, `BF16`, `INT8`, `INT4`).
- Model sharding planner with tensor-parallel, pipeline-parallel, and
  data-parallel strategies.
- Training memory estimator for full fine-tune, LoRA, QLoRA, DPO, RLHF, and
  distillation methods.
- Serde support for all public types.
- CLI binary with `--summary` and `--version` flags.
- CI pipeline (format, clippy, tests, cargo-audit).
- Release automation with version consistency checks.
- Project documentation: `README.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `CHANGELOG.md`.
- `LICENSE` (AGPL-3.0-only).
- `Makefile` for local development (`check`, `fmt`, `clippy`, `test`, `build`,
  `doc`, `clean`).
- `scripts/version-bump.sh` for calendar versioning.
