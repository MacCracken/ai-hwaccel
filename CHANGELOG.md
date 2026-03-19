# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [calendar versioning](https://calver.org/) (`YYYY.M.D`).

## [Unreleased]

### Added

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

- `Cargo.toml` license field corrected from `AGPL-3.0` to the SPDX-correct
  `AGPL-3.0-only`.
- Added missing `homepage` and `readme` fields to `Cargo.toml` for crates.io
  compliance.

## [2026.3.19] - 2026-03-19

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
