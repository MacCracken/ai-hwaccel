# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [calendar versioning](https://calver.org/) (`YYYY.M.D`).

## [Unreleased]

### Changed

- **Modular architecture**: refactored monolithic source files into focused
  modules with single responsibilities.
  - `types.rs` (714 lines) split into `hardware/` (with `tpu.rs`, `gaudi.rs`,
    `neuron.rs`), `profile.rs`, `quantization.rs`, `requirement.rs`,
    `sharding.rs`, and `training.rs`.
  - `detect.rs` (693 lines) split into `registry.rs` (struct + query methods)
    and `detect/` module with one file per hardware backend (`cuda.rs`,
    `rocm.rs`, `apple.rs`, `vulkan.rs`, `intel_npu.rs`, `amd_xdna.rs`,
    `tpu.rs`, `gaudi.rs`, `neuron.rs`, `intel_oneapi.rs`, `qualcomm.rs`).
  - `plan.rs` split into `plan.rs` (sharding logic) and `training.rs`
    (training types and memory estimation).
  - `tests.rs` (849 lines) split into `tests/` module with files per concern:
    `classification.rs`, `display.rs`, `quantization.rs`, `requirement.rs`,
    `registry.rs`, `sharding.rs`, `training.rs`, `serde.rs`.

### Added

- **Structured logging**: CLI binary now uses `tracing-subscriber` with
  `RUST_LOG` environment variable support and optional `--json-log` flag for
  structured JSON output to stderr.
- Added `tracing-subscriber` dependency (with `env-filter` and `json` features)
  for the CLI binary.
- `ROADMAP.md` documenting the path to v1.0.

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
