# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [calendar versioning](https://calver.org/) (`YYYY.M.D`).

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
- CI pipeline (format, clippy, tests, cargo-audit).
- Release automation with version consistency checks.
