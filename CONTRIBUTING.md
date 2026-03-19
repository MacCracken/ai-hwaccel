# Contributing to ai-hwaccel

Thank you for your interest in contributing! This document explains how to get
started, what we expect from contributions, and how the review process works.

## Getting started

1. **Fork and clone** the repository.
2. Install the Rust toolchain (the repo pins a version in `rust-toolchain.toml`).
3. Run `make check` to verify everything builds and passes locally.

## Development workflow

```sh
make check   # format check + clippy (zero warnings) + tests
make fmt     # check formatting only
make clippy  # lint only
make test    # tests only
make doc     # build rustdoc
```

CI runs the same `check` pipeline, so if it passes locally it will pass in CI.

## What to contribute

Contributions are welcome in several areas:

- **New accelerator backends** -- each detector is a self-contained file under
  `src/detect/` (e.g. `src/detect/cuda.rs`).
- **Improved detection accuracy** -- better sysfs paths, parser robustness,
  version identification.
- **Training/inference planning** -- more accurate memory models, new sharding
  strategies in `src/plan.rs` and `src/training.rs`.
- **Documentation** -- rustdoc improvements, examples, guides.
- **Bug fixes** -- always welcome.

## Code style

- Run `cargo fmt` before committing. CI enforces this.
- `cargo clippy -- -D warnings` must pass with zero warnings.
- Prefer explicit types over inference in public API signatures.
- Every public item must have a doc comment (`///`).
- Keep dependencies minimal. This crate intentionally avoids vendor SDKs.

## Project layout

```
src/
├── hardware/           # Type definitions per hardware family
│   ├── mod.rs          #   AcceleratorType, AcceleratorFamily
│   ├── tpu.rs          #   TpuVersion
│   ├── gaudi.rs        #   GaudiGeneration
│   └── neuron.rs       #   NeuronChipType
├── profile.rs          # AcceleratorProfile
├── quantization.rs     # QuantizationLevel
├── requirement.rs      # AcceleratorRequirement
├── sharding.rs         # ShardingStrategy, ModelShard, ShardingPlan
├── training.rs         # TrainingMethod, TrainingTarget, MemoryEstimate
├── registry.rs         # AcceleratorRegistry (struct + query methods)
├── detect/             # One file per hardware backend
│   ├── mod.rs          #   detect() orchestrator + helpers
│   ├── cuda.rs         #   ...through qualcomm.rs
├── plan.rs             # Sharding planner (impl on AcceleratorRegistry)
└── tests/              # One file per concern
```

## Adding a new accelerator

1. Add a variant to `AcceleratorType` in `src/hardware/mod.rs`.
2. Implement the classification methods (`is_gpu()`, `family()`, `rank()`,
   `throughput_multiplier()`, etc.) for the new variant.
3. Create a new `src/detect/<name>.rs` file with a `detect_<name>()` function,
   following the pattern of existing detectors.
4. Register the new module in `src/detect/mod.rs` and call it from `detect()`.
5. Add tests in `src/tests/` covering the new type and detection.
6. Update the hardware table in `src/lib.rs` and `README.md`.

## Commit messages

- Use imperative mood: "add TPU v6 detection", not "added" or "adds".
- Keep the first line under 72 characters.
- Reference issues with `#123` where applicable.

## Pull requests

- One logical change per PR.
- Include tests for new functionality.
- Update documentation if public API changes.
- PRs must pass CI (format, clippy, tests, cargo-audit).
- Maintainers may request changes before merging.

## Versioning

This crate uses calendar versioning (`YYYY.M.D`). Version bumps are handled by
maintainers using `scripts/version-bump.sh`. Contributors do **not** need to
bump the version in their PRs.

## License

By contributing you agree that your contributions will be licensed under the
[GNU Affero General Public License v3.0](LICENSE).
