# Contributing to ai-hwaccel

Thank you for your interest in contributing! This document explains how to get
started, what we expect from contributions, and how the review process works.

## Getting started

1. **Fork and clone** the repository.
2. Install the Cyrius toolchain (3.9.0+).
3. Run `cyrius test && cyrius lint src/main.cyr && cyrius fmt src/main.cyr --check` to verify everything builds and passes locally.

## Development workflow

```sh
cyrius test                          # run all 491 assertions (10 test phases)
cyrius lint src/main.cyr             # lint (zero warnings)
cyrius fmt src/main.cyr --check      # check formatting
cyrius build src/main.cyr build/ai-hwaccel  # build binary
cyrius doc                           # build documentation
```

CI runs the same pipeline, so if it passes locally it will pass in CI.

## What to contribute

Contributions are welcome in several areas:

- **New accelerator backends** -- each detector is a self-contained file under
  `src/detect/` (e.g. `src/detect/cuda.cyr`).
- **Improved detection accuracy** -- better sysfs paths, parser robustness,
  version identification.
- **Training/inference planning** -- more accurate memory models, new sharding
  strategies in `src/plan.cyr` and `src/training.cyr`.
- **Documentation** -- doc improvements, examples, guides.
- **Bug fixes** -- always welcome.

## Code style

- Run `cyrius fmt src/main.cyr` before committing. CI enforces this.
- `cyrius lint src/main.cyr` must pass with zero warnings.
- Prefer explicit types over inference in public API signatures.
- Every public item must have a doc comment (`///`).
- Zero external dependencies. This project intentionally avoids vendor SDKs.

## Project layout

```
src/
├── hardware/            # Type definitions per hardware family
│   ├── mod.cyr          #   AcceleratorType, AcceleratorFamily
│   ├── tpu.cyr          #   TpuVersion
│   ├── gaudi.cyr        #   GaudiGeneration
│   └── neuron.cyr       #   NeuronChipType
├── profile.cyr          # AcceleratorProfile
├── quantization.cyr     # QuantizationLevel
├── requirement.cyr      # AcceleratorRequirement
├── sharding.cyr         # ShardingStrategy, ModelShard, ShardingPlan
├── training.cyr         # TrainingMethod, TrainingTarget, MemoryEstimate
├── registry.cyr         # AcceleratorRegistry (struct + query + what-if)
├── detect/              # One file per hardware backend
│   ├── mod.cyr          #   detect() orchestrator + helpers
│   ├── cuda.cyr         #   ...through qualcomm.cyr
│   ├── interconnect.cyr #   NVLink, NVSwitch, XGMI, ICI, RoCE v2
│   └── environment.cyr  #   Docker/K8s/cloud + GPU device plugin detection
├── plan.cyr             # Sharding planner (impl on AcceleratorRegistry)
├── cost.cyr             # Cloud instance recommendation
├── model_compat.cyr     # Model compatibility database (26 models)
├── model_format.cyr     # File format detection (SafeTensors, GGUF, ONNX, PyTorch)
├── system_io.cyr        # Interconnects, storage, runtime environment
├── cache.cyr            # CachedRegistry with TTL
├── lazy.cyr             # LazyRegistry for deferred detection
├── ffi.cyr              # C FFI bindings
├── units.cyr            # Constants
└── tests/               # .tcyr test files per concern
```

## Adding a new accelerator

1. Add a variant to `AcceleratorType` in `src/hardware/mod.cyr`.
2. Implement the classification methods (`is_gpu()`, `family()`, `rank()`,
   `throughput_multiplier()`, etc.) for the new variant.
3. Create a new `src/detect/<name>.cyr` file with a `detect_<name>()` function,
   following the pattern of existing detectors.
4. Register the new module in `src/detect/mod.cyr` and call it from `detect()`.
5. Add tests in `src/tests/` covering the new type and detection (`.tcyr` files).
6. Update the hardware table in `src/main.cyr` and `README.md`.

## Commit messages

- Use imperative mood: "add TPU v6 detection", not "added" or "adds".
- Keep the first line under 72 characters.
- Reference issues with `#123` where applicable.

## Pull requests

- One logical change per PR.
- Include tests for new functionality.
- Update documentation if public API changes.
- PRs must pass CI (format, lint, tests).
- Maintainers may request changes before merging.

## Versioning

This project uses semantic versioning. Version bumps are handled by maintainers
using `scripts/version-bump.sh`. Contributors do **not** need to bump the
version in their PRs.

## License

By contributing you agree that your contributions will be licensed under the
[GNU General Public License v3.0](LICENSE).
