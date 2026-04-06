# ai-hwaccel

Universal AI hardware accelerator detection, capability querying, and workload
planning for Rust.

`ai-hwaccel` gives you a single-call view of every AI-capable accelerator on
the system -- GPUs, TPUs, NPUs, and cloud inference chips -- then helps you
decide how to quantize and shard a model across them.

## Why this crate?

ML frameworks typically hard-code support for one or two backends. If you need
portable hardware discovery (e.g. for a model server, training launcher, or
benchmarking harness) you end up writing the same sysfs/CLI-probing boilerplate
everywhere. This crate consolidates that into a tested, vendor-neutral library
with **zero compile-time SDK dependencies**.

## Supported hardware

| Family | Variants | Detection method |
|---|---|---|
| NVIDIA CUDA | GeForce, Tesla, A100, H100, ... | `nvidia-smi` on `$PATH` |
| AMD ROCm | MI250, MI300, RX 7900 | `/sys/class/drm` sysfs |
| Apple Metal | M1--M4 GPU cores | `system_profiler` / `sysctl` |
| Apple ANE | Neural Engine | `system_profiler` / `sysctl` |
| Intel NPU | Meteor Lake+ | `/sys/class/misc/intel_npu` |
| AMD XDNA | Ryzen AI NPU | `/sys/class/accel/*/device/driver` |
| Google TPU | v4, v5e, v5p | `/dev/accel*` + sysfs version |
| Intel Gaudi | Gaudi 2, Gaudi 3 (Habana HPU) | `hl-smi` on `$PATH` |
| AWS Inferentia | inf1, inf2 | `/dev/neuron*` or `neuron-ls` |
| AWS Trainium | trn1 | `/dev/neuron*` + sysfs |
| Intel oneAPI | Arc, Data Center Max | `xpu-smi` on `$PATH` |
| Qualcomm Cloud AI | AI 100 | `/dev/qaic_*` or `/sys/class/qaic` |
| Cerebras WSE | Wafer-Scale Engine | `/dev/cerebras*` sysfs |
| Graphcore IPU | IPU-POD | `gc-info` or sysfs |
| Groq LPU | Language Processing Unit | `/dev/groq*` sysfs |
| Samsung NPU | Exynos NPU | `/sys/class/npu` sysfs |
| MediaTek APU | Dimensity APU | `/sys/class/misc/apusys` sysfs |
| Vulkan Compute | Any Vulkan 1.1+ device | `vulkaninfo` on `$PATH` |
| Windows GPU | Any DirectX GPU | WMI / PowerShell |
| CPU | Always present | `/proc/meminfo` (16 GiB fallback) |

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
ai-hwaccel = "1.2"
```

### Library usage

```rust
use ai_hwaccel::{AcceleratorRegistry, QuantizationLevel};

// 1. Discover hardware
let registry = AcceleratorRegistry::detect();
println!("Best device: {}", registry.best_available().unwrap());

// 2. Pick quantization for a 7 B-param model
let quant = registry.suggest_quantization(7_000_000_000);
println!("Recommended: {quant}");

// 3. Plan sharding for a 70 B model at BF16
let plan = registry.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16);
println!(
    "Strategy: {}, est. {:.0} tok/s",
    plan.strategy,
    plan.estimated_tokens_per_sec.unwrap_or(0.0)
);
```

### CLI usage

```sh
ai-hwaccel                  # Full registry JSON to stdout
ai-hwaccel --summary        # Compact summary JSON
ai-hwaccel --version        # Print version

# Logging (logs go to stderr, data to stdout)
RUST_LOG=debug ai-hwaccel   # Verbose detection diagnostics
```

## Architecture

The crate is organized into focused modules, each with a single responsibility:

```
src/
├── lib.rs                  # Crate root, re-exports
├── main.rs                 # CLI binary
├── hardware/               # Device type definitions
│   ├── mod.rs              #   AcceleratorType (18 variants), AcceleratorFamily
│   ├── tpu.rs              #   TpuVersion (v4/v5e/v5p)
│   ├── gaudi.rs            #   GaudiGeneration (Gaudi2/Gaudi3)
│   └── neuron.rs           #   NeuronChipType (Inferentia/Trainium)
├── profile.rs              # AcceleratorProfile (capabilities per device)
├── registry.rs             # AcceleratorRegistry, DetectBuilder
├── plan.rs                 # Sharding planner (topology-aware)
├── quantization.rs         # QuantizationLevel (FP32 → INT4)
├── sharding.rs             # ShardingStrategy, ModelShard, ShardingPlan
├── training.rs             # TrainingMethod, MemoryEstimate
├── cost.rs                 # Cloud instance pricing + recommendations
├── model_compat.rs         # Model compatibility database (26 models)
├── model_format.rs         # .safetensors/.gguf/.onnx/.pt header parsing
├── requirement.rs          # AcceleratorRequirement (scheduling constraints)
├── system_io.rs            # Interconnects, storage, runtime environment
├── cache.rs                # CachedRegistry, DiskCachedRegistry
├── lazy.rs                 # LazyRegistry (on-demand detection)
├── ffi.rs                  # C-compatible FFI bindings
├── async_detect.rs         # Tokio-based async detection
├── units.rs                # Named constants for unit conversions
├── error.rs                # DetectionError variants
├── detect/                 # Hardware detection (one file per backend)
│   ├── mod.rs              #   Orchestrator + backend_table! macro
│   ├── cuda.rs             #   NVIDIA via nvidia-smi
│   ├── rocm.rs             #   AMD via sysfs /sys/class/drm
│   ├── apple.rs            #   Metal + ANE via system_profiler
│   ├── vulkan.rs           #   Vulkan via vulkaninfo
│   ├── tpu.rs, gaudi.rs    #   TPU / Gaudi via sysfs + CLI
│   ├── neuron.rs           #   AWS Neuron via neuron-ls
│   ├── intel_npu.rs        #   Intel NPU via sysfs
│   ├── amd_xdna.rs         #   AMD XDNA via sysfs
│   ├── intel_oneapi.rs     #   Intel oneAPI via xpu-smi
│   ├── qualcomm.rs         #   Qualcomm AI 100 via sysfs
│   ├── cerebras.rs         #   Cerebras WSE via sysfs
│   ├── graphcore.rs        #   Graphcore IPU via gc-info
│   ├── groq.rs             #   Groq LPU via /dev/groq*
│   ├── samsung_npu.rs      #   Samsung NPU via sysfs
│   ├── mediatek_apu.rs     #   MediaTek APU via sysfs
│   ├── windows.rs          #   Windows GPU via WMI
│   ├── interconnect.rs     #   InfiniBand, NVLink, NVSwitch, XGMI, ICI
│   ├── bandwidth.rs        #   Memory bandwidth probing
│   ├── pcie.rs, numa.rs    #   PCIe link + NUMA topology
│   ├── disk.rs             #   Storage device detection
│   ├── command.rs          #   Safe subprocess execution
│   └── environment.rs      #   Runtime environment (Docker, K8s, cloud)
└── tests/                  # 538 tests across 24 modules
```

## Core concepts

### `AcceleratorRegistry`

The main entry point. `detect()` probes the system and returns a registry of
every discovered accelerator. From there you can:

- **Query** -- `available()`, `best_available()`, `by_family()`, `satisfying()`
- **Plan** -- `suggest_quantization()`, `plan_sharding()`
- **Inspect** -- `total_memory()`, `total_accelerator_memory()`, `has_accelerator()`
- **What-if** -- `what_if_add()`, `what_if_remove()`, `what_if_replace()`

### `AcceleratorType`

An 18-variant enum representing each supported device family. Provides
classification helpers (`is_gpu()`, `is_npu()`, `is_tpu()`, `is_ai_asic()`) and
throughput/training multipliers used by the planner.

### `QuantizationLevel`

Five levels from full-precision `FP32` down to `Int4`, each carrying its
bits-per-parameter and memory-reduction factor.

### `ShardingPlan`

Describes how a model should be distributed across devices:

| Strategy | When used |
|---|---|
| `None` | Model fits on a single device |
| `TensorParallel` | TPU pod slices (ICI) or NVSwitch-connected GPUs |
| `PipelineParallel` | Multiple GPUs or AI ASICs |
| `DataParallel` | Replicas for throughput |

### Training memory estimation

`estimate_training_memory()` returns a per-component breakdown (model,
optimizer, activations) for methods including full fine-tune, LoRA, QLoRA, DPO,
RLHF, and distillation.

## How detection works

All detection is best-effort and non-destructive:

1. **sysfs probing** -- reads `/sys/class/drm`, `/sys/class/misc`, etc.
2. **`/dev` introspection** -- checks for device nodes like `/dev/accel*`,
   `/dev/neuron*`, `/dev/qaic_*`.
3. **`$PATH` tool execution** -- runs `nvidia-smi`, `hl-smi`, `vulkaninfo`,
   `neuron-ls` when present, parses their output.

If a tool or sysfs path is absent the accelerator simply isn't registered --
no errors, no panics.

## Logging

The library uses [`tracing`](https://docs.rs/tracing) for all diagnostic output.
No logs are emitted unless the consuming application installs a tracing
subscriber. The CLI binary ships with `tracing-subscriber` and respects the
standard `RUST_LOG` environment variable:

```sh
RUST_LOG=ai_hwaccel=debug   # Library-level debug messages
RUST_LOG=trace               # Everything including dependency traces
```

## Minimum supported Rust version (MSRV)

**Rust 1.89** (edition 2024). Tracked in `rust-toolchain.toml`.

## Binary size

Release builds use LTO, single codegen unit, `opt-level = "z"`, and symbol
stripping:

| Build | Size |
|-------|------|
| All backends (default) | **847 KB** |
| Minimal (CPU only) | **692 KB** |

## Versioning

This crate uses **semantic versioning**. The version is read from the
`VERSION` file at the repo root and kept in sync with `Cargo.toml` via
`scripts/version-bump.sh`.

## Documentation

| Document | Description |
|---|---|
| [Production guide](docs/guides/production.md) | Deployment, caching, security, containers, C FFI |
| [Testing guide](docs/guides/testing.md) | Running tests, hardware setup, CI matrix |
| [Framework integration](docs/guides/framework-integration.md) | candle, burn, tch-rs, ort examples |
| [Threat model](docs/development/threat-model.md) | Security boundaries and mitigations |
| [JSON schema](docs/schema.json) | Serialized registry format specification |
| [Architecture decisions](docs/decisions/) | ADRs for key design choices |
| [Roadmap](docs/development/roadmap.md) | Remaining work and future plans |
| [Contributing](CONTRIBUTING.md) | How to contribute |
| [Changelog](CHANGELOG.md) | Release history |

## Development

```sh
make check   # fmt + clippy + test (same as CI)
make audit   # cargo-audit (vulnerabilities)
make deny    # cargo-deny (licenses, advisories)
make vet     # cargo-vet (dependency audits)
make doc     # generate rustdoc
make build   # release build
```

## License

Licensed under the [GNU General Public License v3.0](LICENSE).

See [LICENSE](LICENSE) for the full text.
