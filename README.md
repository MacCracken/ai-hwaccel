# ai-hwaccel

Universal AI hardware accelerator detection, capability querying, and workload
planning. Written in [Cyrius](https://github.com/MacCracken/cyrius) — zero
external dependencies.

`ai-hwaccel` gives you a single-call view of every AI-capable accelerator on
the system — GPUs, TPUs, NPUs, and cloud inference chips — then helps you
decide how to quantize and shard a model across them.

## Key Numbers

| Metric | Value |
|--------|-------|
| Binary size | **286 KB** |
| Compiler | Cyrius cycc 6.0.0 |
| Tests | 518 assertions (11 test units) |
| Fuzz harnesses | 6 |
| Dependencies | **0** |
| Hardware families | 18 |
| Derived structs | 16 (every heap struct uses `#derive(accessors)`) |

## Supported Hardware

| Family | Variants | Detection method |
|--------|----------|------------------|
| NVIDIA CUDA | GeForce, Tesla, A100, H100, ... | `nvidia-smi` on `$PATH` |
| AMD ROCm | MI250, MI300, RX 7900 | `/sys/class/drm` sysfs |
| Apple Metal | M1–M4 GPU cores | `system_profiler` / `sysctl` |
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
| CPU | Always present | `/proc/meminfo` (16 GiB fallback) |

## Quick Start

```sh
# Build
cyrius build src/main.cyr build/ai-hwaccel

# Run
ai-hwaccel                  # Full registry JSON
ai-hwaccel --summary        # Compact summary JSON
ai-hwaccel --table          # Human-readable table
ai-hwaccel --cost 70B       # Cloud instance recommendation
ai-hwaccel --version        # Print version
```

### Using as a library

`cyrius distlib` bundles every non-CLI module into `dist/ai-hwaccel.cyr`,
a single self-contained file consumers pull via `cyrius deps`. Bundle
includes all 18 detection backends, the registry/profile surface, the
sharding planner, the cost model, the training memory estimator, and
the model-format header parser. Excluded: `src/main.cyr` (CLI argv) and
`src/json_out.cyr` (CLI output formatting — library consumers serialize
on their own terms).

Wire it from a consumer's `cyrius.cyml`:

```toml
[deps.ai-hwaccel]
git = "https://github.com/MacCracken/ai-hwaccel.git"
tag = "2.2.5"
modules = ["dist/ai-hwaccel.cyr"]
```

`cyrius deps` clones at the tag and drops the bundle into the
consumer's `lib/`. Include it like any other dep:

```cyrius
include "lib/ai-hwaccel.cyr"

# Full detection — sysfs probes plus vendor CLIs (nvidia-smi,
# system_profiler, vulkaninfo, hl-smi, neuron-ls, xpu-smi,
# cerebras_cli, gc-info) for the eight EXEC backends, and ibstat /
# nvidia-smi topo for the interconnect post-pass.
var r = registry_detect();
# ... reg_profiles(r), reg_count(r), ...
```

#### No-exec entry point (since 2.2.5)

Consumers with a no-subprocess contract — `mihi`'s probe surface, any
read-only system-info library — call `registry_detect_no_exec()`
instead. It masks off the eight exec-shelling backends (CUDA, Apple,
Vulkan, Gaudi, Neuron, Intel oneAPI, Cerebras, Graphcore) and skips
the `detect_interconnects` post-pass. The remaining eight backends —
ROCm, Intel NPU, AMD XDNA, TPU, Qualcomm, Groq, Samsung NPU, MediaTek
APU — plus the sysfs post-passes still run.

```cyrius
include "lib/ai-hwaccel.cyr"

# Pure sysfs/syscall reads — safe to call from probe contexts that
# forbid spawning processes (e.g. mihi).
var r = registry_detect_no_exec();
```

The classification lives in `backend_uses_exec(b)` in `src/types.cyr`;
`builder_no_exec()` is exposed if you want to compose your own mask
(e.g. `builder_no_exec() & builder_without(builder_no_exec(), BACKEND_GROQ)`).

Library consumers today: `mihi` v0.4.0 (M3 GPU probe, pending —
drives the 2.2.5 no-exec contract). All consumers are listed in
[Consumers](#consumers); historically every entry has been a binary
consumer — library use begins with mihi.

## Architecture

```
src/
├── main.cyr                CLI entry point
├── types.cyr               AcceleratorType (18 variants), AcceleratorFamily
├── profile.cyr             Device profile (memory, capabilities, throughput)
├── registry.cyr            AcceleratorRegistry, DetectBuilder (bitmask)
├── plan.cyr                Sharding planner (tensor/pipeline/data parallel)
├── quantization.cyr        QuantizationLevel (FP32 → INT4, fixed-point x1000)
├── training.cyr            Training memory estimation (8 methods)
├── cost.cyr                Cloud instance pricing + recommendation
├── model.cyr               Model compatibility database (data/models.json)
├── model_format.cyr        SafeTensors/GGUF/ONNX/PyTorch header detection
├── requirement.cyr         Accelerator requirement matching (scheduling)
├── async_detect.cyr        Threaded concurrent detection (thread.cyr)
├── cache.cyr               CachedRegistry + DiskCachedRegistry (TTL)
├── lazy.cyr                LazyRegistry (per-family on-demand detection)
├── json_out.cyr            JSON serialization
├── units.cyr               Named constants for unit conversions
├── error.cyr               Warning/error types
├── system_io.cyr           Interconnects, storage, runtime environment
└── detect/                 Hardware detection (19 modules)
    ├── cuda.cyr             NVIDIA via nvidia-smi
    ├── rocm.cyr             AMD via sysfs
    ├── apple.cyr            Metal + ANE via system_profiler
    ├── vulkan.cyr           Vulkan via vulkaninfo
    ├── tpu.cyr              Google TPU via sysfs
    ├── gaudi.cyr            Intel Gaudi via hl-smi
    ├── neuron.cyr           AWS Neuron via neuron-ls
    ├── intel.cyr            Intel NPU + oneAPI
    ├── cloud_asic.cyr       Cerebras, Graphcore, Groq
    ├── edge.cyr             Qualcomm, Samsung, MediaTek
    ├── interconnect.cyr     InfiniBand, NVLink, NVSwitch, XGMI, ICI
    ├── bandwidth.cyr        Memory bandwidth probing
    ├── pcie.cyr             PCIe link speed
    ├── numa.cyr             NUMA topology
    ├── disk.cyr             Storage device detection
    ├── environment.cyr      Runtime (Docker, K8s, cloud provider)
    ├── platform.cyr         sysfs/procfs helpers
    ├── command.cyr          Safe subprocess execution
    └── amd_xdna.cyr         AMD XDNA NPU
```

## Core Concepts

### Detection Modes

| Mode | Function | Use case |
|------|----------|----------|
| Synchronous | `registry_detect()` | Simple, single-threaded |
| Threaded | `registry_detect_threaded()` | CLI backends run in parallel threads |
| Cached | `cached_get(c)` | Long-running services, configurable TTL |
| Lazy | `lazy_by_family(lr, FAMILY_GPU)` | Probe only what you need |

### Sharding Strategies

| Strategy | When used |
|----------|----------|
| None | Model fits on a single device |
| Tensor Parallel | NVSwitch-connected GPUs or high-bandwidth interconnect |
| Pipeline Parallel | Multiple GPUs or AI ASICs |
| Data Parallel | Replicas for throughput |

### Training Memory Estimation

8 methods: full fine-tune, LoRA, QLoRA (4/8-bit), prefix tuning, DPO, RLHF,
distillation. Per-component breakdown (model, optimizer, activations).
Device-aware: GPU, TPU, Gaudi each have tuned multipliers.

## How Detection Works

All detection is best-effort and non-destructive:

1. **sysfs probing** — reads `/sys/class/drm`, `/sys/class/misc`, etc.
2. **`/dev` introspection** — checks for device nodes (`/dev/accel*`, `/dev/neuron*`)
3. **`$PATH` tool execution** — runs `nvidia-smi`, `hl-smi`, `vulkaninfo`, `neuron-ls` when present

If a tool or sysfs path is absent the accelerator simply isn't registered — no errors, no crashes.

## Development

```sh
cyrius deps                                    # Repopulate lib/ from version-pinned stdlib
cyrius build src/main.cyr build/ai-hwaccel    # Build (≈286 KB ELF, x86_64)
cyrius vet src/main.cyr                        # Include-graph audit
cyrius lint src/main.cyr                       # Static analysis
cyrius fmt src/main.cyr                        # Format check (diff against committed)

# Test suite — 11 units under tests/tcyr/, 518 assertions total
for t in tests/tcyr/*.tcyr; do
    cyrius build "$t" "/tmp/$(basename $t .tcyr)"
    "/tmp/$(basename $t .tcyr)"
done

./scripts/bench-history.sh                     # Run benchmarks, append to CSV
```

### Test units (`tests/tcyr/`)

| Unit | Subject |
|------|---------|
| `foundation_test.tcyr` | error codes, accel types, family classification |
| `profile_test.tcyr` | profile struct construction, throughput, rank |
| `registry_test.tcyr` | registry + detection builder + suggest_quant |
| `requirement_test.tcyr` | accelerator requirement matching |
| `gpu_parser_test.tcyr` | CUDA / Gaudi / Neuron output parsing |
| `backend_test.tcyr` | Apple / Intel / AMD XDNA / cloud ASIC / edge |
| `io_test.tcyr` | `which`, `run_tool`, CSV / sysfs / path helpers |
| `topology_test.tcyr` | interconnect / bandwidth / PCIe / storage / NVSwitch |
| `planning_test.tcyr` | sharding plans + training memory + model checks |
| `model_format_test.tcyr` | SafeTensors / GGUF / ONNX / PyTorch headers |
| `json_output_test.tcyr` | JSON serialization (registry, summary, profile) |

### Pattern: derived struct accessors

Every heap-allocated struct in the project uses `#derive(accessors)`. CI
gates raw `load64(<param> + N)` / `store64(<param> + N, …)` on these
structs outside their defining file. See `.github/workflows/ci.yml`'s
`Raw-offset guard` step.

```cyrius
#derive(accessors)
struct profile {
    accel_type; device_id; available; memory_bytes;
    compute_cap; driver_version; device_name;
    // ... 13 more fields
}

// Generated automatically:
//   profile_accel_type(p)         getter
//   profile_set_accel_type(p, v)  setter
```

## Documentation

| Document | Description |
|----------|-------------|
| [Rust vs Cyrius benchmarks](docs/benchmarks-rust-v-cyrius.md) | Binary size, LOC, performance comparison |
| [Architecture](docs/architecture/) | Module map, data flow |
| [Roadmap](docs/development/roadmap.md) | Development plan |
| [JSON schema](docs/schema.json) | Serialized registry format |
| [Changelog](CHANGELOG.md) | Release history |
| [Contributing](CONTRIBUTING.md) | How to contribute |

## Consumers

hoosh, daimon, Irfan, AgnosAI, murti, tazama

## License

Licensed under the [GNU General Public License v3.0](LICENSE).
