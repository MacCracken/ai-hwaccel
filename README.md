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
| Binary size | **236 KB** (x86_64 ELF, `CYRIUS_DCE=1`) |
| Compiler | Cyrius cycc 6.6.6 |
| Tests | 950 assertions (15 test units) |
| Fuzz harnesses | 6 |
| Dependencies | **0** third-party (bayan's JSON sublib: first-party, optional) |
| Hardware families | 18 |
| Derived structs | 16 (`#derive(accessors)`) |

## Supported Hardware

| Family | Variants | Detection method |
|--------|----------|------------------|
| NVIDIA CUDA | GeForce, Tesla, A100, H100, GH200, ... | `nvidia-smi` on `$PATH` (Linux) |
| AMD ROCm | MI250, MI300, RX 7900 | `/sys/class/drm` sysfs |
| Apple Metal | M-series GPU cores | `sysctl` (`system_profiler` fallback) |
| Apple ANE | Neural Engine | `sysctl` (`system_profiler` fallback) |
| Intel NPU | Meteor Lake+ | `/sys/class/misc/intel_npu` |
| AMD XDNA | Ryzen AI NPU | `/sys/class/accel/*/device/driver` |
| Google TPU | v4, v5e, v5p | `/sys/class/accel` (driver, `tpu_version`) |
| Intel Gaudi | Gaudi 2, Gaudi 3 (Habana HPU) | `hl-smi` on `$PATH` |
| AWS Inferentia | inf1, inf2 | `neuron-ls`, or `/dev/neuron*` + DMI product name |
| AWS Trainium | trn1 | `neuron-ls`, or `/dev/neuron*` + DMI product name |
| Intel oneAPI | Arc, Data Center Max | `xpu-smi` on `$PATH` |
| Qualcomm Cloud AI | AI 100 | `/dev/qaic_*` or `/sys/class/qaic` |
| Cerebras WSE | Wafer-Scale Engine | `cerebras_cli`, or `/dev/cerebras*` |
| Graphcore IPU | IPU-POD | `gc-info`, or `/dev/ipu*` |
| Groq LPU | Language Processing Unit | `/dev/groq*` |
| Samsung NPU | Exynos NPU | `/sys/class/misc/samsung_npu` |
| MediaTek APU | Dimensity APU | `/sys/class/misc/mtk_apu` |
| Vulkan Compute | Any Vulkan 1.1+ device | `vulkaninfo` on `$PATH`, a sysfs scan without it |
| Windows GPU | Any hardware display adapter | DXGI adapter enumeration (`wmic` fallback) |
| CPU | Always present | `/proc/meminfo`, `sysctl hw.memsize` (macOS), `GlobalMemoryStatusEx` (Windows); 16 GiB fallback |

On Windows no vendor tool runs yet (the roadmap's `which()` item), so every
GPU there, NVIDIA included, is a Windows GPU profile.

## Quick Start

```sh
# Build
cyrius build src/main.cyr build/ai-hwaccel

# Run
ai-hwaccel                  # Full registry JSON
ai-hwaccel --summary        # Compact summary JSON
ai-hwaccel --table          # Human-readable table
ai-hwaccel --plan 70B       # Sharding plan JSON (--quant int4, ...)
ai-hwaccel --train 7B --method lora   # Training memory JSON
ai-hwaccel --cost 70B       # Cloud instance recommendation (--json for JSON)
ai-hwaccel --version        # Print version
ai-hwaccel -v               # Log to stderr at debug level (-vv trace, -q silent)
```

Every JSON output is described by [`docs/schema.json`](docs/schema.json).

### From Python

```sh
pip install ai-hwaccel
```

The package wraps the binary and its JSON in typed dataclasses; see
[`bindings/python/README.md`](bindings/python/README.md).

### Using as a library

`cyrius distlib` bundles every non-CLI module into `dist/ai-hwaccel.cyr`,
a single self-contained file consumers pull via `cyrius deps`. The bundle
includes every detection backend, the registry/profile surface, the
sharding planner, the cost model, the training memory estimator, the
model-format header parser, and the JSON serializer.
Only `src/main.cyr` (CLI argv parsing) is excluded. Library consumers
that don't need JSON output and build with `CYRIUS_DCE=1` get the
serializer removed. `profile_from_json_str` needs bayan's JSON sublib,
which the consumer supplies (it is optional here).

Wire it from a consumer's `cyrius.cyml`:

```toml
[deps.ai-hwaccel]
git = "https://github.com/MacCracken/ai-hwaccel.git"
tag = "2.4.0"
modules = ["dist/ai-hwaccel.cyr"]
```

`cyrius deps` clones at the tag and drops the bundle into the
consumer's `lib/`. Include it like any other dep:

```cyrius
include "lib/ai-hwaccel.cyr"

# Full detection — sysfs probes plus vendor CLIs (nvidia-smi, hl-smi,
# neuron-ls, xpu-smi, cerebras_cli, gc-info) for the six EXEC backends,
# vulkaninfo for Vulkan, and `nvidia-smi nvlink -s` for the interconnect
# post-pass (InfiniBand, NVSwitch and XGMI come from sysfs).
var r = registry_detect();
# ... reg_profiles(r), reg_count(r), ...
```

#### No-exec entry point

Consumers with a no-subprocess contract — `mihi`'s probe surface, any
read-only system-info library — call `registry_detect_no_exec()`
instead. It masks off the six exec-shelling backends (CUDA, Gaudi,
Neuron, Intel oneAPI, Cerebras, Graphcore) and skips the
`detect_interconnects` post-pass. The eleven native backends still run,
along with the sysfs post-passes: ROCm, Intel NPU, AMD XDNA, TPU,
Qualcomm, Groq, Samsung NPU, MediaTek APU, Windows (DXGI), Apple
(sysctl on macOS, the device tree on Asahi Linux) and Vulkan (a sysfs
scan of the DRM cards, which leaves AMD cards to ROCm). Their `wmic` /
`system_profiler` / `vulkaninfo` runs do not happen.

```cyrius
include "lib/ai-hwaccel.cyr"

# No subprocess (sysfs, syscalls, DXGI, sysctl) — safe to call from
# probe contexts that forbid spawning processes (e.g. mihi).
var r = registry_detect_no_exec();
```

The classification lives in `backend_uses_exec(b)` in `src/types.cyr`.
To compose your own mask and stay spawn-free, pass it to
`registry_detect_with_opts(mask, 0)`, for example
`registry_detect_with_opts(builder_without(builder_no_exec(), BACKEND_GROQ), 0)`.
`registry_detect_with(mask)` allows exec whatever the mask: the Vulkan
backend runs `vulkaninfo`, the Apple and Windows fallbacks may run
`system_profiler` and `wmic`, and the interconnect post-pass runs
`nvidia-smi`.

## Architecture

```
src/
├── main.cyr                CLI entry point
├── types.cyr               AcceleratorType (20 variants, CPU included), AcceleratorFamily
├── profile.cyr             Device profile (memory, capabilities, throughput)
├── registry.cyr            AcceleratorRegistry, DetectBuilder (bitmask), duplicate-device pass
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
└── detect/                 Hardware detection (20 modules)
    ├── cuda.cyr             NVIDIA via nvidia-smi
    ├── rocm.cyr             AMD via sysfs
    ├── apple.cyr            Metal + ANE via sysctl
    ├── vulkan.cyr           Vulkan via vulkaninfo + sysfs scan
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
    ├── amd_xdna.cyr         AMD XDNA NPU
    └── windows.cyr          Windows GPUs via DXGI, RAM via GlobalMemoryStatusEx
```

## Core Concepts

### Detection Modes

| Mode | Function | Use case |
|------|----------|----------|
| Synchronous | `registry_detect()` | Simple, single-threaded |
| No-exec | `registry_detect_no_exec()` | No subprocess at all (see above) |
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
Device-aware: GPU, TPU, Gaudi and CPU each have tuned multipliers.

## How Detection Works

All detection is best-effort and non-destructive:

1. **sysfs probing** — reads `/sys/class/drm`, `/sys/class/misc`, etc.
2. **`/dev` introspection** — checks for device nodes (`/dev/neuron*`, `/dev/groq*`)
3. **Native OS APIs** — `sysctl` on macOS, DXGI and `GlobalMemoryStatusEx` on
   Windows
4. **`$PATH` tool execution** — runs `nvidia-smi`, `vulkaninfo`, `hl-smi`,
   `neuron-ls`, `xpu-smi`, `cerebras_cli` and `gc-info` when present, without
   a shell
5. **One profile per device** — a GPU that Vulkan and CUDA or
   ROCm both report is listed once, as the CUDA/ROCm profile, matched on its
   PCI vendor and device ID. On Apple Silicon, a Vulkan (MoltenVK, Asahi)
   view of the Metal GPU is dropped.

If a tool or sysfs path is absent the accelerator simply isn't registered, and
the `warnings` array names the missing tool. Detection itself doesn't fail.

## Development

```sh
cyrius lib sync                                # Repopulate lib/ from the version-pinned stdlib snapshot
cyrius deps                                    # Resolve non-stdlib [deps.*] entries (bayan) + transitive stdlib leaves
CYRIUS_DCE=1 cyrius build src/main.cyr build/ai-hwaccel   # Build (≈236 KB ELF, x86_64)
cyrius vet src/main.cyr                        # Include-graph audit
for f in src/*.cyr src/detect/*.cyr; do cyrius lint --strict "$f"; done # Static analysis (fails on a warning)
for f in src/*.cyr src/detect/*.cyr tests/tcyr/*.tcyr fuzz/*.fcyr benches/*.bcyr; do cyrius fmt "$f" --check; done
                                               # Format check (plain `cyrius fmt <file>` rewrites the file)

# Test suite — 15 units under tests/tcyr/, 950 assertions total. Run from the
# repository root: some tests read tests/fixtures/.
for t in tests/tcyr/*.tcyr; do
    CYRIUS_DCE=1 cyrius build "$t" "/tmp/$(basename $t .tcyr)"
    "/tmp/$(basename $t .tcyr)"
done

./scripts/bench-history.sh                     # Run benchmarks, append to CSV
```

### Test units (`tests/tcyr/`)

| Unit | Subject |
|------|---------|
| `foundation_test.tcyr` | error codes, accel types, family classification |
| `profile_test.tcyr` | profile struct construction, throughput, rank |
| `registry_test.tcyr` | registry + detection builder + suggest_quant + memory totals + the duplicate-device pass + every detection entry point, end to end |
| `requirement_test.tcyr` | accelerator requirement matching |
| `gpu_parser_test.tcyr` | CUDA / Gaudi / Neuron output parsing; Vulkan against real `vulkaninfo` captures (`tests/fixtures/vulkaninfo/`) |
| `backend_test.tcyr` | Apple / Intel / AMD XDNA / cloud ASIC / edge |
| `io_test.tcyr` | `which`, `run_tool`, CSV / sysfs / path helpers |
| `topology_test.tcyr` | interconnect / bandwidth / PCIe / storage / NVSwitch |
| `planning_test.tcyr` | sharding plans + training memory + model checks |
| `model_format_test.tcyr` | SafeTensors / GGUF / ONNX / PyTorch headers |
| `json_output_test.tcyr` | JSON serialization (registry, summary, profile) |
| `json_roundtrip_test.tcyr` | profile JSON round-trip (`profile_from_json`) |
| `model_catalog_test.tcyr` | `load_models` against the shipped `data/models.json`, `compatible_with_registry` |
| `windows_test.tcyr` | Windows DXGI adapter descriptors + `wmic` fallback parsers |
| `lazy_test.tcyr` | lazy registry: per-family queries vs full detection, no backend run twice |

### Pattern: derived struct accessors

Heap-allocated structs use `#derive(accessors)`. The two exceptions are small
records: a warning (`src/error.cyr`) and a thread argument
(`src/async_detect.cyr`). CI gates raw `load64(<param> + N)` /
`store64(<param> + N, …)` on the derived structs outside their defining file.
See `.github/workflows/ci.yml`'s `Raw-offset guard` step.

```cyrius
#derive(accessors)
struct profile {
    accel_type; device_id; available; memory_bytes;
    compute_cap; driver_version; device_name;
    // ... 15 more fields
}

// Generated automatically:
//   profile_accel_type(p)         getter
//   profile_set_accel_type(p, v)  setter
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture/overview.md) | Module map, detection flow |
| [JSON schema](docs/schema.json) | Every JSON output: registry, summary, plan, training, cost |
| [Production guide](docs/guides/production.md) | Deploying, logging, version compatibility |
| [Testing guide](docs/guides/testing.md) | Running the tests, hardware setup |
| [Framework integration](docs/guides/framework-integration.md) | Using the output from ML frameworks |
| [Troubleshooting](docs/troubleshooting.md) | Common detection problems |
| [Performance](docs/performance.md) | Benchmarks and cost of detection |
| [Python bindings](bindings/python/README.md) | `pip install ai-hwaccel` |
| [Roadmap](docs/development/roadmap.md) | Open work |
| [Changelog](CHANGELOG.md) | Release history |
| [Contributing](CONTRIBUTING.md) | How to contribute |
| [Rust vs Cyrius benchmarks](docs/benchmarks-rust-v-cyrius.md) | The 2.0.0 port, compared with Rust 1.2.0 |

## Consumers

hoosh, daimon, Irfan, AgnosAI, murti, tazama

## License

Licensed under the [GNU General Public License v3.0](LICENSE).
