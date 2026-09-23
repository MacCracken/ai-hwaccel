# Architecture Overview

ai-hwaccel is a flat Cyrius project: a CLI binary (`src/main.cyr`) over a
library (`dist/ai-hwaccel.cyr`, every other module) that detects AI hardware
accelerators, queries their capabilities, and plans model placement.

## Module Map

```
src/
  main.cyr               CLI: JSON (default), --summary, --table, --plan, --train, --cost
  types.cyr              AcceleratorType (20, CPU included), families, backends, exec classification
  profile.cyr            AcceleratorProfile: one device's memory, capabilities, PCI id, shared RAM
  registry.cyr           Detection entry points, builder masks, post-passes, duplicate-device pass, totals
  async_detect.cyr       registry_detect_threaded(): exec backends in threads
  lazy.cyr               LazyRegistry: probe a family on first query
  cache.cyr              CachedRegistry (TTL) + DiskCachedRegistry
  json_out.cyr           JSON out (registry, summary, plan, training) and profile_from_json
  system_io.cyr          Interconnects, storage, runtime environment
  quantization.cyr       FP32 / FP16 / BF16 / INT8 / INT4 (fixed-point x1000)
  plan.cyr               Sharding planner (none / tensor / pipeline / data parallel)
  training.cyr           Training memory estimation (8 methods, 4 targets)
  cost.cyr               Cloud instance pricing and recommendations (data/cloud_pricing.json)
  model.cyr              Model catalogue (data/models.json) and compatibility
  model_format.cyr       SafeTensors / GGUF / ONNX / PyTorch header detection
  requirement.cyr        Accelerator requirements for scheduling
  error.cyr              Warnings (the tool that was missing or failed)
  log.cyr                Structured logging to stderr (sakshi)
  units.cyr              Named constants and SCHEMA_VERSION
  detect/
    platform.cyr         sysfs / procfs helpers, total RAM (meminfo, sysctl, GlobalMemoryStatusEx)
    command.cyr          Tool execution (PATH lookup, timeout, empty environment), CSV / hex parsing
    cuda.cyr             NVIDIA via nvidia-smi CSV
    rocm.cyr             AMD via sysfs (/sys/class/drm, amdgpu)
    apple.cyr            Metal GPU + Neural Engine via sysctl (system_profiler fallback), Asahi device tree
    vulkan.cyr           Vulkan via vulkaninfo, or a sysfs scan of the DRM cards
    windows.cyr          Windows GPUs via DXGI adapter enumeration (wmic fallback)
    tpu.cyr              Google TPU via sysfs (/sys/class/accel)
    gaudi.cyr            Intel Gaudi via hl-smi CSV
    neuron.cyr           AWS Neuron via neuron-ls JSON, or /dev/neuron*
    intel.cyr            Intel NPU (sysfs) + oneAPI (xpu-smi)
    amd_xdna.cyr         AMD XDNA via sysfs
    cloud_asic.cyr       Cerebras WSE, Graphcore IPU, Groq LPU
    edge.cyr             Qualcomm AI 100, Samsung NPU, MediaTek APU
    bandwidth.cyr        Memory bandwidth estimation
    pcie.cyr             PCIe link bandwidth from sysfs
    numa.cyr             NUMA node per device
    interconnect.cyr     InfiniBand, RoCE, NVLink, NVSwitch, XGMI, ICI
    disk.cyr             Storage device classification
    environment.cyr      Docker, Kubernetes, cloud instance metadata
```

## Detection Flow

```
registry_detect()                      = registry_detect_with_opts(builder_all(), allow_exec 1)
registry_detect_no_exec()              = registry_detect_with_opts(builder_no_exec(), allow_exec 0)
registry_detect_with_opts(mask, allow_exec)
  1. CPU profile                       total RAM: /proc/meminfo, sysctl hw.memsize (macOS),
                                       GlobalMemoryStatusEx (Windows); 16 GiB if all fail
  2. each backend in the mask, in turn  sysfs / syscalls / DXGI / sysctl, and vendor
                                       tools only when allow_exec is 1
  3. registry_post_passes:
     a. duplicate devices              profiles_dedup: a Vulkan profile with a CUDA or
                                       ROCm profile's PCI vendor:device ID, or a Vulkan
                                       iGPU next to Apple's Metal GPU, is dropped; the
                                       survivor keeps its own memory and fields
     b. bandwidth enrichment           clock x bus width, or an estimate
     c. PCIe enrichment                sysfs link speed x width
     d. NUMA enrichment                sysfs numa_node
     e. interconnect detection         InfiniBand, NVLink, XGMI, ... (allow_exec 1 only)
     f. storage detection              NVMe / SATA SSD / HDD
     g. environment detection          Docker, Kubernetes, cloud metadata
```

The other entry points share the backends and `registry_post_passes`:
`registry_detect_threaded()` runs the tool-based backends in threads and the
sysfs ones on the calling thread; `lazy_by_family()` probes one family's
backends on its first query and deduplicates as families accumulate;
`cached_get()` / `disk_cached_get()` reuse a registry until its TTL runs out.

Totals (`reg_total_memory`, `reg_total_accel_memory`) count system RAM once:
a profile's `shared_memory_bytes` is folded into one pool, the largest shared
part, and only memory of its own adds up.

## Data Flow

```
Detection -> registry -> Query / Plan
               |             |
               v             v
   registry_to_json()      reg_suggest_quant()
   registry_to_summary_json()   reg_plan_sharding()
   profile_from_json()     estimate_training_memory()
   DiskCachedRegistry      recommend_instances(), compatible_with_registry()
```

## Design Decisions

- **No vendor SDK dependencies.** Detection uses sysfs, `/dev`, OS APIs
  (DXGI, sysctl) and vendor CLI tools (ADR-001).
- **Backends are selected at run time** with a builder mask
  (`builder_with` / `builder_without`); every backend is compiled in. The
  compile-time flags in ADR-004 were never implemented.
- **Parallel detection is opt-in** (`registry_detect_threaded()`, ADR-003);
  `registry_detect()` runs the backends in turn.
- **Best-effort.** A missing tool or path becomes a warning; the CPU profile
  is always present.
- **No-exec contract.** `registry_detect_no_exec()` spawns nothing
  (`backend_uses_exec()` in `types.cyr` classifies the backends).
- **Fixed-point arithmetic** throughout (x1000 multipliers), no floats.

## Dependency Stack

```
ai-hwaccel
  Cyrius stdlib (vendored in lib/, pinned by cyrius.cyml)
  bayan-json (optional, first-party)   only for profile_from_json_str
```

No third-party dependencies, and no vendor SDK dependencies.
