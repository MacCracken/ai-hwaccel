# Architecture Overview

ai-hwaccel is a flat Cyrius project (CLI binary) that detects AI
hardware accelerators, queries their capabilities, and plans model
distribution across devices.

## Module Map

```
main.cyr                 CLI binary (table, JSON, watch, cost, profile modes)
  hardware/              AcceleratorType, AcceleratorFamily, TpuVersion, etc.
  profile.cyr            AcceleratorProfile — per-device capability snapshot
  registry.cyr           registry_detect() + DetectBuilder
  detect/                Backend detection (one module per hardware family)
    mod.cyr              Orchestrator: threading, enrichment pipeline
    command.cyr          Safe subprocess execution (path resolution, timeout, env sanitization)
    platform.cyr         PlatformProbe trait + LivePlatform + MockPlatform
    cuda.cyr             NVIDIA via nvidia-smi CSV
    rocm.cyr             AMD via sysfs (/sys/class/drm)
    apple.cyr            Metal/ANE via system_profiler + sysctl
    vulkan.cyr           Vulkan via vulkaninfo + sysfs fallback
    windows.cyr          WMI via wmic/PowerShell
    tpu.cyr              Google TPU via sysfs (/sys/class/accel)
    gaudi.cyr            Intel Gaudi via hl-smi CSV
    neuron.cyr           AWS Neuron via neuron-ls JSON
    intel_oneapi.cyr     Intel Arc via xpu-smi CSV
    intel_npu.cyr        Intel NPU via sysfs
    amd_xdna.cyr         AMD XDNA via sysfs
    qualcomm.cyr         Qualcomm AI 100 via sysfs
    cerebras.cyr         Cerebras WSE via CLI + /dev
    graphcore.cyr        Graphcore IPU via gc-info + /dev
    groq.cyr             Groq LPU via /dev
    samsung_npu.cyr      Samsung NPU via sysfs
    mediatek_apu.cyr     MediaTek APU via sysfs
    bandwidth.cyr        Memory bandwidth estimation (NVIDIA clock+bus width)
    interconnect.cyr     NVLink/IB/XGMI/ICI detection
    pcie.cyr             PCIe link speed from sysfs
    numa.cyr             NUMA node affinity
    disk.cyr             Storage device classification
    environment.cyr      Container/cloud detection (Docker, k8s, AWS/GCP/Azure)
  quantization.cyr       QuantizationLevel (FP32/FP16/BF16/INT8/INT4)
  plan.cyr               Sharding planner (tensor/pipeline/data parallel)
  sharding.cyr           ShardingPlan, ModelShard, ShardingStrategy
  training.cyr           Training memory estimation (LoRA, QLoRA, DPO, RLHF, etc.)
  cost.cyr               Cloud instance pricing + recommendations
  requirement.cyr        AcceleratorRequirement for scheduling
  cache.cyr              CachedRegistry + DiskCachedRegistry (TTL-based)
  lazy.cyr               LazyRegistry (detect-on-first-query)
  system_io.cyr          SystemIo, Interconnect, StorageDevice
  error.cyr              DetectionError enum
  units.cyr              Named constants (no magic numbers)
  ffi.cyr                C-compatible FFI bindings
  thread.cyr             Threaded detection for parallel backends
  fuzz_helpers.cyr       Fuzz target entry points
```

## Detection Flow

```
registry_detect()
  -> DetectBuilder (all backends enabled)
  -> detect_with_builder()
       1. cpu_profile()                     Always: read /proc/meminfo or sysctl
       2. spawn backends in parallel        thread.cyr for 2+ backends
          each: run_tool() or read sysfs -> parse -> Vec<AcceleratorProfile>
       3. collect profiles + warnings
       4. post-pass enrichment:
          a. vulkan sysfs fallback          If no GPU found via CLI
          b. dedup vulkan vs cuda/rocm      Remove Vulkan if dedicated driver found
          c. bandwidth enrichment           nvidia-smi clock -> BW estimate
          d. PCIe enrichment                sysfs link speed/width
          e. NUMA enrichment                sysfs numa_node per PCI address
          f. interconnect detection         InfiniBand, NVLink, XGMI
          g. storage detection              NVMe/SATA/HDD classification
          h. environment detection          Docker, k8s, cloud instance metadata
       5. build AcceleratorRegistry
```

## Data Flow

```
Detection -> AcceleratorRegistry -> Query/Plan
                |                      |
                v                      v
            JSON (str_builder)    suggest_quantization()
            from_json()           plan_sharding()
            DiskCachedRegistry    estimate_training_memory()
                                  recommend_instance()
```

## Design Decisions

- **No vendor SDK dependencies** — all detection via CLI tools + sysfs
  (see ADR-001)
- **Compile-time backend selection** — `#ifdef`/`-D` flags control which backends are compiled
  (see ADR-004)
- **Parallel detection** — thread.cyr for 2+ backends
  (see ADR-003)
- **Best-effort** — errors become warnings, CPU always available
- **AcceleratorType is Copy** — zero-cost pass-by-value, device name on profile

## Dependency Stack

```
ai-hwaccel
  (zero external dependencies)
  JSON via str_builder     Manual serialization
  thread.cyr               Parallel detection
```

Zero external dependencies. Zero vendor SDK dependencies.
