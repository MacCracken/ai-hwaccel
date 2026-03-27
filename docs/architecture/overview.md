# Architecture Overview

ai-hwaccel is a flat Rust crate (library + CLI binary) that detects AI
hardware accelerators, queries their capabilities, and plans model
distribution across devices.

## Module Map

```
lib.rs                  Public re-exports
  hardware/             AcceleratorType, AcceleratorFamily, TpuVersion, etc.
  profile.rs            AcceleratorProfile — per-device capability snapshot
  registry.rs           AcceleratorRegistry + DetectBuilder
  detect/               Backend detection (one module per hardware family)
    mod.rs              Orchestrator: threading, enrichment pipeline
    command.rs           Safe subprocess execution (path resolution, timeout, env sanitization)
    platform.rs         PlatformProbe trait + LivePlatform + MockPlatform
    cuda.rs             NVIDIA via nvidia-smi CSV
    rocm.rs             AMD via sysfs (/sys/class/drm)
    apple.rs            Metal/ANE via system_profiler + sysctl
    vulkan.rs           Vulkan via vulkaninfo + sysfs fallback
    windows.rs          WMI via wmic/PowerShell
    tpu.rs              Google TPU via sysfs (/sys/class/accel)
    gaudi.rs            Intel Gaudi via hl-smi CSV
    neuron.rs           AWS Neuron via neuron-ls JSON
    intel_oneapi.rs     Intel Arc via xpu-smi CSV
    intel_npu.rs        Intel NPU via sysfs
    amd_xdna.rs         AMD XDNA via sysfs
    qualcomm.rs         Qualcomm AI 100 via sysfs
    cerebras.rs         Cerebras WSE via CLI + /dev
    graphcore.rs        Graphcore IPU via gc-info + /dev
    groq.rs             Groq LPU via /dev
    samsung_npu.rs      Samsung NPU via sysfs
    mediatek_apu.rs     MediaTek APU via sysfs
    bandwidth.rs        Memory bandwidth estimation (NVIDIA clock+bus width)
    interconnect.rs     NVLink/IB/XGMI/ICI detection
    pcie.rs             PCIe link speed from sysfs
    numa.rs             NUMA node affinity
    disk.rs             Storage device classification
    environment.rs      Container/cloud detection (Docker, k8s, AWS/GCP/Azure)
  quantization.rs       QuantizationLevel (FP32/FP16/BF16/INT8/INT4)
  plan.rs               Sharding planner (tensor/pipeline/data parallel)
  sharding.rs           ShardingPlan, ModelShard, ShardingStrategy
  training.rs           Training memory estimation (LoRA, QLoRA, DPO, RLHF, etc.)
  cost.rs               Cloud instance pricing + recommendations
  requirement.rs        AcceleratorRequirement for scheduling
  cache.rs              CachedRegistry + DiskCachedRegistry (TTL-based)
  lazy.rs               LazyRegistry (detect-on-first-query)
  system_io.rs          SystemIo, Interconnect, StorageDevice
  error.rs              DetectionError enum
  units.rs              Named constants (no magic numbers)
  ffi.rs                C-compatible FFI bindings
  async_detect.rs       Async detection via tokio
  fuzz_helpers.rs       Fuzz target entry points
main.rs                 CLI binary (table, JSON, watch, cost, profile modes)
```

## Detection Flow

```
AcceleratorRegistry::detect()
  -> DetectBuilder (all backends enabled)
  -> detect_with_builder()
       1. cpu_profile()                     Always: read /proc/meminfo or sysctl
       2. spawn backends in parallel        std::thread::scope for 2+ backends
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
            JSON/serde            suggest_quantization()
            from_json()           plan_sharding()
            DiskCachedRegistry    estimate_training_memory()
                                  recommend_instance()
```

## Design Decisions

- **No vendor SDK dependencies** — all detection via CLI tools + sysfs
  (see ADR-001)
- **Feature-gated backends** — consumers pull only what they need
  (see ADR-004)
- **Parallel detection** — thread::scope for 2+ backends
  (see ADR-003)
- **Best-effort** — errors become warnings, CPU always available
- **AcceleratorType is Copy** — zero-cost pass-by-value, device name on profile

## Dependency Stack

```
ai-hwaccel
  serde + serde_json      Serialization (required)
  tracing                 Structured logging (required)
  tracing-subscriber      CLI log formatting (optional: cli feature)
  tokio                   Async detection (optional: async-detect feature)
```

Zero compile-time SDK dependencies. 4 direct deps at minimum.
