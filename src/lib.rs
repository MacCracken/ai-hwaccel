//! Universal AI hardware accelerator detection and workload planning.
//!
//! `ai-hwaccel` provides a unified view of every AI-capable accelerator on the
//! system — GPUs, TPUs, NPUs, and cloud inference chips — through a single
//! detection call. It answers three questions:
//!
//! 1. **What hardware is available?** ([`AcceleratorRegistry::detect`])
//! 2. **What quantisation fits?** ([`AcceleratorRegistry::suggest_quantization`])
//! 3. **How should the model be distributed?** ([`AcceleratorRegistry::plan_sharding`])
//!
//! Zero runtime dependencies beyond `serde` (for serialisation) and `tracing`
//! (for debug diagnostics). Detection probes sysfs, `/dev`, and `$PATH` tools —
//! no vendor SDKs required at compile time.
//!
//! # Supported Hardware
//!
//! | Family | Variant | Detection method |
//! |--------|---------|------------------|
//! | NVIDIA CUDA | GeForce / Tesla / A100 / H100 | `nvidia-smi` on PATH |
//! | AMD ROCm | MI250 / MI300 / RX 7900 | sysfs `/sys/class/drm` |
//! | Apple Metal | M1–M4 GPU | `system_profiler` or device-tree |
//! | Apple ANE | Neural Engine | `system_profiler` or device-tree |
//! | Intel NPU | Meteor Lake+ NPU | `/sys/class/misc/intel_npu` |
//! | AMD XDNA | Ryzen AI NPU | `/sys/class/accel/accel*/device/driver` → `amdxdna` |
//! | Google TPU | v4 / v5e / v5p | `/dev/accel*` + sysfs version |
//! | Intel Gaudi | Gaudi 2 / 3 (Habana HPU) | `hl-smi` on PATH |
//! | AWS Inferentia | inf1 / inf2 | `neuron-ls` or `/dev/neuron*` |
//! | AWS Trainium | trn1 | `neuron-ls` or `/dev/neuron*` + sysfs |
//! | Qualcomm Cloud AI | AI 100 | `/dev/qaic_*` or `/sys/class/qaic` |
//! | Vulkan Compute | Any Vulkan 1.1+ device | `vulkaninfo` on PATH |
//! | CPU | Always present | `/proc/meminfo` or 16 GiB fallback |
//!
//! # Quick start
//!
//! ```rust,no_run
//! use ai_hwaccel::{AcceleratorRegistry, QuantizationLevel};
//!
//! let registry = AcceleratorRegistry::detect();
//! println!("Best device: {}", registry.best_available().unwrap());
//!
//! // What quantisation for a 7B-parameter model?
//! let quant = registry.suggest_quantization(7_000_000_000);
//! println!("Recommended: {quant}");
//!
//! // How to shard a 70B model at BF16?
//! let plan = registry.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16);
//! println!("Strategy: {}, est. {:.0} tok/s",
//!     plan.strategy,
//!     plan.estimated_tokens_per_sec.unwrap_or(0.0));
//! ```
//!
//! # Guide
//!
//! ## Step 1: Detect hardware
//!
//! Call [`AcceleratorRegistry::detect`] to probe the system. All backends
//! run in parallel and detection is best-effort — missing tools or drivers
//! are skipped, not fatal.
//!
//! ```rust
//! use ai_hwaccel::AcceleratorRegistry;
//!
//! let registry = AcceleratorRegistry::detect();
//! for profile in registry.all_profiles() {
//!     println!("{}", profile);
//! }
//! // Check for warnings (tool failures, parse errors, etc.)
//! for w in registry.warnings() {
//!     eprintln!("warning: {}", w);
//! }
//! ```
//!
//! Use [`DetectBuilder`] to control which backends run:
//!
//! ```rust,no_run
//! use ai_hwaccel::AcceleratorRegistry;
//!
//! let registry = AcceleratorRegistry::builder()
//!     .with_cuda()
//!     .with_tpu()
//!     .detect();
//! ```
//!
//! Or disable backends at compile time with cargo features:
//!
//! ```toml
//! [dependencies]
//! ai-hwaccel = { version = "1.1", default-features = false, features = ["cuda", "tpu"] }
//! ```
//!
//! ## Step 2: Query capabilities
//!
//! The registry provides several query methods:
//!
//! ```rust
//! use ai_hwaccel::{AcceleratorRegistry, AcceleratorProfile, AcceleratorFamily,
//!                  AcceleratorRequirement};
//!
//! let registry = AcceleratorRegistry::from_profiles(vec![
//!     AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
//!     AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024),
//! ]);
//!
//! // Best single device
//! let best = registry.best_available().unwrap();
//!
//! // Filter by family
//! let gpus = registry.by_family(AcceleratorFamily::Gpu);
//!
//! // Filter by workload requirement
//! let matches = registry.satisfying(&AcceleratorRequirement::Gpu);
//!
//! // Memory totals
//! let total = registry.total_memory();
//! let accel = registry.total_accelerator_memory();
//! ```
//!
//! ## Step 3: Plan model deployment
//!
//! Given a model's parameter count, the registry can suggest a quantisation
//! level and generate a sharding plan:
//!
//! ```rust
//! use ai_hwaccel::{AcceleratorRegistry, AcceleratorProfile, QuantizationLevel};
//!
//! let registry = AcceleratorRegistry::from_profiles(vec![
//!     AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
//!     AcceleratorProfile::cuda(0, 80 * 1024 * 1024 * 1024),
//!     AcceleratorProfile::cuda(1, 80 * 1024 * 1024 * 1024),
//! ]);
//!
//! // Suggest quantisation for available hardware
//! let quant = registry.suggest_quantization(70_000_000_000);
//!
//! // Generate a sharding plan
//! let plan = registry.plan_sharding(70_000_000_000, &quant);
//! print!("{}", plan); // human-readable summary
//! ```
//!
//! ## Step 4: Estimate training memory
//!
//! For fine-tuning workloads, estimate per-component memory usage:
//!
//! ```rust
//! use ai_hwaccel::{estimate_training_memory, TrainingMethod, TrainingTarget};
//!
//! let est = estimate_training_memory(7000, TrainingMethod::LoRA, TrainingTarget::Gpu);
//! println!("Model: {:.1} GB, Optimizer: {:.1} GB, Activations: {:.1} GB",
//!     est.model_gb, est.optimizer_gb, est.activation_gb);
//! println!("Total: {:.1} GB", est.total_gb);
//! ```
//!
//! ## Step 5: Inspect system I/O
//!
//! After detection, the registry includes system-level I/O information:
//!
//! ```rust,no_run
//! use ai_hwaccel::AcceleratorRegistry;
//!
//! let registry = AcceleratorRegistry::detect();
//! let sio = registry.system_io();
//!
//! for ic in &sio.interconnects {
//!     println!("{} ({}) — {:.1} GB/s", ic.name, ic.kind, ic.bandwidth_gbps);
//! }
//! for dev in &sio.storage {
//!     println!("{} ({}) — {:.1} GB/s", dev.name, dev.kind, dev.bandwidth_gbps);
//! }
//!
//! // Estimate how long to load a 100 GB dataset from local storage
//! if let Some(secs) = sio.estimate_ingestion_secs(100 * 1024 * 1024 * 1024) {
//!     println!("Estimated ingestion time: {:.0}s", secs);
//! }
//! ```
//!
//! # Error handling
//!
//! Detection is best-effort. Errors are collected as warnings, not panics:
//!
//! ```rust,no_run
//! use ai_hwaccel::{AcceleratorRegistry, DetectionError};
//!
//! let registry = AcceleratorRegistry::detect();
//! for w in registry.warnings() {
//!     match w {
//!         DetectionError::ToolNotFound { tool } => {
//!             // Tool not installed — expected on systems without that hardware.
//!             eprintln!("skipped: {} not found", tool);
//!         }
//!         DetectionError::Timeout { tool, timeout_secs } => {
//!             // Tool hung — may want to retry with a longer timeout.
//!             eprintln!("{} timed out after {:.0}s", tool, timeout_secs);
//!         }
//!         DetectionError::ToolFailed { tool, exit_code, stderr } => {
//!             eprintln!("{} failed (exit {}): {}", tool,
//!                 exit_code.unwrap_or(-1), stderr);
//!         }
//!         _ => eprintln!("warning: {}", w),
//!     }
//! }
//! ```
//!
//! # Custom backends
//!
//! Build profiles manually and add them to a registry for hardware that
//! isn't auto-detected:
//!
//! ```rust
//! use ai_hwaccel::{AcceleratorProfile, AcceleratorRegistry, AcceleratorType};
//!
//! let mut registry = AcceleratorRegistry::detect();
//!
//! // Add a device from an external detection system
//! let mut custom = AcceleratorProfile::cuda(4, 80 * 1024 * 1024 * 1024);
//! custom.compute_capability = Some("9.0".into());
//! custom.memory_bandwidth_gbps = Some(3350.0);
//! registry.add_profile(custom);
//! ```
//!
//! # Serde integration
//!
//! The registry and all sub-types implement `Serialize`/`Deserialize`.
//! Use [`CachedRegistry`] for disk persistence with TTL-based invalidation:
//!
//! ```rust,no_run
//! use ai_hwaccel::CachedRegistry;
//! use std::time::Duration;
//!
//! let cache = CachedRegistry::new(Duration::from_secs(300));
//! let registry = cache.get(); // detects on first call, caches for 5 min
//! let registry2 = cache.get(); // returns cached result
//! ```
//!
//! The [`SCHEMA_VERSION`] constant tracks the JSON schema. Bumps indicate
//! new fields or structural changes. Old JSON (lower version) can still be
//! deserialized — new fields use `#[serde(default)]`.
//!
//! # Cargo features
//!
//! Each hardware backend can be individually enabled or disabled:
//!
//! | Feature | Backend | Default |
//! |---------|---------|---------|
//! | `cuda` | NVIDIA CUDA | yes |
//! | `rocm` | AMD ROCm | yes |
//! | `apple` | Apple Metal + ANE | yes |
//! | `vulkan` | Vulkan Compute | yes |
//! | `intel-npu` | Intel NPU | yes |
//! | `amd-xdna` | AMD XDNA NPU | yes |
//! | `tpu` | Google TPU | yes |
//! | `gaudi` | Intel Gaudi | yes |
//! | `aws-neuron` | AWS Inferentia/Trainium | yes |
//! | `intel-oneapi` | Intel oneAPI | yes |
//! | `qualcomm` | Qualcomm Cloud AI | yes |
//! | `all-backends` | All of the above | yes |
//!
//! To include only specific backends:
//!
//! ```toml
//! [dependencies]
//! ai-hwaccel = { version = "1.1", default-features = false, features = ["cuda"] }
//! ```

mod async_detect;
pub mod cache;
pub mod cost;
pub mod detect;
pub mod error;
pub mod ffi;
#[cfg(feature = "fuzz")]
#[doc(hidden)]
pub mod fuzz_helpers;
pub mod hardware;
pub mod lazy;
pub mod plan;
pub mod profile;
pub mod quantization;
pub mod registry;
pub mod requirement;
pub mod sharding;
pub mod system_io;
pub mod training;
pub mod units;

pub use cache::{CachedRegistry, DiskCachedRegistry};
pub use cost::{CloudGpuInstance, CloudProvider, InstanceRecommendation};
pub use detect::TimedDetection;
pub use error::DetectionError;
pub use hardware::{
    AcceleratorFamily, AcceleratorType, GaudiGeneration, NeuronChipType, TpuVersion,
};
pub use lazy::LazyRegistry;
pub use profile::AcceleratorProfile;
pub use quantization::QuantizationLevel;
pub use registry::{AcceleratorRegistry, Backend, DetectBuilder, SCHEMA_VERSION};
pub use requirement::AcceleratorRequirement;
pub use sharding::{ModelShard, ShardingPlan, ShardingStrategy};
pub use system_io::{
    CloudInstanceMeta, Interconnect, InterconnectKind, RuntimeEnvironment, StorageDevice,
    StorageKind, SystemIo,
};
pub use training::{MemoryEstimate, TrainingMethod, TrainingTarget, estimate_training_memory};

#[cfg(test)]
mod tests;
