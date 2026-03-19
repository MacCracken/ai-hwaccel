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
//! ai-hwaccel = { version = "2026.3", default-features = false, features = ["cuda", "tpu"] }
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
//! ai-hwaccel = { version = "2026.3", default-features = false, features = ["cuda"] }
//! ```

pub mod detect;
pub mod error;
pub mod hardware;
pub mod plan;
pub mod profile;
pub mod quantization;
pub mod registry;
pub mod requirement;
pub mod sharding;
pub mod training;

pub use error::DetectionError;
pub use hardware::{
    AcceleratorFamily, AcceleratorType, GaudiGeneration, NeuronChipType, TpuVersion,
};
pub use profile::AcceleratorProfile;
pub use quantization::QuantizationLevel;
pub use registry::{AcceleratorRegistry, Backend, DetectBuilder, SCHEMA_VERSION};
pub use requirement::AcceleratorRequirement;
pub use sharding::{ModelShard, ShardingPlan, ShardingStrategy};
pub use training::{estimate_training_memory, MemoryEstimate, TrainingMethod, TrainingTarget};

#[cfg(test)]
mod tests;
