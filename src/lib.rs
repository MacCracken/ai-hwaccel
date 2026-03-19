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
//! | AMD ROCm | MI250 / MI300 / RX 7900 | `rocm-smi` on PATH |
//! | Apple Metal | M1–M4 GPU | `/proc/device-tree/compatible` |
//! | Apple ANE | Neural Engine | `/proc/device-tree/compatible` |
//! | Intel NPU | Meteor Lake+ NPU | `/sys/class/misc/intel_npu` |
//! | AMD XDNA | Ryzen AI NPU | `/sys/class/accel/accel*/device/driver` → `amdxdna` |
//! | Google TPU | v4 / v5e / v5p | `/dev/accel*` + sysfs version |
//! | Intel Gaudi | Gaudi 2 / 3 (Habana HPU) | `hl-smi` on PATH |
//! | AWS Inferentia | inf1 / inf2 | `/dev/neuron*` |
//! | AWS Trainium | trn1 | `/dev/neuron*` + sysfs |
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

pub mod detect;
pub mod hardware;
pub mod plan;
pub mod profile;
pub mod quantization;
pub mod registry;
pub mod requirement;
pub mod sharding;
pub mod training;

pub use hardware::{
    AcceleratorFamily, AcceleratorType, GaudiGeneration, NeuronChipType, TpuVersion,
};
pub use profile::AcceleratorProfile;
pub use quantization::QuantizationLevel;
pub use registry::AcceleratorRegistry;
pub use requirement::AcceleratorRequirement;
pub use sharding::{ModelShard, ShardingPlan, ShardingStrategy};
pub use training::{estimate_training_memory, MemoryEstimate, TrainingMethod, TrainingTarget};

#[cfg(test)]
mod tests;
