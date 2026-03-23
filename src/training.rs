//! Training method types and memory estimation.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::requirement::AcceleratorRequirement;

/// Fine-tuning / training method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrainingMethod {
    #[default]
    FullFineTune,
    LoRA,
    QLoRA { bits: u8 },
    Prefix,
    DPO,
    RLHF,
    Distillation,
}

impl TrainingMethod {
    /// Preferred accelerator requirement for this training method.
    pub fn preferred_accelerator(&self) -> AcceleratorRequirement {
        match self {
            // LoRA/QLoRA benefit from custom CUDA kernels
            Self::LoRA | Self::QLoRA { .. } => AcceleratorRequirement::Gpu,
            // Full fine-tune, DPO, RLHF, distillation work well on GPU or TPU
            Self::FullFineTune | Self::DPO | Self::RLHF | Self::Distillation | Self::Prefix => {
                AcceleratorRequirement::GpuOrTpu
            }
        }
    }
}

impl fmt::Display for TrainingMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullFineTune => write!(f, "full"),
            Self::LoRA => write!(f, "lora"),
            Self::QLoRA { bits } => write!(f, "qlora-{}bit", bits),
            Self::Prefix => write!(f, "prefix"),
            Self::DPO => write!(f, "dpo"),
            Self::RLHF => write!(f, "rlhf"),
            Self::Distillation => write!(f, "distillation"),
        }
    }
}

/// Target accelerator family for training memory estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainingTarget {
    #[default]
    Gpu,
    Tpu,
    Gaudi,
    Cpu,
}

/// Estimated device memory breakdown for a training/fine-tuning job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    /// Model weights in GB.
    pub model_gb: f64,
    /// Optimizer states in GB.
    pub optimizer_gb: f64,
    /// Activations / KV cache in GB.
    pub activation_gb: f64,
    /// Total device memory needed in GB.
    pub total_gb: f64,
}

impl fmt::Display for MemoryEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Model: {:.1} GB, Optimizer: {:.1} GB, Activations: {:.1} GB (total: {:.1} GB)",
            self.model_gb, self.optimizer_gb, self.activation_gb, self.total_gb
        )
    }
}

/// Estimate device memory for a fine-tuning job on a specific accelerator family.
///
/// This is a heuristic approximation assuming:
/// - FP16/BF16 model weights (2 bytes per parameter)
/// - Adam optimizer states (2x model weights for full fine-tune)
/// - Batch size 1, moderate sequence length
///
/// Real memory usage depends on batch size, sequence length, gradient
/// checkpointing, and precision mixing. Use these estimates as lower-bound
/// guidance, not exact predictions.
///
/// # Examples
///
/// ```rust
/// use ai_hwaccel::{estimate_training_memory, TrainingMethod, TrainingTarget};
///
/// let est = estimate_training_memory(7000, TrainingMethod::LoRA, TrainingTarget::Gpu);
/// assert!(est.total_gb > 0.0);
/// assert!((est.model_gb + est.optimizer_gb + est.activation_gb - est.total_gb).abs() < 0.001);
/// ```
pub fn estimate_training_memory(
    model_params_millions: u64,
    method: TrainingMethod,
    target: TrainingTarget,
) -> MemoryEstimate {
    let base_gb = (model_params_millions as f64 * crate::units::PARAMS_PER_MILLION
        * crate::units::FP16_BYTES_PER_PARAM)
        / crate::units::BYTES_PER_GIB;

    match target {
        TrainingTarget::Tpu => estimate_tpu_training(base_gb, method),
        TrainingTarget::Gaudi => estimate_gaudi_training(base_gb, method),
        TrainingTarget::Gpu | TrainingTarget::Cpu => estimate_gpu_training(base_gb, method),
    }
}

fn estimate_gpu_training(base_gb: f64, method: TrainingMethod) -> MemoryEstimate {
    let (model, optimizer, activation) = match method {
        TrainingMethod::FullFineTune => (base_gb, base_gb * 2.0, base_gb * 1.0),
        TrainingMethod::LoRA => (base_gb, base_gb * 0.1, base_gb * 0.1),
        TrainingMethod::QLoRA { bits } => {
            let qf = if bits <= 4 { 0.25 } else { 0.5 };
            (base_gb * qf, base_gb * 0.1, base_gb * 0.1 * qf)
        }
        TrainingMethod::Prefix => (base_gb, base_gb * 0.05, base_gb * 0.05),
        TrainingMethod::DPO | TrainingMethod::RLHF => {
            // DPO/RLHF: two model copies + optimizer
            (base_gb * 2.0, base_gb * 2.0, base_gb * 1.5)
        }
        TrainingMethod::Distillation => {
            // Teacher + student
            (base_gb * 1.5, base_gb * 1.0, base_gb * 0.8)
        }
    };
    MemoryEstimate {
        model_gb: model,
        optimizer_gb: optimizer,
        activation_gb: activation,
        total_gb: model + optimizer + activation,
    }
}

fn estimate_tpu_training(base_gb: f64, method: TrainingMethod) -> MemoryEstimate {
    // TPU: BF16 native, XLA fuses activations, BF16 optimizer states (1.5x not 2x)
    let (model, optimizer, activation) = match method {
        TrainingMethod::FullFineTune => (base_gb, base_gb * 1.5, base_gb * 0.8),
        TrainingMethod::LoRA => (base_gb, base_gb * 0.15, base_gb * 0.12),
        TrainingMethod::QLoRA { bits } => {
            let qf = if bits <= 4 { 0.4 } else { 0.6 };
            (base_gb * qf, base_gb * 0.15, base_gb * 0.12 * qf)
        }
        TrainingMethod::Prefix => (base_gb, base_gb * 0.05, base_gb * 0.05),
        TrainingMethod::DPO | TrainingMethod::RLHF => (base_gb * 2.0, base_gb * 1.5, base_gb * 1.2),
        TrainingMethod::Distillation => (base_gb * 1.5, base_gb * 0.8, base_gb * 0.6),
    };
    MemoryEstimate {
        model_gb: model,
        optimizer_gb: optimizer,
        activation_gb: activation,
        total_gb: model + optimizer + activation,
    }
}

fn estimate_gaudi_training(base_gb: f64, method: TrainingMethod) -> MemoryEstimate {
    // Gaudi: BF16 native like TPU, but with different memory controller.
    // Similar to TPU estimates but with slightly higher activation overhead.
    let (model, optimizer, activation) = match method {
        TrainingMethod::FullFineTune => (base_gb, base_gb * 1.5, base_gb * 0.9),
        TrainingMethod::LoRA => (base_gb, base_gb * 0.12, base_gb * 0.12),
        TrainingMethod::QLoRA { bits } => {
            let qf = if bits <= 4 { 0.35 } else { 0.55 };
            (base_gb * qf, base_gb * 0.12, base_gb * 0.12 * qf)
        }
        TrainingMethod::Prefix => (base_gb, base_gb * 0.05, base_gb * 0.06),
        TrainingMethod::DPO | TrainingMethod::RLHF => (base_gb * 2.0, base_gb * 1.5, base_gb * 1.3),
        TrainingMethod::Distillation => (base_gb * 1.5, base_gb * 0.9, base_gb * 0.7),
    };
    MemoryEstimate {
        model_gb: model,
        optimizer_gb: optimizer,
        activation_gb: activation,
        total_gb: model + optimizer + activation,
    }
}
