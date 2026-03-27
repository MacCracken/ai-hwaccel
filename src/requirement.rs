//! Workload accelerator requirements for scheduling integration.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

/// Hardware accelerator requirement for a workload (scheduling integration).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum AcceleratorRequirement {
    /// No accelerator needed — CPU is sufficient.
    #[default]
    None,
    /// Requires a GPU (CUDA, ROCm, Metal, Vulkan).
    Gpu,
    /// Requires a TPU with at least `min_chips` chips.
    Tpu { min_chips: u32 },
    /// Requires Intel Gaudi HPU.
    Gaudi,
    /// Requires AWS Neuron (Inferentia or Trainium).
    AwsNeuron,
    /// Either GPU or TPU will satisfy the requirement.
    GpuOrTpu,
    /// Any accelerator (GPU, TPU, or AI ASIC) — not CPU-only.
    AnyAccelerator,
}

impl fmt::Display for AcceleratorRequirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Gpu => write!(f, "gpu"),
            Self::Tpu { min_chips } => write!(f, "tpu({}+ chips)", min_chips),
            Self::Gaudi => write!(f, "gaudi"),
            Self::AwsNeuron => write!(f, "aws-neuron"),
            Self::GpuOrTpu => write!(f, "gpu-or-tpu"),
            Self::AnyAccelerator => write!(f, "any-accelerator"),
        }
    }
}

impl AcceleratorRequirement {
    /// Check whether a given [`AcceleratorProfile`] satisfies this requirement.
    #[must_use]
    #[inline]
    pub fn satisfied_by(&self, profile: &AcceleratorProfile) -> bool {
        if !profile.available {
            return false;
        }
        match self {
            Self::None => true,
            Self::Gpu => profile.accelerator.is_gpu(),
            Self::Tpu { min_chips } => match &profile.accelerator {
                AcceleratorType::Tpu { chip_count, .. } => *chip_count >= *min_chips,
                _ => false,
            },
            Self::Gaudi => matches!(profile.accelerator, AcceleratorType::Gaudi { .. }),
            Self::AwsNeuron => matches!(profile.accelerator, AcceleratorType::AwsNeuron { .. }),
            Self::GpuOrTpu => profile.accelerator.is_gpu() || profile.accelerator.is_tpu(),
            Self::AnyAccelerator => !matches!(profile.accelerator, AcceleratorType::Cpu),
        }
    }
}
