//! Hardware accelerator type definitions.
//!
//! Each supported hardware family has its own submodule for generation/version
//! types. The top-level [`AcceleratorType`] enum ties them together.

mod gaudi;
mod neuron;
mod tpu;

pub use gaudi::GaudiGeneration;
pub use neuron::NeuronChipType;
pub use tpu::TpuVersion;

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// AcceleratorFamily
// ---------------------------------------------------------------------------

/// Broad device family categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AcceleratorFamily {
    Cpu,
    Gpu,
    Npu,
    Tpu,
    AiAsic,
}

impl fmt::Display for AcceleratorFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => write!(f, "GPU"),
            Self::Npu => write!(f, "NPU"),
            Self::Tpu => write!(f, "TPU"),
            Self::AiAsic => write!(f, "AI ASIC"),
        }
    }
}

// ---------------------------------------------------------------------------
// AcceleratorType
// ---------------------------------------------------------------------------

/// Every supported hardware accelerator family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AcceleratorType {
    /// Default CPU execution — always available.
    Cpu,
    /// NVIDIA CUDA GPU (GeForce, Tesla, A100, H100, ...).
    CudaGpu { device_id: u32 },
    /// AMD ROCm GPU (MI250, MI300, RX 7900, ...).
    RocmGpu { device_id: u32 },
    /// Apple Metal GPU (M1–M4).
    MetalGpu,
    /// Vulkan compute device.
    VulkanGpu { device_id: u32 },
    /// Intel Neural Processing Unit (Meteor Lake+).
    IntelNpu,
    /// AMD XDNA / Ryzen AI NPU.
    AmdXdnaNpu { device_id: u32 },
    /// Apple Neural Engine (ANE).
    AppleNpu,
    /// Google TPU (Tensor Processing Unit).
    Tpu {
        device_id: u32,
        chip_count: u32,
        version: TpuVersion,
    },
    /// Intel Gaudi / Habana Labs HPU.
    Gaudi {
        device_id: u32,
        generation: GaudiGeneration,
    },
    /// AWS Inferentia or Trainium (Neuron SDK).
    AwsNeuron {
        device_id: u32,
        chip_type: NeuronChipType,
        core_count: u32,
    },
    /// Qualcomm Cloud AI 100.
    QualcommAi100 { device_id: u32 },
    /// Intel Arc / Data Center GPU Max (oneAPI / SYCL).
    IntelOneApi { device_id: u32 },
    /// Cerebras Wafer-Scale Engine (WSE-2/WSE-3).
    CerebrasWse { device_id: u32 },
    /// Graphcore Intelligence Processing Unit (IPU).
    GraphcoreIpu { device_id: u32 },
    /// Groq Language Processing Unit (LPU).
    GroqLpu { device_id: u32 },
    /// Samsung Exynos AI NPU.
    SamsungNpu { device_id: u32 },
    /// MediaTek APU (MDLA-based).
    MediaTekApu { device_id: u32 },
}

impl AcceleratorType {
    /// Returns `true` for any GPU variant.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(
            self,
            Self::CudaGpu { .. }
                | Self::RocmGpu { .. }
                | Self::MetalGpu
                | Self::VulkanGpu { .. }
                | Self::IntelOneApi { .. }
        )
    }

    /// Returns `true` for any NPU variant.
    #[inline]
    pub fn is_npu(&self) -> bool {
        matches!(
            self,
            Self::IntelNpu
                | Self::AppleNpu
                | Self::AmdXdnaNpu { .. }
                | Self::SamsungNpu { .. }
                | Self::MediaTekApu { .. }
        )
    }

    /// Returns `true` for Google TPU.
    #[inline]
    pub fn is_tpu(&self) -> bool {
        matches!(self, Self::Tpu { .. })
    }

    /// Returns `true` for any cloud/data-centre AI ASIC (Gaudi, Neuron, Qualcomm, Cerebras, Graphcore, Groq).
    #[inline]
    pub fn is_ai_asic(&self) -> bool {
        matches!(
            self,
            Self::Gaudi { .. }
                | Self::AwsNeuron { .. }
                | Self::QualcommAi100 { .. }
                | Self::CerebrasWse { .. }
                | Self::GraphcoreIpu { .. }
                | Self::GroqLpu { .. }
        )
    }

    /// Broad category for the device family.
    #[inline]
    pub fn family(&self) -> AcceleratorFamily {
        match self {
            Self::Cpu => AcceleratorFamily::Cpu,
            Self::CudaGpu { .. }
            | Self::RocmGpu { .. }
            | Self::MetalGpu
            | Self::VulkanGpu { .. }
            | Self::IntelOneApi { .. } => AcceleratorFamily::Gpu,
            Self::IntelNpu
            | Self::AppleNpu
            | Self::AmdXdnaNpu { .. }
            | Self::SamsungNpu { .. }
            | Self::MediaTekApu { .. } => AcceleratorFamily::Npu,
            Self::Tpu { .. } => AcceleratorFamily::Tpu,
            Self::Gaudi { .. }
            | Self::AwsNeuron { .. }
            | Self::QualcommAi100 { .. }
            | Self::CerebrasWse { .. }
            | Self::GraphcoreIpu { .. }
            | Self::GroqLpu { .. } => AcceleratorFamily::AiAsic,
        }
    }

    /// Relative throughput multiplier vs CPU (rough inference estimate).
    #[inline]
    pub fn throughput_multiplier(&self) -> f64 {
        match self {
            Self::Cpu => 1.0,
            Self::CudaGpu { .. } => 20.0,
            Self::RocmGpu { .. } => 15.0,
            Self::MetalGpu => 12.0,
            Self::VulkanGpu { .. } => 10.0,
            Self::IntelNpu => 8.0,
            Self::AppleNpu => 10.0,
            Self::AmdXdnaNpu { .. } => 7.0,
            Self::Tpu { version, .. } => match version {
                TpuVersion::V4 => 25.0,
                TpuVersion::V5e => 18.0,
                TpuVersion::V5p => 35.0,
            },
            Self::Gaudi { generation, .. } => match generation {
                GaudiGeneration::Gaudi2 => 22.0,
                GaudiGeneration::Gaudi3 => 30.0,
            },
            Self::AwsNeuron {
                chip_type,
                core_count,
                ..
            } => {
                let base = match chip_type {
                    NeuronChipType::Inferentia => 15.0,
                    NeuronChipType::Trainium => 24.0,
                };
                base * (*core_count as f64 / 2.0).max(1.0)
            }
            Self::QualcommAi100 { .. } => 14.0,
            Self::IntelOneApi { .. } => 13.0,
            Self::CerebrasWse { .. } => 50.0,
            Self::GraphcoreIpu { .. } => 16.0,
            Self::GroqLpu { .. } => 30.0,
            Self::SamsungNpu { .. } => 6.0,
            Self::MediaTekApu { .. } => 5.0,
        }
    }

    /// Relative training throughput multiplier vs CPU.
    /// Differs from inference throughput for some devices (e.g. Inferentia is
    /// inference-only, Trainium excels at training).
    #[inline]
    pub fn training_multiplier(&self) -> f64 {
        match self {
            Self::Cpu => 1.0,
            Self::CudaGpu { .. } => 20.0,
            Self::RocmGpu { .. } => 15.0,
            Self::MetalGpu => 10.0,
            Self::VulkanGpu { .. } => 6.0, // Vulkan compute shaders, limited training support
            Self::IntelNpu
            | Self::AppleNpu
            | Self::AmdXdnaNpu { .. }
            | Self::SamsungNpu { .. }
            | Self::MediaTekApu { .. } => 0.0, // NPUs: inference only
            Self::Tpu { version, .. } => match version {
                TpuVersion::V4 => 28.0,
                TpuVersion::V5e => 20.0,
                TpuVersion::V5p => 40.0,
            },
            Self::Gaudi { generation, .. } => match generation {
                GaudiGeneration::Gaudi2 => 25.0,
                GaudiGeneration::Gaudi3 => 35.0,
            },
            Self::AwsNeuron {
                chip_type,
                core_count,
                ..
            } => match chip_type {
                NeuronChipType::Inferentia => 0.0, // inference only
                NeuronChipType::Trainium => 26.0 * (*core_count as f64 / 2.0).max(1.0),
            },
            Self::QualcommAi100 { .. } => 0.0, // inference only
            Self::IntelOneApi { .. } => 10.0,  // oneAPI training support via SYCL
            Self::CerebrasWse { .. } => 60.0,
            Self::GraphcoreIpu { .. } => 18.0,
            Self::GroqLpu { .. } => 0.0, // inference only
        }
    }

    /// Whether this device supports training workloads at all.
    #[inline]
    pub fn supports_training(&self) -> bool {
        self.training_multiplier() > 0.0
    }

    /// Priority rank for [`crate::AcceleratorRegistry::best_available`] (higher = preferred).
    pub(crate) fn rank(&self) -> u32 {
        match self {
            Self::CerebrasWse { .. } => 85,
            Self::Tpu {
                version: TpuVersion::V5p,
                ..
            } => 80,
            Self::Gaudi {
                generation: GaudiGeneration::Gaudi3,
                ..
            } => 75,
            Self::Tpu {
                version: TpuVersion::V4,
                ..
            } => 70,
            Self::Gaudi {
                generation: GaudiGeneration::Gaudi2,
                ..
            } => 65,
            Self::CudaGpu { .. } => 60,
            Self::AwsNeuron {
                chip_type: NeuronChipType::Trainium,
                ..
            } => 58,
            Self::Tpu {
                version: TpuVersion::V5e,
                ..
            } => 55,
            Self::GroqLpu { .. } => 55,
            Self::RocmGpu { .. } => 50,
            Self::GraphcoreIpu { .. } => 48,
            Self::AwsNeuron {
                chip_type: NeuronChipType::Inferentia,
                ..
            } => 45,
            Self::IntelOneApi { .. } => 42,
            Self::MetalGpu => 40,
            Self::QualcommAi100 { .. } => 38,
            Self::VulkanGpu { .. } => 35,
            Self::AppleNpu => 30,
            Self::AmdXdnaNpu { .. } => 25,
            Self::IntelNpu => 20,
            Self::SamsungNpu { .. } => 18,
            Self::MediaTekApu { .. } => 15,
            Self::Cpu => 10,
        }
    }
}

impl fmt::Display for AcceleratorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::CudaGpu { device_id } => write!(f, "CUDA GPU (device {})", device_id),
            Self::RocmGpu { device_id } => write!(f, "ROCm GPU (device {})", device_id),
            Self::MetalGpu => write!(f, "Metal GPU"),
            Self::VulkanGpu { device_id } => write!(f, "Vulkan GPU (device {})", device_id),
            Self::IntelNpu => write!(f, "Intel NPU"),
            Self::AmdXdnaNpu { device_id } => write!(f, "AMD XDNA NPU (device {})", device_id),
            Self::AppleNpu => write!(f, "Apple Neural Engine"),
            Self::Tpu {
                device_id,
                chip_count,
                version,
            } => {
                write!(
                    f,
                    "TPU {} (device {}, {} chips)",
                    version, device_id, chip_count
                )
            }
            Self::Gaudi {
                device_id,
                generation,
            } => {
                write!(f, "Intel {} (device {})", generation, device_id)
            }
            Self::AwsNeuron {
                device_id,
                chip_type,
                core_count,
            } => {
                write!(
                    f,
                    "AWS {} (device {}, {} cores)",
                    chip_type, device_id, core_count
                )
            }
            Self::QualcommAi100 { device_id } => {
                write!(f, "Qualcomm Cloud AI 100 (device {})", device_id)
            }
            Self::IntelOneApi { device_id } => {
                write!(f, "Intel oneAPI GPU (device {})", device_id)
            }
            Self::CerebrasWse { device_id } => {
                write!(f, "Cerebras WSE (device {})", device_id)
            }
            Self::GraphcoreIpu { device_id } => {
                write!(f, "Graphcore IPU (device {})", device_id)
            }
            Self::GroqLpu { device_id } => {
                write!(f, "Groq LPU (device {})", device_id)
            }
            Self::SamsungNpu { device_id } => {
                write!(f, "Samsung NPU (device {})", device_id)
            }
            Self::MediaTekApu { device_id } => {
                write!(f, "MediaTek APU (device {})", device_id)
            }
        }
    }
}
