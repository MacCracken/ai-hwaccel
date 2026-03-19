//! Core types: accelerator families, quantisation levels, sharding strategies.

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TpuVersion
// ---------------------------------------------------------------------------

/// Google TPU generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TpuVersion {
    V4,
    V5e,
    V5p,
}

impl TpuVersion {
    /// HBM (High Bandwidth Memory) per chip in bytes.
    pub fn hbm_per_chip_bytes(&self) -> u64 {
        match self {
            Self::V4 => 32 * 1024 * 1024 * 1024,
            Self::V5e => 16 * 1024 * 1024 * 1024,
            Self::V5p => 95 * 1024 * 1024 * 1024,
        }
    }
}

impl fmt::Display for TpuVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V4 => write!(f, "v4"),
            Self::V5e => write!(f, "v5e"),
            Self::V5p => write!(f, "v5p"),
        }
    }
}

// ---------------------------------------------------------------------------
// GaudiGeneration
// ---------------------------------------------------------------------------

/// Intel Gaudi (Habana Labs HPU) generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GaudiGeneration {
    Gaudi2,
    Gaudi3,
}

impl GaudiGeneration {
    /// HBM per device in bytes.
    pub fn hbm_bytes(&self) -> u64 {
        match self {
            Self::Gaudi2 => 96 * 1024 * 1024 * 1024,
            Self::Gaudi3 => 128 * 1024 * 1024 * 1024,
        }
    }
}

impl fmt::Display for GaudiGeneration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gaudi2 => write!(f, "Gaudi2"),
            Self::Gaudi3 => write!(f, "Gaudi3"),
        }
    }
}

// ---------------------------------------------------------------------------
// NeuronChipType
// ---------------------------------------------------------------------------

/// AWS Neuron chip type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NeuronChipType {
    /// Inference-optimised (inf1, inf2).
    Inferentia,
    /// Training-optimised (trn1).
    Trainium,
}

impl NeuronChipType {
    /// HBM per NeuronCore in bytes.
    pub fn hbm_per_core_bytes(&self) -> u64 {
        match self {
            // inf2 NeuronCore-v2: 32 GB HBM per accelerator (2 cores share it)
            Self::Inferentia => 16 * 1024 * 1024 * 1024,
            // trn1 NeuronCore-v2: 32 GB HBM per accelerator
            Self::Trainium => 32 * 1024 * 1024 * 1024,
        }
    }
}

impl fmt::Display for NeuronChipType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inferentia => write!(f, "Inferentia"),
            Self::Trainium => write!(f, "Trainium"),
        }
    }
}

// ---------------------------------------------------------------------------
// AcceleratorType
// ---------------------------------------------------------------------------

/// Every supported hardware accelerator family.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    VulkanGpu { device_id: u32, device_name: String },
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
}

impl AcceleratorType {
    /// Returns `true` for any GPU variant.
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
    pub fn is_npu(&self) -> bool {
        matches!(self, Self::IntelNpu | Self::AppleNpu | Self::AmdXdnaNpu { .. })
    }

    /// Returns `true` for Google TPU.
    pub fn is_tpu(&self) -> bool {
        matches!(self, Self::Tpu { .. })
    }

    /// Returns `true` for any cloud/data-centre AI ASIC (Gaudi, Neuron, Qualcomm).
    pub fn is_ai_asic(&self) -> bool {
        matches!(
            self,
            Self::Gaudi { .. } | Self::AwsNeuron { .. } | Self::QualcommAi100 { .. }
        )
    }

    /// Broad category for the device family.
    pub fn family(&self) -> AcceleratorFamily {
        match self {
            Self::Cpu => AcceleratorFamily::Cpu,
            Self::CudaGpu { .. }
            | Self::RocmGpu { .. }
            | Self::MetalGpu
            | Self::VulkanGpu { .. }
            | Self::IntelOneApi { .. } => AcceleratorFamily::Gpu,
            Self::IntelNpu | Self::AppleNpu | Self::AmdXdnaNpu { .. } => AcceleratorFamily::Npu,
            Self::Tpu { .. } => AcceleratorFamily::Tpu,
            Self::Gaudi { .. }
            | Self::AwsNeuron { .. }
            | Self::QualcommAi100 { .. } => AcceleratorFamily::AiAsic,
        }
    }

    /// Relative throughput multiplier vs CPU (rough inference estimate).
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
            Self::AwsNeuron { chip_type, core_count, .. } => {
                let base = match chip_type {
                    NeuronChipType::Inferentia => 15.0,
                    NeuronChipType::Trainium => 24.0,
                };
                base * (*core_count as f64 / 2.0).max(1.0)
            }
            Self::QualcommAi100 { .. } => 14.0,
            Self::IntelOneApi { .. } => 13.0,
        }
    }

    /// Relative training throughput multiplier vs CPU.
    /// Differs from inference throughput for some devices (e.g. Inferentia is
    /// inference-only, Trainium excels at training).
    pub fn training_multiplier(&self) -> f64 {
        match self {
            Self::Cpu => 1.0,
            Self::CudaGpu { .. } => 20.0,
            Self::RocmGpu { .. } => 15.0,
            Self::MetalGpu => 10.0,
            Self::VulkanGpu { .. } => 6.0, // Vulkan compute shaders, limited training support
            Self::IntelNpu | Self::AppleNpu | Self::AmdXdnaNpu { .. } => 0.0, // NPUs: inference only
            Self::Tpu { version, .. } => match version {
                TpuVersion::V4 => 28.0,
                TpuVersion::V5e => 20.0,
                TpuVersion::V5p => 40.0,
            },
            Self::Gaudi { generation, .. } => match generation {
                GaudiGeneration::Gaudi2 => 25.0,
                GaudiGeneration::Gaudi3 => 35.0,
            },
            Self::AwsNeuron { chip_type, core_count, .. } => match chip_type {
                NeuronChipType::Inferentia => 0.0, // inference only
                NeuronChipType::Trainium => 26.0 * (*core_count as f64 / 2.0).max(1.0),
            },
            Self::QualcommAi100 { .. } => 0.0, // inference only
            Self::IntelOneApi { .. } => 10.0, // oneAPI training support via SYCL
        }
    }

    /// Whether this device supports training workloads at all.
    pub fn supports_training(&self) -> bool {
        self.training_multiplier() > 0.0
    }

    /// Priority rank for [`AcceleratorRegistry::best_available`] (higher = preferred).
    pub(crate) fn rank(&self) -> u32 {
        match self {
            Self::Tpu { version: TpuVersion::V5p, .. } => 80,
            Self::Gaudi { generation: GaudiGeneration::Gaudi3, .. } => 75,
            Self::Tpu { version: TpuVersion::V4, .. } => 70,
            Self::Gaudi { generation: GaudiGeneration::Gaudi2, .. } => 65,
            Self::CudaGpu { .. } => 60,
            Self::AwsNeuron { chip_type: NeuronChipType::Trainium, .. } => 58,
            Self::Tpu { version: TpuVersion::V5e, .. } => 55,
            Self::RocmGpu { .. } => 50,
            Self::AwsNeuron { chip_type: NeuronChipType::Inferentia, .. } => 45,
            Self::MetalGpu => 40,
            Self::QualcommAi100 { .. } => 38,
            Self::VulkanGpu { .. } => 35,
            Self::AppleNpu => 30,
            Self::AmdXdnaNpu { .. } => 25,
            Self::IntelOneApi { .. } => 42,
            Self::IntelNpu => 20,
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
            Self::VulkanGpu { device_id, device_name } => {
                write!(f, "Vulkan GPU (device {}, {})", device_id, device_name)
            }
            Self::IntelNpu => write!(f, "Intel NPU"),
            Self::AmdXdnaNpu { device_id } => write!(f, "AMD XDNA NPU (device {})", device_id),
            Self::AppleNpu => write!(f, "Apple Neural Engine"),
            Self::Tpu { device_id, chip_count, version } => {
                write!(f, "TPU {} (device {}, {} chips)", version, device_id, chip_count)
            }
            Self::Gaudi { device_id, generation } => {
                write!(f, "Intel {} (device {})", generation, device_id)
            }
            Self::AwsNeuron { device_id, chip_type, core_count } => {
                write!(f, "AWS {} (device {}, {} cores)", chip_type, device_id, core_count)
            }
            Self::QualcommAi100 { device_id } => {
                write!(f, "Qualcomm Cloud AI 100 (device {})", device_id)
            }
            Self::IntelOneApi { device_id } => {
                write!(f, "Intel oneAPI GPU (device {})", device_id)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AcceleratorFamily
// ---------------------------------------------------------------------------

/// Broad device family categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
// QuantizationLevel
// ---------------------------------------------------------------------------

/// Model weight quantisation levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationLevel {
    /// Full precision — FP32, 32 bits per parameter.
    None,
    /// Half precision — FP16, 16 bits per parameter.
    Float16,
    /// Brain floating point — BF16, 16 bits per parameter.
    BFloat16,
    /// 8-bit integer quantisation.
    Int8,
    /// 4-bit integer quantisation (GPTQ / AWQ style).
    Int4,
}

impl QuantizationLevel {
    /// Number of bits used per model parameter.
    pub fn bits_per_param(&self) -> u32 {
        match self {
            Self::None => 32,
            Self::Float16 | Self::BFloat16 => 16,
            Self::Int8 => 8,
            Self::Int4 => 4,
        }
    }

    /// Memory reduction factor relative to FP32.
    pub fn memory_reduction_factor(&self) -> f64 {
        32.0 / self.bits_per_param() as f64
    }
}

impl fmt::Display for QuantizationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "FP32"),
            Self::Float16 => write!(f, "FP16"),
            Self::BFloat16 => write!(f, "BF16"),
            Self::Int8 => write!(f, "INT8"),
            Self::Int4 => write!(f, "INT4"),
        }
    }
}

// ---------------------------------------------------------------------------
// AcceleratorProfile
// ---------------------------------------------------------------------------

/// A detected hardware accelerator and its capabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorProfile {
    /// The accelerator type.
    pub accelerator: AcceleratorType,
    /// Whether this device is currently available for use.
    pub available: bool,
    /// Total device memory in bytes (VRAM, HBM, or system RAM for CPU).
    pub memory_bytes: u64,
    /// Compute capability or version string (e.g. `"8.6"` for CUDA Ampere).
    pub compute_capability: Option<String>,
    /// Driver version string.
    pub driver_version: Option<String>,
}

impl AcceleratorProfile {
    /// Whether this profile supports the given quantisation level.
    pub fn supports_quantization(&self, level: &QuantizationLevel) -> bool {
        match self.accelerator.family() {
            AcceleratorFamily::Cpu | AcceleratorFamily::Gpu => true,
            AcceleratorFamily::Npu => {
                matches!(level, QuantizationLevel::Int8 | QuantizationLevel::Int4)
            }
            AcceleratorFamily::Tpu => {
                // TPUs natively support FP32, BF16, and INT8.
                // FP16 and INT4 are not hardware-native.
                matches!(
                    level,
                    QuantizationLevel::None | QuantizationLevel::BFloat16 | QuantizationLevel::Int8
                )
            }
            AcceleratorFamily::AiAsic => {
                match &self.accelerator {
                    // Gaudi supports FP32, BF16, FP16, FP8
                    AcceleratorType::Gaudi { .. } => {
                        matches!(
                            level,
                            QuantizationLevel::None
                                | QuantizationLevel::BFloat16
                                | QuantizationLevel::Float16
                                | QuantizationLevel::Int8
                        )
                    }
                    // Neuron supports FP32, BF16, FP16, INT8
                    AcceleratorType::AwsNeuron { .. } => {
                        matches!(
                            level,
                            QuantizationLevel::None
                                | QuantizationLevel::BFloat16
                                | QuantizationLevel::Float16
                                | QuantizationLevel::Int8
                        )
                    }
                    // Qualcomm AI 100 optimised for INT8/INT4 inference
                    AcceleratorType::QualcommAi100 { .. } => {
                        matches!(
                            level,
                            QuantizationLevel::Float16
                                | QuantizationLevel::Int8
                                | QuantizationLevel::Int4
                        )
                    }
                    _ => true,
                }
            }
        }
    }

    /// The preferred (most efficient) quantisation level for this device.
    pub fn preferred_quantization(&self) -> QuantizationLevel {
        match self.accelerator.family() {
            AcceleratorFamily::Cpu => QuantizationLevel::Float16,
            AcceleratorFamily::Gpu => QuantizationLevel::Float16,
            AcceleratorFamily::Npu => QuantizationLevel::Int8,
            AcceleratorFamily::Tpu => QuantizationLevel::BFloat16,
            AcceleratorFamily::AiAsic => match &self.accelerator {
                AcceleratorType::Gaudi { .. } => QuantizationLevel::BFloat16,
                AcceleratorType::AwsNeuron { .. } => QuantizationLevel::BFloat16,
                AcceleratorType::QualcommAi100 { .. } => QuantizationLevel::Int8,
                _ => QuantizationLevel::Float16,
            },
        }
    }
}

impl fmt::Display for AcceleratorProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({:.1} GB{})",
            self.accelerator,
            self.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            if self.available { "" } else { ", unavailable" }
        )
    }
}

// ---------------------------------------------------------------------------
// ShardingStrategy
// ---------------------------------------------------------------------------

/// Strategy for distributing a model across devices.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// No sharding — run on a single device.
    None,
    /// Split layers across devices in a pipeline.
    PipelineParallel { num_stages: u32 },
    /// Split individual tensors across devices.
    TensorParallel { num_devices: u32 },
    /// Replicate the full model for higher throughput.
    DataParallel { num_replicas: u32 },
}

impl ShardingStrategy {
    /// Minimum number of devices required.
    pub fn min_devices(&self) -> u32 {
        match self {
            Self::None => 1,
            Self::PipelineParallel { num_stages } => *num_stages,
            Self::TensorParallel { num_devices } => *num_devices,
            Self::DataParallel { num_replicas } => *num_replicas,
        }
    }
}

impl fmt::Display for ShardingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::PipelineParallel { num_stages } => {
                write!(f, "Pipeline Parallel ({} stages)", num_stages)
            }
            Self::TensorParallel { num_devices } => {
                write!(f, "Tensor Parallel ({} devices)", num_devices)
            }
            Self::DataParallel { num_replicas } => {
                write!(f, "Data Parallel ({} replicas)", num_replicas)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ModelShard
// ---------------------------------------------------------------------------

/// A slice of model layers assigned to a specific device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelShard {
    pub shard_id: u32,
    /// Inclusive layer range `(start, end)`.
    pub layer_range: (u32, u32),
    pub device: AcceleratorType,
    /// Estimated memory consumption in bytes.
    pub memory_bytes: u64,
}

impl ModelShard {
    /// Number of layers in this shard.
    pub fn num_layers(&self) -> u32 {
        if self.layer_range.1 >= self.layer_range.0 {
            self.layer_range.1 - self.layer_range.0 + 1
        } else {
            0
        }
    }

    /// Whether the layer range is valid.
    pub fn is_valid(&self) -> bool {
        self.layer_range.0 <= self.layer_range.1
    }
}

// ---------------------------------------------------------------------------
// ShardingPlan
// ---------------------------------------------------------------------------

/// A concrete plan for distributing model shards across devices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardingPlan {
    pub shards: Vec<ModelShard>,
    pub strategy: ShardingStrategy,
    pub total_memory_bytes: u64,
    pub estimated_tokens_per_sec: Option<f64>,
}

// ---------------------------------------------------------------------------
// AcceleratorRequirement
// ---------------------------------------------------------------------------

/// Hardware accelerator requirement for a workload (scheduling integration).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceleratorRequirement {
    /// No accelerator needed — CPU is sufficient.
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

impl Default for AcceleratorRequirement {
    fn default() -> Self {
        Self::None
    }
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

// ---------------------------------------------------------------------------
// VramEstimate (training workloads)
// ---------------------------------------------------------------------------

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
