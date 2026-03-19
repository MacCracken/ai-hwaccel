//! Accelerator capability profile.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::hardware::{AcceleratorFamily, AcceleratorType};
use crate::quantization::QuantizationLevel;

/// A detected hardware accelerator and its capabilities.
///
/// # Examples
///
/// ```rust
/// use ai_hwaccel::{AcceleratorProfile, QuantizationLevel};
///
/// let gpu = AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024);
/// assert!(gpu.supports_quantization(&QuantizationLevel::Float16));
/// assert_eq!(gpu.preferred_quantization(), QuantizationLevel::Float16);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
                    // Qualcomm AI 100: no native FP32/BF16 — hardware is
                    // optimised for quantised INT8/INT4 inference.
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

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

impl AcceleratorProfile {
    /// Create a CUDA GPU profile (for testing or manual config).
    pub fn cuda(device_id: u32, vram_bytes: u64) -> Self {
        Self {
            accelerator: AcceleratorType::CudaGpu { device_id },
            available: true,
            memory_bytes: vram_bytes,
            compute_capability: None,
            driver_version: None,
        }
    }

    /// Create a ROCm GPU profile.
    pub fn rocm(device_id: u32, vram_bytes: u64) -> Self {
        Self {
            accelerator: AcceleratorType::RocmGpu { device_id },
            available: true,
            memory_bytes: vram_bytes,
            compute_capability: None,
            driver_version: None,
        }
    }

    /// Create a TPU profile.
    pub fn tpu(device_id: u32, chip_count: u32, version: crate::hardware::TpuVersion) -> Self {
        let hbm = version.hbm_per_chip_bytes() * chip_count as u64;
        Self {
            accelerator: AcceleratorType::Tpu {
                device_id,
                chip_count,
                version,
            },
            available: true,
            memory_bytes: hbm,
            compute_capability: Some(format!("TPU {}", version)),
            driver_version: None,
        }
    }

    /// Create an Intel Gaudi HPU profile.
    pub fn gaudi(device_id: u32, generation: crate::hardware::GaudiGeneration) -> Self {
        Self {
            accelerator: AcceleratorType::Gaudi {
                device_id,
                generation,
            },
            available: true,
            memory_bytes: generation.hbm_bytes(),
            compute_capability: Some(generation.to_string()),
            driver_version: None,
        }
    }

    /// Create a CPU profile with the given amount of system RAM.
    pub fn cpu(memory_bytes: u64) -> Self {
        Self {
            accelerator: AcceleratorType::Cpu,
            available: true,
            memory_bytes,
            compute_capability: None,
            driver_version: None,
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
