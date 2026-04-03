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
    /// Human-readable device name (e.g. "RTX 4090", "AMD Radeon RX 7900 XTX").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name: Option<String>,

    // --- System I/O fields (0.20) -------------------------------------------
    /// Measured memory bandwidth in GB/s (e.g. HBM throughput).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_bandwidth_gbps: Option<f64>,
    /// Currently used device memory in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_used_bytes: Option<u64>,
    /// Currently free device memory in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_free_bytes: Option<u64>,
    /// PCIe host-to-device bandwidth in GB/s (theoretical max from link width × speed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pcie_bandwidth_gbps: Option<f64>,
    /// NUMA node this device is attached to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numa_node: Option<u32>,

    // --- Power and thermal (0.20) -------------------------------------------
    /// GPU temperature in degrees Celsius.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_c: Option<u32>,
    /// Current power draw in watts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_watts: Option<f64>,
    /// GPU utilization as a percentage (0–100).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_utilization_percent: Option<u32>,
}

impl AcceleratorProfile {
    /// Whether this profile supports the given quantisation level.
    #[must_use]
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
                    // Cerebras WSE: FP32, BF16, FP16, INT8
                    AcceleratorType::CerebrasWse { .. } => {
                        matches!(
                            level,
                            QuantizationLevel::None
                                | QuantizationLevel::BFloat16
                                | QuantizationLevel::Float16
                                | QuantizationLevel::Int8
                        )
                    }
                    // Graphcore IPU: FP32, FP16, INT8
                    AcceleratorType::GraphcoreIpu { .. } => {
                        matches!(
                            level,
                            QuantizationLevel::None
                                | QuantizationLevel::Float16
                                | QuantizationLevel::Int8
                        )
                    }
                    // Groq LPU: FP16, INT8, INT4
                    AcceleratorType::GroqLpu { .. } => {
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
    #[must_use]
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
                AcceleratorType::CerebrasWse { .. } => QuantizationLevel::BFloat16,
                AcceleratorType::GraphcoreIpu { .. } => QuantizationLevel::Float16,
                AcceleratorType::GroqLpu { .. } => QuantizationLevel::Int8,
                _ => QuantizationLevel::Float16,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

impl Default for AcceleratorProfile {
    fn default() -> Self {
        Self {
            accelerator: AcceleratorType::Cpu,
            available: true,
            memory_bytes: 0,
            compute_capability: None,
            driver_version: None,
            device_name: None,
            memory_bandwidth_gbps: None,
            memory_used_bytes: None,
            memory_free_bytes: None,
            pcie_bandwidth_gbps: None,
            numa_node: None,
            temperature_c: None,
            power_watts: None,
            gpu_utilization_percent: None,
        }
    }
}

impl AcceleratorProfile {
    /// Create a CUDA GPU profile (for testing or manual config).
    pub fn cuda(device_id: u32, vram_bytes: u64) -> Self {
        Self {
            accelerator: AcceleratorType::CudaGpu { device_id },
            memory_bytes: vram_bytes,
            ..Default::default()
        }
    }

    /// Create a ROCm GPU profile.
    pub fn rocm(device_id: u32, vram_bytes: u64) -> Self {
        Self {
            accelerator: AcceleratorType::RocmGpu { device_id },
            memory_bytes: vram_bytes,
            ..Default::default()
        }
    }

    /// Create a TPU profile.
    pub fn tpu(device_id: u32, chip_count: u32, version: crate::hardware::TpuVersion) -> Self {
        let hbm = version
            .hbm_per_chip_bytes()
            .saturating_mul(chip_count as u64);
        Self {
            accelerator: AcceleratorType::Tpu {
                device_id,
                chip_count,
                version,
            },
            memory_bytes: hbm,
            compute_capability: Some(format!("TPU {}", version)),
            ..Default::default()
        }
    }

    /// Create an Intel Gaudi HPU profile.
    pub fn gaudi(device_id: u32, generation: crate::hardware::GaudiGeneration) -> Self {
        Self {
            accelerator: AcceleratorType::Gaudi {
                device_id,
                generation,
            },
            memory_bytes: generation.hbm_bytes(),
            compute_capability: Some(generation.to_string()),
            ..Default::default()
        }
    }

    /// Create a CPU profile with the given amount of system RAM.
    pub fn cpu(memory_bytes: u64) -> Self {
        Self {
            memory_bytes,
            ..Default::default()
        }
    }
}

impl fmt::Display for AcceleratorProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.device_name {
            write!(
                f,
                "{} [{}] ({:.1} GB{})",
                self.accelerator,
                name,
                self.memory_bytes as f64 / crate::units::BYTES_PER_GIB,
                if self.available { "" } else { ", unavailable" }
            )
        } else {
            write!(
                f,
                "{} ({:.1} GB{})",
                self.accelerator,
                self.memory_bytes as f64 / crate::units::BYTES_PER_GIB,
                if self.available { "" } else { ", unavailable" }
            )
        }
    }
}
