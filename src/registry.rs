//! Accelerator registry: query, filter, and memory estimation helpers.

use crate::hardware::{AcceleratorFamily, AcceleratorType};
use crate::profile::AcceleratorProfile;
use crate::quantization::QuantizationLevel;
use crate::requirement::AcceleratorRequirement;

/// Registry of detected hardware accelerators with planning helpers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AcceleratorRegistry {
    pub(crate) profiles: Vec<AcceleratorProfile>,
}

impl AcceleratorRegistry {
    /// Creates a registry containing only a default CPU profile.
    pub fn new() -> Self {
        Self {
            profiles: vec![crate::detect::cpu_profile()],
        }
    }

    /// Build a registry from a pre-built list of profiles (for testing or config-driven setups).
    pub fn from_profiles(profiles: Vec<AcceleratorProfile>) -> Self {
        Self { profiles }
    }

    /// All registered profiles (including unavailable ones).
    pub fn all_profiles(&self) -> &[AcceleratorProfile] {
        &self.profiles
    }

    /// Only the available accelerator profiles.
    pub fn available(&self) -> Vec<&AcceleratorProfile> {
        self.profiles.iter().filter(|p| p.available).collect()
    }

    /// The highest-ranked available device.
    pub fn best_available(&self) -> Option<&AcceleratorProfile> {
        self.profiles
            .iter()
            .filter(|p| p.available)
            .max_by_key(|p| p.accelerator.rank())
    }

    /// Total memory across all **available** devices.
    pub fn total_memory(&self) -> u64 {
        self.profiles
            .iter()
            .filter(|p| p.available)
            .map(|p| p.memory_bytes)
            .sum()
    }

    /// Total memory across all available non-CPU devices (GPU + NPU + TPU + ASIC).
    pub fn total_accelerator_memory(&self) -> u64 {
        self.profiles
            .iter()
            .filter(|p| p.available && !matches!(p.accelerator, AcceleratorType::Cpu))
            .map(|p| p.memory_bytes)
            .sum()
    }

    /// Whether any non-CPU accelerator is available.
    pub fn has_accelerator(&self) -> bool {
        self.profiles
            .iter()
            .any(|p| p.available && !matches!(p.accelerator, AcceleratorType::Cpu))
    }

    /// All profiles matching a given [`AcceleratorFamily`].
    pub fn by_family(&self, family: AcceleratorFamily) -> Vec<&AcceleratorProfile> {
        self.profiles
            .iter()
            .filter(|p| p.available && p.accelerator.family() == family)
            .collect()
    }

    /// All profiles satisfying an [`AcceleratorRequirement`].
    pub fn satisfying(&self, req: &AcceleratorRequirement) -> Vec<&AcceleratorProfile> {
        self.profiles
            .iter()
            .filter(|p| req.satisfied_by(p))
            .collect()
    }

    /// Add a profile manually (for testing or manual config).
    pub fn add_profile(&mut self, profile: AcceleratorProfile) {
        self.profiles.push(profile);
    }

    /// Estimate memory required for `model_params` parameters at the given quantisation.
    ///
    /// Formula: `params * (bits / 8)` plus 20% overhead for activations/KV cache.
    pub fn estimate_memory(model_params: u64, quant: &QuantizationLevel) -> u64 {
        let bytes_per_param = quant.bits_per_param() as u64;
        let raw = model_params * bytes_per_param / 8;
        raw + raw / 5
    }

    /// Suggest a quantisation level based on available hardware and model size.
    pub fn suggest_quantization(&self, model_params: u64) -> QuantizationLevel {
        // Check for TPU first — TPUs strongly prefer BFloat16
        if let Some(tpu_mem) = self.best_memory_for(AcceleratorFamily::Tpu) {
            if Self::estimate_memory(model_params, &QuantizationLevel::BFloat16) <= tpu_mem {
                return QuantizationLevel::BFloat16;
            }
            if Self::estimate_memory(model_params, &QuantizationLevel::Int8) <= tpu_mem {
                return QuantizationLevel::Int8;
            }
        }

        // Check for Gaudi — also prefers BFloat16
        if let Some(gaudi_mem) = self.best_memory_for(AcceleratorFamily::AiAsic)
            && Self::estimate_memory(model_params, &QuantizationLevel::BFloat16) <= gaudi_mem
        {
            return QuantizationLevel::BFloat16;
        }

        // Check GPU
        if let Some(gpu_mem) = self.best_memory_for(AcceleratorFamily::Gpu) {
            for quant in &[
                QuantizationLevel::Float16,
                QuantizationLevel::Int8,
                QuantizationLevel::Int4,
            ] {
                if Self::estimate_memory(model_params, quant) <= gpu_mem {
                    return *quant;
                }
            }
        }

        // Check NPU (INT8/INT4 only)
        if let Some(npu_mem) = self.best_memory_for(AcceleratorFamily::Npu) {
            for quant in &[QuantizationLevel::Int8, QuantizationLevel::Int4] {
                if Self::estimate_memory(model_params, quant) <= npu_mem {
                    return *quant;
                }
            }
        }

        // Fallback: FP16 on CPU
        QuantizationLevel::Float16
    }

    /// Returns the largest device memory for available devices of a given family.
    fn best_memory_for(&self, family: AcceleratorFamily) -> Option<u64> {
        self.profiles
            .iter()
            .filter(|p| p.available && p.accelerator.family() == family)
            .map(|p| p.memory_bytes)
            .max()
    }
}

impl Default for AcceleratorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
