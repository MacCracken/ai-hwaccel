//! Accelerator registry: query, filter, and memory estimation helpers.

use crate::error::DetectionError;
use crate::hardware::{AcceleratorFamily, AcceleratorType};
use crate::profile::AcceleratorProfile;
use crate::quantization::QuantizationLevel;
use crate::requirement::AcceleratorRequirement;

/// Registry of detected hardware accelerators with planning helpers.
///
/// # Examples
///
/// ```rust
/// use ai_hwaccel::{AcceleratorRegistry, AcceleratorProfile, QuantizationLevel};
///
/// // Build a registry manually for testing.
/// let registry = AcceleratorRegistry::from_profiles(vec![
///     AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
///     AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024),
/// ]);
///
/// assert!(registry.has_accelerator());
/// let quant = registry.suggest_quantization(7_000_000_000);
/// let plan = registry.plan_sharding(7_000_000_000, &quant);
/// assert!(!plan.shards.is_empty());
/// ```
/// Current schema version for serialized registries.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorRegistry {
    /// Schema version for forward-compatibility checking.
    #[serde(default = "default_schema_version")]
    pub(crate) schema_version: u32,
    pub(crate) profiles: Vec<AcceleratorProfile>,
    /// Non-fatal warnings encountered during detection.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) warnings: Vec<DetectionError>,
}

fn default_schema_version() -> u32 {
    SCHEMA_VERSION
}

impl AcceleratorRegistry {
    /// Creates a registry containing only a default CPU profile.
    pub fn new() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            profiles: vec![crate::detect::cpu_profile()],
            warnings: vec![],
        }
    }

    /// Build a registry from a pre-built list of profiles (for testing or config-driven setups).
    pub fn from_profiles(profiles: Vec<AcceleratorProfile>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            profiles,
            warnings: vec![],
        }
    }

    /// Schema version of this registry (for forward-compatibility checks).
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns a [`DetectBuilder`] for fine-grained control over which backends
    /// to probe.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ai_hwaccel::AcceleratorRegistry;
    ///
    /// let registry = AcceleratorRegistry::builder()
    ///     .with_cuda()
    ///     .with_rocm()
    ///     .without_vulkan()
    ///     .detect();
    /// ```
    pub fn builder() -> DetectBuilder {
        DetectBuilder::new()
    }

    /// All registered profiles (including unavailable ones).
    pub fn all_profiles(&self) -> &[AcceleratorProfile] {
        &self.profiles
    }

    /// Non-fatal warnings from detection (tool not found, parse errors, etc.).
    pub fn warnings(&self) -> &[DetectionError] {
        &self.warnings
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

// ---------------------------------------------------------------------------
// DetectBuilder
// ---------------------------------------------------------------------------

/// Which backends are enabled in the builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    Cuda,
    Rocm,
    Apple,
    Vulkan,
    IntelNpu,
    AmdXdna,
    Tpu,
    Gaudi,
    AwsNeuron,
    IntelOneApi,
    Qualcomm,
}

impl Backend {
    /// All known backends.
    pub const ALL: &[Backend] = &[
        Backend::Cuda,
        Backend::Rocm,
        Backend::Apple,
        Backend::Vulkan,
        Backend::IntelNpu,
        Backend::AmdXdna,
        Backend::Tpu,
        Backend::Gaudi,
        Backend::AwsNeuron,
        Backend::IntelOneApi,
        Backend::Qualcomm,
    ];
}

/// Builder for selective hardware detection.
///
/// By default all backends are enabled. Use `without_*` methods to disable
/// specific backends, or start from `none()` and use `with_*` to enable only
/// the ones you need.
#[derive(Debug, Clone)]
pub struct DetectBuilder {
    enabled: Vec<bool>,
}

impl DetectBuilder {
    /// All backends enabled (default).
    pub fn new() -> Self {
        Self {
            enabled: vec![true; Backend::ALL.len()],
        }
    }

    /// No backends enabled — start from scratch with `with_*` methods.
    pub fn none() -> Self {
        Self {
            enabled: vec![false; Backend::ALL.len()],
        }
    }

    fn set(mut self, backend: Backend, enabled: bool) -> Self {
        self.enabled[backend as usize] = enabled;
        self
    }

    fn is_enabled(&self, backend: Backend) -> bool {
        self.enabled[backend as usize]
    }

    pub fn with_cuda(self) -> Self { self.set(Backend::Cuda, true) }
    pub fn with_rocm(self) -> Self { self.set(Backend::Rocm, true) }
    pub fn with_apple(self) -> Self { self.set(Backend::Apple, true) }
    pub fn with_vulkan(self) -> Self { self.set(Backend::Vulkan, true) }
    pub fn with_intel_npu(self) -> Self { self.set(Backend::IntelNpu, true) }
    pub fn with_amd_xdna(self) -> Self { self.set(Backend::AmdXdna, true) }
    pub fn with_tpu(self) -> Self { self.set(Backend::Tpu, true) }
    pub fn with_gaudi(self) -> Self { self.set(Backend::Gaudi, true) }
    pub fn with_aws_neuron(self) -> Self { self.set(Backend::AwsNeuron, true) }
    pub fn with_intel_oneapi(self) -> Self { self.set(Backend::IntelOneApi, true) }
    pub fn with_qualcomm(self) -> Self { self.set(Backend::Qualcomm, true) }

    pub fn without_cuda(self) -> Self { self.set(Backend::Cuda, false) }
    pub fn without_rocm(self) -> Self { self.set(Backend::Rocm, false) }
    pub fn without_apple(self) -> Self { self.set(Backend::Apple, false) }
    pub fn without_vulkan(self) -> Self { self.set(Backend::Vulkan, false) }
    pub fn without_intel_npu(self) -> Self { self.set(Backend::IntelNpu, false) }
    pub fn without_amd_xdna(self) -> Self { self.set(Backend::AmdXdna, false) }
    pub fn without_tpu(self) -> Self { self.set(Backend::Tpu, false) }
    pub fn without_gaudi(self) -> Self { self.set(Backend::Gaudi, false) }
    pub fn without_aws_neuron(self) -> Self { self.set(Backend::AwsNeuron, false) }
    pub fn without_intel_oneapi(self) -> Self { self.set(Backend::IntelOneApi, false) }
    pub fn without_qualcomm(self) -> Self { self.set(Backend::Qualcomm, false) }

    /// Run detection with only the enabled backends.
    pub fn detect(self) -> AcceleratorRegistry {
        crate::detect::detect_with_builder(self)
    }
}

impl Default for DetectBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Make is_enabled accessible to detect module
impl DetectBuilder {
    pub(crate) fn backend_enabled(&self, backend: Backend) -> bool {
        self.is_enabled(backend)
    }
}
