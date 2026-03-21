//! Accelerator registry: query, filter, and memory estimation helpers.

use crate::error::DetectionError;
use crate::hardware::{AcceleratorFamily, AcceleratorType};
use crate::profile::AcceleratorProfile;
use crate::quantization::QuantizationLevel;
use crate::requirement::AcceleratorRequirement;
use crate::system_io::SystemIo;

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
///
/// # Schema history
///
/// - **v1**: Initial schema — profiles with accelerator type, availability,
///   memory, compute capability, driver version. Warnings array.
/// - **v2**: System I/O — per-device bandwidth, VRAM usage, PCIe link speed,
///   NUMA node. System-level interconnects and storage. `Timeout` error variant.
/// - **v3**: Runtime environment detection (Docker, Kubernetes, cloud instance
///   metadata). Per-backend timing API. Cost-aware planning.
pub const SCHEMA_VERSION: u32 = 3;

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
    /// System-level I/O topology (interconnects, storage).
    #[serde(default = "SystemIo::empty")]
    pub(crate) system_io: SystemIo,
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
            system_io: SystemIo::empty(),
        }
    }

    /// Build a registry from a pre-built list of profiles (for testing or config-driven setups).
    pub fn from_profiles(profiles: Vec<AcceleratorProfile>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            profiles,
            warnings: vec![],
            system_io: SystemIo::empty(),
        }
    }

    /// Schema version of this registry (for forward-compatibility checks).
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Deserialize from JSON with schema version validation.
    ///
    /// Returns `Err` if the JSON is malformed. Logs a warning if the schema
    /// version is newer than the current library version (forward-incompatible).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let registry: Self = serde_json::from_str(json)?;
        if registry.schema_version > SCHEMA_VERSION {
            tracing::warn!(
                json_version = registry.schema_version,
                lib_version = SCHEMA_VERSION,
                "registry JSON has newer schema version than this library — \
                 some fields may be missing or ignored"
            );
        }
        Ok(registry)
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
    #[inline]
    pub fn all_profiles(&self) -> &[AcceleratorProfile] {
        &self.profiles
    }

    /// Non-fatal warnings from detection (tool not found, parse errors, etc.).
    #[inline]
    pub fn warnings(&self) -> &[DetectionError] {
        &self.warnings
    }

    /// Only the available accelerator profiles.
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn has_accelerator(&self) -> bool {
        self.profiles
            .iter()
            .any(|p| p.available && !matches!(p.accelerator, AcceleratorType::Cpu))
    }

    /// All profiles matching a given [`AcceleratorFamily`].
    #[inline]
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

    /// System-level I/O topology (interconnects, storage).
    #[inline]
    pub fn system_io(&self) -> &SystemIo {
        &self.system_io
    }

    /// Estimate memory required for `model_params` parameters at the given quantisation.
    ///
    /// Formula: `params * (bits / 8)` plus 20% overhead for activations/KV cache.
    #[inline]
    pub fn estimate_memory(model_params: u64, quant: &QuantizationLevel) -> u64 {
        let bytes_per_param = quant.bits_per_param() as u64;
        let raw = model_params * bytes_per_param / 8;
        raw + raw / 5
    }

    /// Suggest a quantisation level based on available hardware and model size.
    ///
    /// The suggestion considers device-specific preferences (TPU → BF16,
    /// GPU → FP16, NPU → INT8) and falls back through progressively smaller
    /// quantisation levels until the model fits. The returned level is always
    /// supported by at least one available device (or the CPU fallback).
    ///
    /// Note: this is a heuristic. For production deployments, verify the
    /// returned level against [`AcceleratorProfile::supports_quantization`].
    pub fn suggest_quantization(&self, model_params: u64) -> QuantizationLevel {
        // Single pass: collect best memory per family and check for Gaudi.
        let mut best_tpu: u64 = 0;
        let mut best_gaudi: u64 = 0;
        let mut best_gpu: u64 = 0;
        let mut best_npu: u64 = 0;
        let mut best_cpu: u64 = 0;

        for p in &self.profiles {
            if !p.available {
                continue;
            }
            let mem = p.memory_bytes;
            match p.accelerator.family() {
                AcceleratorFamily::Tpu => best_tpu = best_tpu.max(mem),
                AcceleratorFamily::Gpu => best_gpu = best_gpu.max(mem),
                AcceleratorFamily::Npu => best_npu = best_npu.max(mem),
                AcceleratorFamily::Cpu => best_cpu = best_cpu.max(mem),
                AcceleratorFamily::AiAsic => {
                    if matches!(p.accelerator, AcceleratorType::Gaudi { .. }) {
                        best_gaudi = best_gaudi.max(mem);
                    }
                }
            }
        }

        // TPU → BF16 preferred
        if best_tpu > 0 {
            if Self::estimate_memory(model_params, &QuantizationLevel::BFloat16) <= best_tpu {
                return QuantizationLevel::BFloat16;
            }
            if Self::estimate_memory(model_params, &QuantizationLevel::Int8) <= best_tpu {
                return QuantizationLevel::Int8;
            }
        }

        // Gaudi → BF16 preferred
        if best_gaudi > 0 {
            if Self::estimate_memory(model_params, &QuantizationLevel::BFloat16) <= best_gaudi {
                return QuantizationLevel::BFloat16;
            }
            if Self::estimate_memory(model_params, &QuantizationLevel::Int8) <= best_gaudi {
                return QuantizationLevel::Int8;
            }
        }

        // GPU → FP16 preferred, step down
        if best_gpu > 0 {
            for quant in &[
                QuantizationLevel::Float16,
                QuantizationLevel::Int8,
                QuantizationLevel::Int4,
            ] {
                if Self::estimate_memory(model_params, quant) <= best_gpu {
                    return *quant;
                }
            }
        }

        // NPU → INT8/INT4 only
        if best_npu > 0 {
            for quant in &[QuantizationLevel::Int8, QuantizationLevel::Int4] {
                if Self::estimate_memory(model_params, quant) <= best_npu {
                    return *quant;
                }
            }
        }

        // CPU fallback — step down until it fits
        let cpu_mem = if best_cpu > 0 {
            best_cpu
        } else {
            16 * 1024 * 1024 * 1024
        };
        for quant in &[
            QuantizationLevel::Float16,
            QuantizationLevel::Int8,
            QuantizationLevel::Int4,
        ] {
            if Self::estimate_memory(model_params, quant) <= cpu_mem {
                return *quant;
            }
        }
        QuantizationLevel::Int4
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
    Cerebras,
    Graphcore,
    Groq,
    SamsungNpu,
    MediaTekApu,
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
        Backend::Cerebras,
        Backend::Graphcore,
        Backend::Groq,
        Backend::SamsungNpu,
        Backend::MediaTekApu,
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

    pub fn with_cuda(self) -> Self {
        self.set(Backend::Cuda, true)
    }
    pub fn with_rocm(self) -> Self {
        self.set(Backend::Rocm, true)
    }
    pub fn with_apple(self) -> Self {
        self.set(Backend::Apple, true)
    }
    pub fn with_vulkan(self) -> Self {
        self.set(Backend::Vulkan, true)
    }
    pub fn with_intel_npu(self) -> Self {
        self.set(Backend::IntelNpu, true)
    }
    pub fn with_amd_xdna(self) -> Self {
        self.set(Backend::AmdXdna, true)
    }
    pub fn with_tpu(self) -> Self {
        self.set(Backend::Tpu, true)
    }
    pub fn with_gaudi(self) -> Self {
        self.set(Backend::Gaudi, true)
    }
    pub fn with_aws_neuron(self) -> Self {
        self.set(Backend::AwsNeuron, true)
    }
    pub fn with_intel_oneapi(self) -> Self {
        self.set(Backend::IntelOneApi, true)
    }
    pub fn with_qualcomm(self) -> Self {
        self.set(Backend::Qualcomm, true)
    }
    pub fn with_cerebras(self) -> Self {
        self.set(Backend::Cerebras, true)
    }
    pub fn with_graphcore(self) -> Self {
        self.set(Backend::Graphcore, true)
    }
    pub fn with_groq(self) -> Self {
        self.set(Backend::Groq, true)
    }
    pub fn with_samsung_npu(self) -> Self {
        self.set(Backend::SamsungNpu, true)
    }
    pub fn with_mediatek_apu(self) -> Self {
        self.set(Backend::MediaTekApu, true)
    }

    pub fn without_cuda(self) -> Self {
        self.set(Backend::Cuda, false)
    }
    pub fn without_rocm(self) -> Self {
        self.set(Backend::Rocm, false)
    }
    pub fn without_apple(self) -> Self {
        self.set(Backend::Apple, false)
    }
    pub fn without_vulkan(self) -> Self {
        self.set(Backend::Vulkan, false)
    }
    pub fn without_intel_npu(self) -> Self {
        self.set(Backend::IntelNpu, false)
    }
    pub fn without_amd_xdna(self) -> Self {
        self.set(Backend::AmdXdna, false)
    }
    pub fn without_tpu(self) -> Self {
        self.set(Backend::Tpu, false)
    }
    pub fn without_gaudi(self) -> Self {
        self.set(Backend::Gaudi, false)
    }
    pub fn without_aws_neuron(self) -> Self {
        self.set(Backend::AwsNeuron, false)
    }
    pub fn without_intel_oneapi(self) -> Self {
        self.set(Backend::IntelOneApi, false)
    }
    pub fn without_qualcomm(self) -> Self {
        self.set(Backend::Qualcomm, false)
    }
    pub fn without_cerebras(self) -> Self {
        self.set(Backend::Cerebras, false)
    }
    pub fn without_graphcore(self) -> Self {
        self.set(Backend::Graphcore, false)
    }
    pub fn without_groq(self) -> Self {
        self.set(Backend::Groq, false)
    }
    pub fn without_samsung_npu(self) -> Self {
        self.set(Backend::SamsungNpu, false)
    }
    pub fn without_mediatek_apu(self) -> Self {
        self.set(Backend::MediaTekApu, false)
    }

    /// Run detection with only the enabled backends.
    pub fn detect(self) -> AcceleratorRegistry {
        crate::detect::detect_with_builder(self)
    }

    /// Run detection with per-backend timing information.
    pub fn detect_with_timing(self) -> crate::detect::TimedDetection {
        crate::detect::detect_with_builder_timed(self)
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

    /// Count of enabled backends (for deciding sequential vs parallel).
    pub(crate) fn enabled_count(&self) -> usize {
        self.enabled.iter().filter(|&&e| e).count()
    }
}
