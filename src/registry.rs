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
/// assert!(!plan.shards().is_empty());
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
    #[must_use]
    #[inline]
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
    #[must_use = "iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn available(&self) -> impl Iterator<Item = &AcceleratorProfile> {
        self.profiles.iter().filter(|p| p.available)
    }

    /// The highest-ranked available device.
    #[must_use]
    pub fn best_available(&self) -> Option<&AcceleratorProfile> {
        self.profiles
            .iter()
            .filter(|p| p.available)
            .max_by_key(|p| p.accelerator.rank())
    }

    /// Total memory across all **available** devices.
    #[must_use]
    #[inline]
    pub fn total_memory(&self) -> u64 {
        self.profiles
            .iter()
            .filter(|p| p.available)
            .map(|p| p.memory_bytes)
            .sum()
    }

    /// Total memory across all available non-CPU devices (GPU + NPU + TPU + ASIC).
    #[must_use]
    pub fn total_accelerator_memory(&self) -> u64 {
        self.profiles
            .iter()
            .filter(|p| p.available && !matches!(p.accelerator, AcceleratorType::Cpu))
            .map(|p| p.memory_bytes)
            .sum()
    }

    /// Whether any non-CPU accelerator is available.
    #[must_use]
    #[inline]
    pub fn has_accelerator(&self) -> bool {
        self.profiles
            .iter()
            .any(|p| p.available && !matches!(p.accelerator, AcceleratorType::Cpu))
    }

    /// All profiles matching a given [`AcceleratorFamily`].
    #[must_use = "iterator is lazy and does nothing unless consumed"]
    #[inline]
    pub fn by_family(
        &self,
        family: AcceleratorFamily,
    ) -> impl Iterator<Item = &AcceleratorProfile> {
        self.profiles
            .iter()
            .filter(move |p| p.available && p.accelerator.family() == family)
    }

    /// All profiles satisfying an [`AcceleratorRequirement`].
    #[must_use = "iterator is lazy and does nothing unless consumed"]
    pub fn satisfying<'a>(
        &'a self,
        req: &'a AcceleratorRequirement,
    ) -> impl Iterator<Item = &'a AcceleratorProfile> {
        self.profiles.iter().filter(move |p| req.satisfied_by(p))
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
    #[must_use]
    #[inline]
    pub fn estimate_memory(model_params: u64, quant: &QuantizationLevel) -> u64 {
        let bytes_per_param = quant.bits_per_param() as u64;
        let raw = model_params * bytes_per_param / crate::units::BITS_PER_BYTE as u64;
        raw + raw / crate::units::ACTIVATION_OVERHEAD_DIVISOR
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
    #[must_use]
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

        // Precompute memory estimates once (avoids up to 9 redundant calls).
        let est_bf16 = Self::estimate_memory(model_params, &QuantizationLevel::BFloat16);
        let est_fp16 = Self::estimate_memory(model_params, &QuantizationLevel::Float16);
        let est_int8 = Self::estimate_memory(model_params, &QuantizationLevel::Int8);
        let est_int4 = Self::estimate_memory(model_params, &QuantizationLevel::Int4);

        // TPU → BF16 preferred
        if best_tpu > 0 {
            if est_bf16 <= best_tpu {
                return QuantizationLevel::BFloat16;
            }
            if est_int8 <= best_tpu {
                return QuantizationLevel::Int8;
            }
        }

        // Gaudi → BF16 preferred
        if best_gaudi > 0 {
            if est_bf16 <= best_gaudi {
                return QuantizationLevel::BFloat16;
            }
            if est_int8 <= best_gaudi {
                return QuantizationLevel::Int8;
            }
        }

        // GPU → FP16 preferred, step down
        if best_gpu > 0 {
            if est_fp16 <= best_gpu {
                return QuantizationLevel::Float16;
            }
            if est_int8 <= best_gpu {
                return QuantizationLevel::Int8;
            }
            if est_int4 <= best_gpu {
                return QuantizationLevel::Int4;
            }
        }

        // NPU → INT8/INT4 only
        if best_npu > 0 {
            if est_int8 <= best_npu {
                return QuantizationLevel::Int8;
            }
            if est_int4 <= best_npu {
                return QuantizationLevel::Int4;
            }
        }

        // CPU fallback — step down until it fits
        let cpu_mem = if best_cpu > 0 {
            best_cpu
        } else {
            16 * 1024 * 1024 * 1024
        };
        if est_fp16 <= cpu_mem {
            return QuantizationLevel::Float16;
        }
        if est_int8 <= cpu_mem {
            return QuantizationLevel::Int8;
        }
        if est_int4 <= cpu_mem {
            return QuantizationLevel::Int4;
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
    WindowsWmi,
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
        Backend::WindowsWmi,
    ];
}

/// Builder for selective hardware detection.
///
/// By default all backends are enabled. Use `without_*` methods to disable
/// specific backends, or start from `none()` and use `with_*` to enable only
/// the ones you need.
///
/// Uses a `u32` bitmask internally — zero heap allocation.
#[derive(Debug, Clone, Copy)]
pub struct DetectBuilder {
    enabled: u32,
}

impl DetectBuilder {
    /// All backends enabled (default).
    pub fn new() -> Self {
        Self {
            enabled: (1u32 << Backend::ALL.len()) - 1,
        }
    }

    /// No backends enabled — start from scratch with `with_*` methods.
    pub fn none() -> Self {
        Self { enabled: 0 }
    }

    /// Enable a specific backend.
    pub fn with(mut self, backend: Backend) -> Self {
        self.enabled |= 1 << backend as u32;
        self
    }

    /// Disable a specific backend.
    pub fn without(mut self, backend: Backend) -> Self {
        self.enabled &= !(1 << backend as u32);
        self
    }

    fn is_enabled(&self, backend: Backend) -> bool {
        self.enabled & (1 << backend as u32) != 0
    }

    // Convenience methods (delegate to generic with/without).
    #[inline]
    pub fn with_cuda(self) -> Self {
        self.with(Backend::Cuda)
    }
    #[inline]
    pub fn with_rocm(self) -> Self {
        self.with(Backend::Rocm)
    }
    #[inline]
    pub fn with_apple(self) -> Self {
        self.with(Backend::Apple)
    }
    #[inline]
    pub fn with_vulkan(self) -> Self {
        self.with(Backend::Vulkan)
    }
    #[inline]
    pub fn with_intel_npu(self) -> Self {
        self.with(Backend::IntelNpu)
    }
    #[inline]
    pub fn with_amd_xdna(self) -> Self {
        self.with(Backend::AmdXdna)
    }
    #[inline]
    pub fn with_tpu(self) -> Self {
        self.with(Backend::Tpu)
    }
    #[inline]
    pub fn with_gaudi(self) -> Self {
        self.with(Backend::Gaudi)
    }
    #[inline]
    pub fn with_aws_neuron(self) -> Self {
        self.with(Backend::AwsNeuron)
    }
    #[inline]
    pub fn with_intel_oneapi(self) -> Self {
        self.with(Backend::IntelOneApi)
    }
    #[inline]
    pub fn with_qualcomm(self) -> Self {
        self.with(Backend::Qualcomm)
    }
    #[inline]
    pub fn with_cerebras(self) -> Self {
        self.with(Backend::Cerebras)
    }
    #[inline]
    pub fn with_graphcore(self) -> Self {
        self.with(Backend::Graphcore)
    }
    #[inline]
    pub fn with_groq(self) -> Self {
        self.with(Backend::Groq)
    }
    #[inline]
    pub fn with_samsung_npu(self) -> Self {
        self.with(Backend::SamsungNpu)
    }
    #[inline]
    pub fn with_mediatek_apu(self) -> Self {
        self.with(Backend::MediaTekApu)
    }
    #[inline]
    pub fn with_windows_wmi(self) -> Self {
        self.with(Backend::WindowsWmi)
    }

    #[inline]
    pub fn without_cuda(self) -> Self {
        self.without(Backend::Cuda)
    }
    #[inline]
    pub fn without_rocm(self) -> Self {
        self.without(Backend::Rocm)
    }
    #[inline]
    pub fn without_apple(self) -> Self {
        self.without(Backend::Apple)
    }
    #[inline]
    pub fn without_vulkan(self) -> Self {
        self.without(Backend::Vulkan)
    }
    #[inline]
    pub fn without_intel_npu(self) -> Self {
        self.without(Backend::IntelNpu)
    }
    #[inline]
    pub fn without_amd_xdna(self) -> Self {
        self.without(Backend::AmdXdna)
    }
    #[inline]
    pub fn without_tpu(self) -> Self {
        self.without(Backend::Tpu)
    }
    #[inline]
    pub fn without_gaudi(self) -> Self {
        self.without(Backend::Gaudi)
    }
    #[inline]
    pub fn without_aws_neuron(self) -> Self {
        self.without(Backend::AwsNeuron)
    }
    #[inline]
    pub fn without_intel_oneapi(self) -> Self {
        self.without(Backend::IntelOneApi)
    }
    #[inline]
    pub fn without_qualcomm(self) -> Self {
        self.without(Backend::Qualcomm)
    }
    #[inline]
    pub fn without_cerebras(self) -> Self {
        self.without(Backend::Cerebras)
    }
    #[inline]
    pub fn without_graphcore(self) -> Self {
        self.without(Backend::Graphcore)
    }
    #[inline]
    pub fn without_groq(self) -> Self {
        self.without(Backend::Groq)
    }
    #[inline]
    pub fn without_samsung_npu(self) -> Self {
        self.without(Backend::SamsungNpu)
    }
    #[inline]
    pub fn without_mediatek_apu(self) -> Self {
        self.without(Backend::MediaTekApu)
    }
    #[inline]
    pub fn without_windows_wmi(self) -> Self {
        self.without(Backend::WindowsWmi)
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
        self.enabled.count_ones() as usize
    }
}
