//! Hardware detection: probes sysfs, /dev, and PATH tools to discover accelerators.

#[cfg(feature = "amd-xdna")]
mod amd_xdna;
#[cfg(feature = "apple")]
mod apple;
pub(crate) mod command;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "gaudi")]
mod gaudi;
#[cfg(feature = "intel-npu")]
mod intel_npu;
#[cfg(feature = "intel-oneapi")]
mod intel_oneapi;
#[cfg(feature = "aws-neuron")]
mod neuron;
#[cfg(feature = "qualcomm")]
mod qualcomm;
#[cfg(feature = "rocm")]
mod rocm;
#[cfg(feature = "tpu")]
mod tpu;
#[cfg(feature = "vulkan")]
mod vulkan;

use std::path::Path;

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;
use crate::registry::{AcceleratorRegistry, Backend, DetectBuilder};

/// Per-backend detection result.
type DetectResult = (Vec<AcceleratorProfile>, Vec<DetectionError>);

impl AcceleratorRegistry {
    /// Probes the system for all available accelerators.
    ///
    /// Detection is best-effort: missing tools or sysfs entries simply mean
    /// the corresponding accelerator is not registered. Non-fatal issues are
    /// collected in [`AcceleratorRegistry::warnings`].
    ///
    /// All backends run **in parallel** via [`std::thread::scope`] for
    /// lower wall-clock latency on systems with multiple CLI tools.
    ///
    /// Backends can be disabled at compile time via cargo features
    /// (e.g. `default-features = false, features = ["cuda", "tpu"]`).
    pub fn detect() -> Self {
        detect_with_builder(DetectBuilder::new())
    }
}

/// Run detection with a builder's backend selection.
pub(crate) fn detect_with_builder(builder: DetectBuilder) -> AcceleratorRegistry {
    let mut all_profiles = vec![cpu_profile()];
    let mut all_warnings: Vec<DetectionError> = Vec::new();

    std::thread::scope(|s| {
        let mut handles: Vec<std::thread::ScopedJoinHandle<'_, DetectResult>> = Vec::new();

        macro_rules! spawn_backend {
            ($feature:literal, $backend:expr, $detect_fn:expr) => {
                #[cfg(feature = $feature)]
                if builder.backend_enabled($backend) {
                    handles.push(s.spawn(|| {
                        let mut p = Vec::new();
                        let mut w = Vec::new();
                        $detect_fn(&mut p, &mut w);
                        (p, w)
                    }));
                }
            };
        }

        spawn_backend!("cuda", Backend::Cuda, cuda::detect_cuda);
        spawn_backend!("rocm", Backend::Rocm, rocm::detect_rocm);
        spawn_backend!("apple", Backend::Apple, apple::detect_metal_and_ane);
        spawn_backend!("vulkan", Backend::Vulkan, vulkan::detect_vulkan);
        spawn_backend!("intel-npu", Backend::IntelNpu, intel_npu::detect_intel_npu);
        spawn_backend!("amd-xdna", Backend::AmdXdna, amd_xdna::detect_amd_xdna);
        spawn_backend!("tpu", Backend::Tpu, tpu::detect_tpu);
        spawn_backend!("gaudi", Backend::Gaudi, gaudi::detect_gaudi);
        spawn_backend!("aws-neuron", Backend::AwsNeuron, neuron::detect_aws_neuron);
        spawn_backend!(
            "intel-oneapi",
            Backend::IntelOneApi,
            intel_oneapi::detect_intel_oneapi
        );
        spawn_backend!(
            "qualcomm",
            Backend::Qualcomm,
            qualcomm::detect_qualcomm_ai100
        );

        for handle in handles {
            if let Ok((profiles, warnings)) = handle.join() {
                all_profiles.extend(profiles);
                all_warnings.extend(warnings);
            }
        }
    });

    // Post-pass: remove Vulkan GPUs if a dedicated CUDA or ROCm GPU was found.
    let has_dedicated = all_profiles.iter().any(|p| {
        matches!(
            p.accelerator,
            AcceleratorType::CudaGpu { .. } | AcceleratorType::RocmGpu { .. }
        )
    });
    if has_dedicated {
        all_profiles.retain(|p| !matches!(p.accelerator, AcceleratorType::VulkanGpu { .. }));
    }

    debug!(
        count = all_profiles.len(),
        warnings = all_warnings.len(),
        "accelerator detection complete"
    );
    AcceleratorRegistry {
        schema_version: crate::registry::SCHEMA_VERSION,
        profiles: all_profiles,
        warnings: all_warnings,
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Build a default CPU profile with detected system memory.
pub(crate) fn cpu_profile() -> AcceleratorProfile {
    AcceleratorProfile {
        accelerator: AcceleratorType::Cpu,
        available: true,
        memory_bytes: detect_cpu_memory(),
        compute_capability: None,
        driver_version: None,
    }
}

/// System memory from /proc/meminfo (fallback: 16 GiB).
fn detect_cpu_memory() -> u64 {
    if let Ok(info) = std::fs::read_to_string("/proc/meminfo") {
        for line in info.lines() {
            if line.starts_with("MemTotal:")
                && let Some(kb_str) = line.split_whitespace().nth(1)
                && let Ok(kb) = kb_str.parse::<u64>()
            {
                return kb.saturating_mul(1024);
            }
        }
    }
    // macOS fallback via safe command runner (absolute path, timeout).
    if let Ok(output) = command::run_tool("sysctl", &["-n", "hw.memsize"], command::DEFAULT_TIMEOUT)
        && let Ok(bytes) = output.stdout.trim().parse::<u64>()
    {
        return bytes;
    }
    tracing::debug!("could not read system memory, defaulting to 16 GiB");
    16 * 1024 * 1024 * 1024
}

/// Read a u64 from a sysfs file.
pub(super) fn read_sysfs_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}
