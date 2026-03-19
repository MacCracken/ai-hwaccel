//! Hardware detection: probes sysfs, /dev, and PATH tools to discover accelerators.

mod amd_xdna;
mod apple;
pub(crate) mod command;
mod cuda;
mod gaudi;
mod intel_npu;
mod intel_oneapi;
mod neuron;
mod qualcomm;
mod rocm;
mod tpu;
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
    pub fn detect() -> Self {
        detect_with_builder(DetectBuilder::new())
    }
}

/// Run detection with a builder's backend selection.
///
/// Backends are executed in parallel via `std::thread::scope`. Results are
/// merged after all threads complete. Post-processing deduplicates Vulkan
/// GPUs when a dedicated CUDA/ROCm driver was also found.
pub(crate) fn detect_with_builder(builder: DetectBuilder) -> AcceleratorRegistry {
    let mut all_profiles = vec![cpu_profile()];
    let mut all_warnings: Vec<DetectionError> = Vec::new();

    // Spawn all enabled backends in parallel.
    std::thread::scope(|s| {
        let mut handles: Vec<std::thread::ScopedJoinHandle<'_, DetectResult>> = Vec::new();

        macro_rules! spawn_backend {
            ($backend:expr, $detect_fn:expr) => {
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

        spawn_backend!(Backend::Cuda, cuda::detect_cuda);
        spawn_backend!(Backend::Rocm, rocm::detect_rocm);
        spawn_backend!(Backend::Apple, apple::detect_metal_and_ane);
        spawn_backend!(Backend::Vulkan, vulkan::detect_vulkan);
        spawn_backend!(Backend::IntelNpu, intel_npu::detect_intel_npu);
        spawn_backend!(Backend::AmdXdna, amd_xdna::detect_amd_xdna);
        spawn_backend!(Backend::Tpu, tpu::detect_tpu);
        spawn_backend!(Backend::Gaudi, gaudi::detect_gaudi);
        spawn_backend!(Backend::AwsNeuron, neuron::detect_aws_neuron);
        spawn_backend!(Backend::IntelOneApi, intel_oneapi::detect_intel_oneapi);
        spawn_backend!(Backend::Qualcomm, qualcomm::detect_qualcomm_ai100);

        for handle in handles {
            if let Ok((profiles, warnings)) = handle.join() {
                all_profiles.extend(profiles);
                all_warnings.extend(warnings);
            }
        }
    });

    // Post-pass: remove Vulkan GPUs if a dedicated CUDA or ROCm GPU was found
    // (avoids double-counting the same physical device).
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
            if line.starts_with("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(kb_str) = parts.get(1)
                    && let Ok(kb) = kb_str.parse::<u64>()
                {
                    return kb * 1024;
                }
            }
        }
    }
    // macOS fallback: sysctl hw.memsize
    if let Ok(output) = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        && output.status.success()
        && let Ok(bytes) = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<u64>()
    {
        return bytes;
    }
    16 * 1024 * 1024 * 1024
}

/// Read a u64 from a sysfs file.
pub(super) fn read_sysfs_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}
