//! Hardware detection: probes sysfs, /dev, and PATH tools to discover accelerators.

mod amd_xdna;
mod apple;
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

use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;
use crate::registry::{AcceleratorRegistry, Backend, DetectBuilder};

impl AcceleratorRegistry {
    /// Probes the system for all available accelerators.
    ///
    /// Detection is best-effort: missing tools or sysfs entries simply mean
    /// the corresponding accelerator is not registered. Non-fatal issues are
    /// collected in [`AcceleratorRegistry::warnings`].
    pub fn detect() -> Self {
        detect_with_builder(DetectBuilder::new())
    }
}

/// Run detection with a builder's backend selection.
pub(crate) fn detect_with_builder(builder: DetectBuilder) -> AcceleratorRegistry {
    let mut profiles = vec![cpu_profile()];
    let mut warnings = Vec::new();

    macro_rules! run_backend {
        ($backend:expr, $detect_fn:expr) => {
            if builder.backend_enabled($backend) {
                $detect_fn(&mut profiles, &mut warnings);
            }
        };
    }

    run_backend!(Backend::Cuda, cuda::detect_cuda);
    run_backend!(Backend::Rocm, rocm::detect_rocm);
    run_backend!(Backend::Apple, apple::detect_metal_and_ane);
    run_backend!(Backend::Vulkan, vulkan::detect_vulkan);
    run_backend!(Backend::IntelNpu, intel_npu::detect_intel_npu);
    run_backend!(Backend::AmdXdna, amd_xdna::detect_amd_xdna);
    run_backend!(Backend::Tpu, tpu::detect_tpu);
    run_backend!(Backend::Gaudi, gaudi::detect_gaudi);
    run_backend!(Backend::AwsNeuron, neuron::detect_aws_neuron);
    run_backend!(Backend::IntelOneApi, intel_oneapi::detect_intel_oneapi);
    run_backend!(Backend::Qualcomm, qualcomm::detect_qualcomm_ai100);

    debug!(
        count = profiles.len(),
        warnings = warnings.len(),
        "accelerator detection complete"
    );
    AcceleratorRegistry { profiles, warnings }
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
