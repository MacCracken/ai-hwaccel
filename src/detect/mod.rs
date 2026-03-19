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
use crate::registry::AcceleratorRegistry;

impl AcceleratorRegistry {
    /// Probes the system for all available accelerators.
    ///
    /// Detection is best-effort: missing tools or sysfs entries simply mean
    /// the corresponding accelerator is not registered.
    pub fn detect() -> Self {
        let mut profiles = vec![cpu_profile()];

        cuda::detect_cuda(&mut profiles);
        rocm::detect_rocm(&mut profiles);
        apple::detect_metal_and_ane(&mut profiles);
        vulkan::detect_vulkan(&mut profiles);
        intel_npu::detect_intel_npu(&mut profiles);
        amd_xdna::detect_amd_xdna(&mut profiles);
        tpu::detect_tpu(&mut profiles);
        gaudi::detect_gaudi(&mut profiles);
        neuron::detect_aws_neuron(&mut profiles);
        intel_oneapi::detect_intel_oneapi(&mut profiles);
        qualcomm::detect_qualcomm_ai100(&mut profiles);

        debug!(count = profiles.len(), "accelerator detection complete");
        Self::from_profiles(profiles)
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
    16 * 1024 * 1024 * 1024
}

/// Check if an executable is on `$PATH`.
pub(super) fn which_exists(name: &str) -> bool {
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            if Path::new(dir).join(name).exists() {
                return true;
            }
        }
    }
    false
}

/// Read a u64 from a sysfs file.
pub(super) fn read_sysfs_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}
