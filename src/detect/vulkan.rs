//! Vulkan compute device detection via `vulkaninfo`.

use tracing::debug;

use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::which_exists;

pub(crate) fn detect_vulkan(profiles: &mut Vec<AcceleratorProfile>) {
    if which_exists("vulkaninfo") {
        // Register a generic Vulkan device if the tool exists but only if we
        // didn't already find a CUDA or ROCm GPU (avoid double-counting).
        let has_dedicated_gpu = profiles.iter().any(|p| {
            matches!(
                p.accelerator,
                AcceleratorType::CudaGpu { .. } | AcceleratorType::RocmGpu { .. }
            )
        });

        if !has_dedicated_gpu {
            debug!("vulkaninfo found (no CUDA/ROCm), registering Vulkan GPU");
            profiles.push(AcceleratorProfile {
                accelerator: AcceleratorType::VulkanGpu {
                    device_id: 0,
                    device_name: "Unknown Vulkan Device".into(),
                },
                available: true,
                memory_bytes: 4 * 1024 * 1024 * 1024,
                compute_capability: None,
                driver_version: None,
            });
        }
    }
}
