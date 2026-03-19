//! Apple Metal GPU and Neural Engine (ANE) detection.

use tracing::debug;

use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_metal_and_ane(profiles: &mut Vec<AcceleratorProfile>) {
    if let Ok(compat) = std::fs::read_to_string("/proc/device-tree/compatible")
        && compat.contains("apple")
    {
        debug!("Apple device detected, registering Metal GPU + ANE");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::MetalGpu,
            available: true,
            memory_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        });
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::AppleNpu,
            available: true,
            memory_bytes: 4 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        });
    }
}
