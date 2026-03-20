//! AMD ROCm GPU detection via sysfs.

use std::path::Path;

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::read_sysfs_u64;

pub(crate) fn detect_rocm(
    profiles: &mut Vec<AcceleratorProfile>,
    _warnings: &mut Vec<DetectionError>,
) {
    let drm = Path::new("/sys/class/drm");
    if !drm.exists() {
        return;
    }

    let mut device_id = 0u32;
    for entry in std::fs::read_dir(drm).into_iter().flatten().flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }

        let device_dir = entry.path().join("device");
        let driver_link = device_dir.join("driver");
        let driver_name = std::fs::read_link(&driver_link)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));

        if driver_name.as_deref() != Some("amdgpu") {
            continue;
        }

        let mem_total = read_sysfs_u64(&device_dir.join("mem_info_vram_total"))
            .unwrap_or(8 * 1024 * 1024 * 1024);
        let mem_used = read_sysfs_u64(&device_dir.join("mem_info_vram_used"));
        let mem_free = mem_used.map(|used| mem_total.saturating_sub(used));

        debug!(device_id, "AMD ROCm GPU detected via sysfs");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::RocmGpu { device_id },
            available: true,
            memory_bytes: mem_total,
            compute_capability: None,
            driver_version: None,
            memory_bandwidth_gbps: None,
            memory_used_bytes: mem_used,
            memory_free_bytes: mem_free,
            pcie_bandwidth_gbps: None,
            numa_node: None,
        });
        device_id += 1;
    }
}
