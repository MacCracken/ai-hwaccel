//! AMD XDNA / Ryzen AI NPU detection via sysfs.

use std::path::Path;

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_amd_xdna(
    profiles: &mut Vec<AcceleratorProfile>,
    _warnings: &mut Vec<DetectionError>,
) {
    let accel_dir = Path::new("/sys/class/accel");
    if !accel_dir.exists() {
        return;
    }
    for entry in std::fs::read_dir(accel_dir).into_iter().flatten().flatten() {
        let driver_link = entry.path().join("device/driver");
        if let Ok(target) = std::fs::read_link(&driver_link) {
            let driver_name = target
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if driver_name == "amdxdna" {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                let device_id: u32 = name_str
                    .strip_prefix("accel")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                debug!(
                    device_id,
                    memory_mb = 2048,
                    "AMD XDNA NPU detected via sysfs"
                );
                profiles.push(AcceleratorProfile {
                    accelerator: AcceleratorType::AmdXdnaNpu { device_id },
                    available: true,
                    memory_bytes: 2 * 1024 * 1024 * 1024,
                    compute_capability: None,
                    driver_version: None,
                    memory_bandwidth_gbps: None,
                    memory_used_bytes: None,
                    memory_free_bytes: None,
                    pcie_bandwidth_gbps: None,
                    numa_node: None,
                    temperature_c: None,
                    power_watts: None,
                    gpu_utilization_percent: None,
                });
            }
        }
    }
}
