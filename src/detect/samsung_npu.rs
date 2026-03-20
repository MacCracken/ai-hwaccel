//! Samsung Exynos AI NPU detection via sysfs and `/dev`.

use std::path::Path;

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

/// Default memory: 2 GB shared.
const DEFAULT_MEMORY_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub(crate) fn detect_samsung_npu(
    profiles: &mut Vec<AcceleratorProfile>,
    _warnings: &mut Vec<DetectionError>,
) {
    if Path::new("/sys/class/misc/samsung_npu").exists() {
        debug!("Samsung NPU detected via sysfs");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::SamsungNpu { device_id: 0 },
            available: true,
            memory_bytes: DEFAULT_MEMORY_BYTES,
            compute_capability: Some("Exynos AI".into()),
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
        return;
    }

    for entry in std::fs::read_dir("/dev").into_iter().flatten().flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("samsung_npu") {
            debug!("Samsung NPU detected via /dev");
            profiles.push(AcceleratorProfile {
                accelerator: AcceleratorType::SamsungNpu { device_id: 0 },
                available: true,
                memory_bytes: DEFAULT_MEMORY_BYTES,
                compute_capability: Some("Exynos AI".into()),
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
            return;
        }
    }
}
