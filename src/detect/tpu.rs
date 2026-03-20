//! Google TPU detection via `/dev/accel*` and sysfs.

use std::path::Path;

use tracing::{debug, warn};

use crate::error::DetectionError;
use crate::hardware::{AcceleratorType, TpuVersion};
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_tpu(
    profiles: &mut Vec<AcceleratorProfile>,
    _warnings: &mut Vec<DetectionError>,
) {
    let dev_dir = Path::new("/dev");
    for entry in std::fs::read_dir(dev_dir).into_iter().flatten().flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("accel") {
            continue;
        }
        let suffix = &name_str[5..];
        if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let device_id: u32 = match suffix.parse() {
            Ok(id) => id,
            Err(_) => continue,
        };

        // Skip if this is an AMD XDNA device
        let driver_link = format!("/sys/class/accel/accel{}/device/driver", device_id);
        if let Ok(target) = std::fs::read_link(&driver_link)
            && target.to_string_lossy().contains("amdxdna")
        {
            continue;
        }

        let version = detect_tpu_version(device_id);
        let chip_count = detect_tpu_chip_count(device_id);
        let hbm = version
            .hbm_per_chip_bytes()
            .saturating_mul(chip_count as u64);

        debug!(device_id, %version, chip_count, "Google TPU detected");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::Tpu {
                device_id,
                chip_count,
                version,
            },
            available: true,
            memory_bytes: hbm,
            compute_capability: Some(format!("TPU {}", version)),
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

fn detect_tpu_version(device_id: u32) -> TpuVersion {
    let path = format!("/sys/class/accel/accel{}/device/tpu_version", device_id);
    if let Some(ver) = super::read_sysfs_string(std::path::Path::new(&path), 256) {
        let ver = ver.trim();
        if ver.contains("v5p") {
            return TpuVersion::V5p;
        } else if ver.contains("v5e") || ver.contains("v5litepod") {
            return TpuVersion::V5e;
        } else if ver.contains("v4") {
            return TpuVersion::V4;
        }
    }
    TpuVersion::V5e
}

fn detect_tpu_chip_count(device_id: u32) -> u32 {
    let path = format!("/sys/class/accel/accel{}/device/chip_count", device_id);
    if let Some(count) = super::read_sysfs_string(std::path::Path::new(&path), 64)
        && let Ok(n) = count.trim().parse::<u32>()
        && n > 0
    {
        return n;
    }
    warn!(device_id, "could not read TPU chip_count, defaulting to 1");
    1
}
