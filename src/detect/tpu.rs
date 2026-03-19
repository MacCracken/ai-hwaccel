//! Google TPU detection via `/dev/accel*` and sysfs.

use std::path::Path;

use tracing::debug;

use crate::hardware::{AcceleratorType, TpuVersion};
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_tpu(profiles: &mut Vec<AcceleratorProfile>) {
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
        let device_id: u32 = suffix.parse().unwrap_or(0);

        // Skip if this is an AMD XDNA device
        let driver_link = format!("/sys/class/accel/accel{}/device/driver", device_id);
        if let Ok(target) = std::fs::read_link(&driver_link)
            && target.to_string_lossy().contains("amdxdna")
        {
            continue;
        }

        let version = detect_tpu_version(device_id);
        let chip_count = detect_tpu_chip_count(device_id);
        let hbm = version.hbm_per_chip_bytes() * chip_count as u64;

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
        });
    }
}

fn detect_tpu_version(device_id: u32) -> TpuVersion {
    let path = format!("/sys/class/accel/accel{}/device/tpu_version", device_id);
    if let Ok(ver) = std::fs::read_to_string(&path) {
        let ver = ver.trim();
        if ver.contains("v5p") {
            return TpuVersion::V5p;
        } else if ver.contains("v5e") || ver.contains("v5litepod") {
            return TpuVersion::V5e;
        } else if ver.contains("v4") {
            return TpuVersion::V4;
        }
    }
    TpuVersion::V5e // default — most common cloud TPU
}

fn detect_tpu_chip_count(device_id: u32) -> u32 {
    let path = format!("/sys/class/accel/accel{}/device/chip_count", device_id);
    if let Ok(count) = std::fs::read_to_string(&path)
        && let Ok(n) = count.trim().parse::<u32>()
        && n > 0
    {
        return n;
    }
    1
}
