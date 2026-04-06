//! Disk I/O throughput detection via sysfs.
//!
//! Probes `/sys/block/*/queue/` to determine storage type (NVMe, SATA SSD, HDD)
//! and estimate sequential read bandwidth.

use std::path::Path;

use tracing::debug;

use crate::system_io::{StorageDevice, StorageKind};

/// Detect storage devices and estimate their throughput.
pub(crate) fn detect_storage() -> Vec<StorageDevice> {
    let mut devices = Vec::new();
    let block_dir = Path::new("/sys/block");
    let Ok(entries) = std::fs::read_dir(block_dir) else {
        return devices;
    };

    for entry in entries.flatten() {
        let os_name = entry.file_name();
        let name_ref = os_name.to_string_lossy();

        // Skip virtual devices (loop, dm, ram, etc.)
        if name_ref.starts_with("loop")
            || name_ref.starts_with("dm-")
            || name_ref.starts_with("ram")
            || name_ref.starts_with("zram")
            || name_ref.starts_with("sr")
            || name_ref.starts_with("fd")
            || name_ref.starts_with("md")
        {
            continue;
        }

        let queue_dir = entry.path().join("queue");
        if !queue_dir.exists() {
            continue;
        }

        let name = name_ref.into_owned();
        let kind = detect_storage_kind(&name, &queue_dir);
        let bandwidth_gbps = estimate_bandwidth(&kind, &queue_dir);

        debug!(device = %name, %kind, bandwidth_gbps, "storage device detected");
        devices.push(StorageDevice {
            name,
            kind,
            bandwidth_gbps,
        });
    }

    devices
}

/// Classify a block device as NVMe, SATA SSD, or HDD.
fn detect_storage_kind(name: &str, queue_dir: &Path) -> StorageKind {
    // NVMe devices have names like "nvme0n1"
    if name.starts_with("nvme") {
        return StorageKind::NVMe;
    }

    // Check rotational flag: 0 = SSD, 1 = HDD
    let rotational = super::read_sysfs_string(&queue_dir.join("rotational"), 64)
        .and_then(|s| s.trim().parse::<u32>().ok());

    match rotational {
        Some(0) => StorageKind::SataSsd,
        Some(1) => StorageKind::Hdd,
        _ => StorageKind::Unknown,
    }
}

/// Estimate sequential read bandwidth for a storage device.
///
/// Uses sysfs `max_hw_sectors_kb` and device type heuristics. These are
/// conservative estimates — real throughput depends on workload and queue depth.
fn estimate_bandwidth(kind: &StorageKind, queue_dir: &Path) -> f64 {
    // Try to read max_hw_sectors_kb for a rough capability indicator
    let _max_sectors_kb = super::read_sysfs_string(&queue_dir.join("max_hw_sectors_kb"), 64)
        .and_then(|s| s.trim().parse::<u64>().ok());

    // Use known typical throughput by device class
    match kind {
        StorageKind::NVMe => {
            // Typical NVMe: 3.5 GB/s (Gen3x4), up to 7 GB/s (Gen4x4)
            // Conservative: 3.5 GB/s
            3.5
        }
        StorageKind::SataSsd => {
            // SATA SSD: ~0.55 GB/s (SATA III max)
            0.55
        }
        StorageKind::Hdd => {
            // HDD: ~0.15 GB/s sequential
            0.15
        }
        StorageKind::Unknown => 0.5,
    }
}
