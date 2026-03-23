//! Intel NPU detection (Meteor Lake+) via sysfs.

use std::path::Path;

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_intel_npu(
    profiles: &mut Vec<AcceleratorProfile>,
    _warnings: &mut Vec<DetectionError>,
) {
    if Path::new("/sys/class/misc/intel_npu").exists() {
        debug!(
            device_id = 0,
            memory_mb = 2048,
            "Intel NPU detected via sysfs"
        );
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::IntelNpu,
            available: true,
            memory_bytes: 2 * 1024 * 1024 * 1024,
            ..Default::default()
        });
    }
}
