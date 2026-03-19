//! Intel NPU detection (Meteor Lake+) via sysfs.

use std::path::Path;

use tracing::debug;

use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_intel_npu(profiles: &mut Vec<AcceleratorProfile>) {
    if Path::new("/sys/class/misc/intel_npu").exists() {
        debug!("Intel NPU detected via sysfs");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::IntelNpu,
            available: true,
            memory_bytes: 2 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        });
    }
}
