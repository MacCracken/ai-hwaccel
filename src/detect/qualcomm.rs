//! Qualcomm Cloud AI 100 detection via sysfs and `/dev`.

use std::path::Path;

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_qualcomm_ai100(
    profiles: &mut Vec<AcceleratorProfile>,
    _warnings: &mut Vec<DetectionError>,
) {
    if Path::new("/sys/class/qaic").exists() {
        debug!(
            device_id = 0,
            memory_gb = 32,
            "Qualcomm Cloud AI 100 detected via sysfs"
        );
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::QualcommAi100 { device_id: 0 },
            available: true,
            memory_bytes: 32 * 1024 * 1024 * 1024,
            compute_capability: Some("AI 100".into()),
            ..Default::default()
        });
        return;
    }

    if super::has_dev_device("qaic_") {
        debug!(
            device_id = 0,
            memory_gb = 32,
            "Qualcomm Cloud AI 100 detected via /dev"
        );
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::QualcommAi100 { device_id: 0 },
            available: true,
            memory_bytes: 32 * 1024 * 1024 * 1024,
            compute_capability: Some("AI 100".into()),
            ..Default::default()
        });
    }
}
