//! Groq LPU (Language Processing Unit) detection via `/dev`.
//!
//! The Groq driver is not yet publicly available; this is a placeholder
//! that checks for `/dev/groq*` device nodes.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

/// Default memory: 230 MB SRAM per LPU chip.
const DEFAULT_MEMORY_BYTES: u64 = 230 * 1024 * 1024;

pub(crate) fn detect_groq_lpu(
    profiles: &mut Vec<AcceleratorProfile>,
    _warnings: &mut Vec<DetectionError>,
) {
    if super::has_dev_device("groq") {
        debug!(device_id = 0, memory_mb = 230, "Groq LPU detected via /dev");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::GroqLpu { device_id: 0 },
            available: true,
            memory_bytes: DEFAULT_MEMORY_BYTES,
            compute_capability: Some("LPU".into()),
            ..Default::default()
        });
    }
}
