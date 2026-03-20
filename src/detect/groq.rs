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
    for entry in std::fs::read_dir("/dev").into_iter().flatten().flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("groq") {
            debug!(device_id = 0, memory_mb = 230, "Groq LPU detected via /dev");
            profiles.push(AcceleratorProfile {
                accelerator: AcceleratorType::GroqLpu { device_id: 0 },
                available: true,
                memory_bytes: DEFAULT_MEMORY_BYTES,
                compute_capability: Some("LPU".into()),
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
