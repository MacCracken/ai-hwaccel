//! Intel Gaudi (Habana Labs HPU) detection via `hl-smi`.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::{AcceleratorType, GaudiGeneration};
use crate::profile::AcceleratorProfile;

use super::command::{DEFAULT_TIMEOUT, run_tool, validate_memory_mb};

pub(crate) fn detect_gaudi(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    let output = match run_tool(
        "hl-smi",
        &[
            "--query-aip=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        DEFAULT_TIMEOUT,
    ) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => return,
        Err(e) => {
            warnings.push(e);
            return;
        }
    };

    for line in output.stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }
        let device_id: u32 = parts[0].parse().unwrap_or(0);
        let name = parts[1].to_lowercase();
        let mem_total_mb = match validate_memory_mb(parts[2], "gaudi") {
            Ok(mb) => mb,
            Err(e) => {
                warnings.push(e);
                continue;
            }
        };

        let generation = if name.contains("gaudi3") || name.contains("hl-325") {
            GaudiGeneration::Gaudi3
        } else {
            GaudiGeneration::Gaudi2
        };

        debug!(device_id, %generation, "Intel Gaudi HPU detected");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::Gaudi {
                device_id,
                generation,
            },
            available: true,
            memory_bytes: mem_total_mb * 1024 * 1024,
            compute_capability: Some(generation.to_string()),
            driver_version: None,
        });
    }
}
