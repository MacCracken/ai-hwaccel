//! Intel Gaudi (Habana Labs HPU) detection via `hl-smi`.

use tracing::{debug, trace};

use crate::error::DetectionError;
use crate::hardware::{AcceleratorType, GaudiGeneration};
use crate::profile::AcceleratorProfile;

use super::command::{DEFAULT_TIMEOUT, run_tool, validate_device_id, validate_memory_mb};

const HL_SMI_ARGS: &[&str] = &[
    "--query-aip=index,name,memory.total,memory.free",
    "--format=csv,noheader,nounits",
];

pub(crate) fn detect_gaudi(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    let output = match run_tool("hl-smi", HL_SMI_ARGS, DEFAULT_TIMEOUT) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => {
            debug!("hl-smi not found on $PATH, skipping Gaudi detection");
            return;
        }
        Err(e) => {
            warnings.push(e);
            return;
        }
    };
    parse_gaudi_output(&output.stdout, profiles, warnings);
}

#[cfg(feature = "async-detect")]
pub(crate) async fn detect_gaudi_async() -> super::DetectResult {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let output = match super::command::run_tool_async("hl-smi", HL_SMI_ARGS, DEFAULT_TIMEOUT).await
    {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => {
            debug!("hl-smi not found on $PATH, skipping Gaudi detection");
            return (profiles, warnings);
        }
        Err(e) => {
            warnings.push(e);
            return (profiles, warnings);
        }
    };
    parse_gaudi_output(&output.stdout, &mut profiles, &mut warnings);
    (profiles, warnings)
}

pub fn parse_gaudi_output(
    stdout: &str,
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    for line in stdout.lines() {
        trace!(line, "parsing hl-smi CSV line");
        let parts: Vec<&str> = line.split(',').take(20).map(|s| s.trim()).collect();
        if parts.len() < 4 {
            warnings.push(DetectionError::ParseError {
                backend: "gaudi".into(),
                message: format!("expected 4 CSV fields, got {}: {}", parts.len(), line),
            });
            continue;
        }
        let device_id = match validate_device_id(parts[0], "gaudi") {
            Ok(id) => id,
            Err(e) => {
                warnings.push(e);
                continue;
            }
        };
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
            ..Default::default()
        });
    }
}
