//! Cerebras Wafer-Scale Engine (WSE) detection via `/dev` and `cerebras_cli`.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::command::{DEFAULT_TIMEOUT, run_tool};

/// Default memory: 44 GB SRAM (WSE-3).
const DEFAULT_MEMORY_BYTES: u64 = 44 * 1024 * 1024 * 1024;

pub(crate) fn detect_cerebras_wse(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    // Try CLI first for richer info.
    match run_tool("cerebras_cli", &["system-info"], DEFAULT_TIMEOUT) {
        Ok(output) => {
            debug!(
                device_id = 0,
                memory_gb = 44,
                "Cerebras WSE detected via cerebras_cli"
            );
            profiles.push(AcceleratorProfile {
                accelerator: AcceleratorType::CerebrasWse { device_id: 0 },
                available: true,
                memory_bytes: parse_memory_from_cli(&output.stdout).unwrap_or(DEFAULT_MEMORY_BYTES),
                compute_capability: Some("WSE".into()),
                ..Default::default()
            });
            return;
        }
        Err(DetectionError::ToolNotFound { .. }) => {
            debug!("cerebras_cli not found on $PATH, skipping Cerebras CLI detection");
        }
        Err(_) => {}
    }

    // Fallback: check /dev/cerebras* devices.
    if super::has_dev_device("cerebras") {
        debug!(
            device_id = 0,
            memory_gb = 44,
            "Cerebras WSE detected via /dev"
        );
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::CerebrasWse { device_id: 0 },
            available: true,
            memory_bytes: DEFAULT_MEMORY_BYTES,
            compute_capability: Some("WSE".into()),
            ..Default::default()
        });
        return;
    }

    let _ = warnings;
}

/// Best-effort parse of memory from `cerebras_cli system-info` output.
pub(crate) fn parse_memory_from_cli(stdout: &str) -> Option<u64> {
    for line in stdout.lines() {
        let lower = line.to_lowercase();
        if lower.contains("memory") || lower.contains("sram") {
            // Look for a number followed by "gb" or "GB".
            for word in line.split_whitespace() {
                if let Some(num_str) = word.strip_suffix("GB").or_else(|| word.strip_suffix("gb"))
                    && let Ok(gb) = num_str.parse::<u64>()
                {
                    return Some(gb.saturating_mul(1024 * 1024 * 1024));
                }
            }
        }
    }
    None
}
