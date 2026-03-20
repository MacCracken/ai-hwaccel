//! Cerebras Wafer-Scale Engine (WSE) detection via `/dev` and `cerebras_cli`.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::command::{run_tool, DEFAULT_TIMEOUT};

/// Default memory: 44 GB SRAM (WSE-3).
const DEFAULT_MEMORY_BYTES: u64 = 44 * 1024 * 1024 * 1024;

pub(crate) fn detect_cerebras_wse(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    // Try CLI first for richer info.
    if let Ok(output) = run_tool("cerebras_cli", &["system-info"], DEFAULT_TIMEOUT) {
        debug!("Cerebras WSE detected via cerebras_cli");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::CerebrasWse { device_id: 0 },
            available: true,
            memory_bytes: parse_memory_from_cli(&output.stdout).unwrap_or(DEFAULT_MEMORY_BYTES),
            compute_capability: Some("WSE".into()),
            driver_version: None,
            memory_bandwidth_gbps: None,
            memory_used_bytes: None,
            memory_free_bytes: None,
            pcie_bandwidth_gbps: None,
            numa_node: None,
        });
        return;
    }

    // Fallback: check /dev/cerebras* devices.
    for entry in std::fs::read_dir("/dev").into_iter().flatten().flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("cerebras") {
            debug!("Cerebras WSE detected via /dev");
            profiles.push(AcceleratorProfile {
                accelerator: AcceleratorType::CerebrasWse { device_id: 0 },
                available: true,
                memory_bytes: DEFAULT_MEMORY_BYTES,
                compute_capability: Some("WSE".into()),
                driver_version: None,
                memory_bandwidth_gbps: None,
                memory_used_bytes: None,
                memory_free_bytes: None,
                pcie_bandwidth_gbps: None,
                numa_node: None,
            });
            return;
        }
    }

    let _ = warnings;
}

/// Best-effort parse of memory from `cerebras_cli system-info` output.
fn parse_memory_from_cli(stdout: &str) -> Option<u64> {
    for line in stdout.lines() {
        let lower = line.to_lowercase();
        if lower.contains("memory") || lower.contains("sram") {
            // Look for a number followed by "gb" or "GB".
            for word in line.split_whitespace() {
                if let Some(num_str) = word.strip_suffix("GB").or_else(|| word.strip_suffix("gb")) {
                    if let Ok(gb) = num_str.parse::<u64>() {
                        return Some(gb * 1024 * 1024 * 1024);
                    }
                }
            }
        }
    }
    None
}
