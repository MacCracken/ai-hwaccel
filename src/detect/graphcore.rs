//! Graphcore IPU detection via `gc-info` CLI and `/dev`.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::command::{DEFAULT_TIMEOUT, run_tool};

/// Default memory: 900 MB SRAM per IPU (Bow-2000).
const DEFAULT_MEMORY_BYTES: u64 = 900 * 1024 * 1024;

pub(crate) fn detect_graphcore_ipu(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    // Try gc-info CLI for device 0.
    match run_tool("gc-info", &["-d", "0", "-i"], DEFAULT_TIMEOUT) {
        Ok(output) => {
            let memory = parse_memory_from_gcinfo(&output.stdout).unwrap_or(DEFAULT_MEMORY_BYTES);
            debug!(
                device_id = 0,
                memory_mb = memory / (1024 * 1024),
                "Graphcore IPU detected via gc-info"
            );
            profiles.push(AcceleratorProfile {
                accelerator: AcceleratorType::GraphcoreIpu { device_id: 0 },
                available: true,
                memory_bytes: memory,
                compute_capability: Some("IPU".into()),
                ..Default::default()
            });
            return;
        }
        Err(DetectionError::ToolNotFound { .. }) => {
            debug!("gc-info not found on $PATH, skipping Graphcore CLI detection");
        }
        Err(_) => {}
    }

    // Fallback: check /dev/ipu* devices.
    if super::has_dev_device("ipu") {
        debug!(
            device_id = 0,
            memory_mb = 900,
            "Graphcore IPU detected via /dev"
        );
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::GraphcoreIpu { device_id: 0 },
            available: true,
            memory_bytes: DEFAULT_MEMORY_BYTES,
            compute_capability: Some("IPU".into()),
            ..Default::default()
        });
        return;
    }

    let _ = warnings;
}

/// Best-effort parse of memory from `gc-info` JSON or text output.
pub(crate) fn parse_memory_from_gcinfo(stdout: &str) -> Option<u64> {
    // Try JSON parse first.
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(stdout) {
        if let Some(mem) = val.get("memory").and_then(|m| m.as_u64()) {
            return Some(mem);
        }
        if let Some(mem) = val.get("sram_size").and_then(|m| m.as_u64()) {
            return Some(mem);
        }
    }
    // Fallback: line-based parsing.
    for line in stdout.lines() {
        let lower = line.to_lowercase();
        if lower.contains("memory") || lower.contains("sram") {
            for word in line.split_whitespace() {
                if let Some(num_str) = word.strip_suffix("MB").or_else(|| word.strip_suffix("mb"))
                    && let Ok(mb) = num_str.parse::<u64>()
                {
                    return Some(mb.saturating_mul(1024 * 1024));
                }
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
