//! Apple Metal GPU and Neural Engine (ANE) detection.
//!
//! On Linux (Asahi): reads `/proc/device-tree/compatible`.
//! On macOS: uses `system_profiler SPHardwareDataType` for chip name and memory.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::command::{DEFAULT_TIMEOUT, run_tool};

const SYSTEM_PROFILER_ARGS: &[&str] = &["SPHardwareDataType"];

pub(crate) fn detect_metal_and_ane(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    // Try macOS detection first (system_profiler).
    if detect_macos(profiles, warnings) {
        return;
    }

    // Fallback: Linux (Asahi) device-tree detection.
    detect_linux_device_tree(profiles);
}

#[cfg(feature = "async-detect")]
pub(crate) async fn detect_metal_and_ane_async() -> super::DetectResult {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();

    // Try macOS detection via system_profiler (async CLI call).
    if let Ok(output) = super::command::run_tool_async("system_profiler", SYSTEM_PROFILER_ARGS, DEFAULT_TIMEOUT).await {
        if parse_system_profiler_output(&output.stdout, &mut profiles, &mut warnings) {
            return (profiles, warnings);
        }
    }

    // Fallback: Linux (Asahi) device-tree detection (sync sysfs, runs inline).
    detect_linux_device_tree(&mut profiles);
    (profiles, warnings)
}

/// macOS detection via `system_profiler`. Returns `true` if on macOS.
fn detect_macos(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) -> bool {
    let output = match run_tool("system_profiler", SYSTEM_PROFILER_ARGS, DEFAULT_TIMEOUT) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => return false, // not macOS
        Err(e) => {
            warnings.push(e);
            return false;
        }
    };
    parse_system_profiler_output(&output.stdout, profiles, warnings)
}

/// Parse `system_profiler SPHardwareDataType` output. Returns `true` if this
/// is a macOS system (even Intel Mac with no Apple Silicon).
pub(crate) fn parse_system_profiler_output(
    stdout: &str,
    profiles: &mut Vec<AcceleratorProfile>,
    _warnings: &mut Vec<DetectionError>,
) -> bool {
    let chip_name = extract_field(stdout, "Chip");
    let memory_str = extract_field(stdout, "Memory");

    let memory_bytes = memory_str
        .as_deref()
        .and_then(parse_memory_string)
        .unwrap_or(16 * 1024 * 1024 * 1024);

    if chip_name.is_none() && !stdout.contains("Apple") {
        return true; // macOS but Intel Mac
    }

    let compute_cap = chip_name.clone();

    debug!(
        chip = chip_name.as_deref().unwrap_or("unknown"),
        memory_gb = memory_bytes / (1024 * 1024 * 1024),
        "Apple Silicon detected via system_profiler"
    );

    profiles.push(AcceleratorProfile {
        accelerator: AcceleratorType::MetalGpu,
        available: true,
        memory_bytes,
        compute_capability: compute_cap.clone(),
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

    profiles.push(AcceleratorProfile {
        accelerator: AcceleratorType::AppleNpu,
        available: true,
        memory_bytes: estimate_ane_memory(&compute_cap),
        compute_capability: compute_cap,
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

    true
}

/// Linux (Asahi) detection via `/proc/device-tree/compatible`.
fn detect_linux_device_tree(profiles: &mut Vec<AcceleratorProfile>) {
    if let Some(compat) = super::read_sysfs_string(&std::path::Path::new("/proc/device-tree/compatible"), 4096)
        && compat.contains("apple")
    {
        debug!("Apple device detected via device-tree, registering Metal GPU + ANE");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::MetalGpu,
            available: true,
            memory_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
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
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::AppleNpu,
            available: true,
            memory_bytes: 4 * 1024 * 1024 * 1024,
            compute_capability: None,
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
    }
}

fn extract_field(output: &str, key: &str) -> Option<String> {
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(key)
            && let Some(value) = rest.strip_prefix(':')
        {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn parse_memory_string(s: &str) -> Option<u64> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }
    let amount: u64 = parts[0].parse().ok()?;
    let unit = parts[1].to_uppercase();
    match unit.as_str() {
        "GB" => Some(amount * 1024 * 1024 * 1024),
        "MB" => Some(amount * 1024 * 1024),
        "TB" => Some(amount * 1024 * 1024 * 1024 * 1024),
        _ => None,
    }
}

fn estimate_ane_memory(chip: &Option<String>) -> u64 {
    let chip = match chip {
        Some(c) => c.to_lowercase(),
        None => return 4 * 1024 * 1024 * 1024,
    };
    if chip.contains("m4") || chip.contains("m3") {
        8 * 1024 * 1024 * 1024
    } else if chip.contains("m2") {
        6 * 1024 * 1024 * 1024
    } else {
        4 * 1024 * 1024 * 1024
    }
}
