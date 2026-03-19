//! Apple Metal GPU and Neural Engine (ANE) detection.
//!
//! On Linux (Asahi): reads `/proc/device-tree/compatible`.
//! On macOS: uses `system_profiler SPHardwareDataType` for chip name and memory.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

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

/// macOS detection via `system_profiler`. Returns `true` if on macOS.
fn detect_macos(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) -> bool {
    let output = match std::process::Command::new("system_profiler")
        .args(["SPHardwareDataType"])
        .output()
    {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            // system_profiler exists but failed — we're on macOS but something is wrong
            if o.status.code() != Some(127) {
                warnings.push(DetectionError::ToolFailed {
                    tool: "system_profiler".into(),
                    exit_code: o.status.code(),
                    stderr: String::from_utf8_lossy(&o.stderr).to_string(),
                });
            }
            return false;
        }
        Err(_) => return false, // not macOS
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let chip_name = extract_field(&stdout, "Chip");
    let memory_str = extract_field(&stdout, "Memory");

    // Parse memory (e.g. "36 GB" → bytes)
    let memory_bytes = memory_str
        .as_deref()
        .and_then(parse_memory_string)
        .unwrap_or(16 * 1024 * 1024 * 1024);

    // Only register if this looks like Apple Silicon (has a chip field)
    if chip_name.is_none() && !stdout.contains("Apple") {
        return true; // We're on macOS but not Apple Silicon (Intel Mac)
    }

    let compute_cap = chip_name.clone();

    debug!(
        chip = chip_name.as_deref().unwrap_or("unknown"),
        memory_gb = memory_bytes / (1024 * 1024 * 1024),
        "Apple Silicon detected via system_profiler"
    );

    // Metal GPU — unified memory, so it gets the full system memory
    profiles.push(AcceleratorProfile {
        accelerator: AcceleratorType::MetalGpu,
        available: true,
        memory_bytes,
        compute_capability: compute_cap.clone(),
        driver_version: None,
    });

    // ANE — shares unified memory but has its own budget (~4 GB typical)
    profiles.push(AcceleratorProfile {
        accelerator: AcceleratorType::AppleNpu,
        available: true,
        memory_bytes: estimate_ane_memory(&compute_cap),
        compute_capability: compute_cap,
        driver_version: None,
    });

    true
}

/// Linux (Asahi) detection via `/proc/device-tree/compatible`.
fn detect_linux_device_tree(profiles: &mut Vec<AcceleratorProfile>) {
    if let Ok(compat) = std::fs::read_to_string("/proc/device-tree/compatible")
        && compat.contains("apple")
    {
        debug!("Apple device detected via device-tree, registering Metal GPU + ANE");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::MetalGpu,
            available: true,
            memory_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        });
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::AppleNpu,
            available: true,
            memory_bytes: 4 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        });
    }
}

/// Extract a "Key: Value" field from system_profiler output.
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

/// Parse a memory string like "36 GB" or "128 GB" into bytes.
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

/// Estimate ANE memory budget based on chip generation.
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
