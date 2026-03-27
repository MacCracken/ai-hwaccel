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
    if let Ok(output) =
        super::command::run_tool_async("system_profiler", SYSTEM_PROFILER_ARGS, DEFAULT_TIMEOUT)
            .await
        && parse_system_profiler_output(&output.stdout, &mut profiles, &mut warnings)
    {
        return (profiles, warnings);
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
        Err(DetectionError::ToolNotFound { .. }) => {
            debug!("system_profiler not found on $PATH, skipping Apple detection");
            return false; // not macOS
        }
        Err(e) => {
            warnings.push(e);
            return false;
        }
    };

    let is_mac = parse_system_profiler_output(&output.stdout, profiles, warnings);
    if !is_mac {
        return false;
    }

    // Enrich GPU profiles with Metal feature set from SPDisplaysDataType -json.
    enrich_with_displays_json(profiles);

    // Enrich CPU info from sysctl (core topology).
    enrich_with_sysctl();

    true
}

/// Enrich Metal GPU profiles with Metal family and GPU core count from
/// `system_profiler SPDisplaysDataType -json`.
fn enrich_with_displays_json(profiles: &mut [AcceleratorProfile]) {
    let json_output = match run_tool(
        "system_profiler",
        &["SPDisplaysDataType", "-json"],
        DEFAULT_TIMEOUT,
    ) {
        Ok(o) => o,
        Err(_) => return,
    };

    let gpus = parse_displays_json(&json_output.stdout);
    if gpus.is_empty() {
        return;
    }

    // Enrich the first Metal GPU profile with display info.
    if let Some(metal_profile) = profiles
        .iter_mut()
        .find(|p| matches!(p.accelerator, AcceleratorType::MetalGpu))
        && let Some(gpu) = gpus.first()
    {
        // Set device_name from display info if not already set.
        if metal_profile.device_name.is_none() {
            metal_profile.device_name = Some(gpu.name.clone());
        }
        // Append Metal family to compute_capability.
        if let Some(ref metal_family) = gpu.metal_family {
            let existing = metal_profile
                .compute_capability
                .as_deref()
                .unwrap_or("Metal GPU");
            metal_profile.compute_capability = Some(format!("{}, {}", existing, metal_family));
        }
    }
}

/// Read macOS CPU topology from `sysctl` for logging/enrichment.
fn enrich_with_sysctl() {
    let sysctl_keys = &[
        "hw.memsize",
        "hw.ncpu",
        "hw.cpufrequency",
        "hw.perflevel0.logicalcpu",
        "hw.perflevel1.logicalcpu",
    ];
    let args: Vec<&str> = std::iter::once("-a")
        .chain(sysctl_keys.iter().copied())
        .collect();

    if let Ok(output) = run_tool("sysctl", &args, DEFAULT_TIMEOUT) {
        let info = parse_sysctl_output(&output.stdout);
        debug!(
            cpu_count = info.cpu_count,
            perf_cores = info.perf_cores,
            eff_cores = info.eff_cores,
            cpu_freq_hz = info.cpu_freq_hz,
            "macOS CPU topology from sysctl"
        );
    }
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
        ..Default::default()
    });

    profiles.push(AcceleratorProfile {
        accelerator: AcceleratorType::AppleNpu,
        available: true,
        memory_bytes: estimate_ane_memory(&compute_cap),
        compute_capability: compute_cap,
        ..Default::default()
    });

    true
}

/// Linux (Asahi) detection via `/proc/device-tree/compatible`.
fn detect_linux_device_tree(profiles: &mut Vec<AcceleratorProfile>) {
    if let Some(compat) =
        super::read_sysfs_string(std::path::Path::new("/proc/device-tree/compatible"), 4096)
        && compat.contains("apple")
    {
        debug!(
            memory_gb = 16,
            "Apple device detected via device-tree, registering Metal GPU + ANE"
        );
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::MetalGpu,
            available: true,
            memory_bytes: 16 * 1024 * 1024 * 1024,
            ..Default::default()
        });
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::AppleNpu,
            available: true,
            memory_bytes: 4 * 1024 * 1024 * 1024,
            ..Default::default()
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
                // Cap field length to prevent unbounded allocation.
                return Some(value.chars().take(256).collect());
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
        "GB" => Some(amount.saturating_mul(1024 * 1024 * 1024)),
        "MB" => Some(amount.saturating_mul(1024 * 1024)),
        "TB" => Some(amount.saturating_mul(1024 * 1024 * 1024 * 1024)),
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

// ---------------------------------------------------------------------------
// system_profiler SPDisplaysDataType -json — GPU enumeration
// ---------------------------------------------------------------------------

/// GPU information extracted from `system_profiler SPDisplaysDataType -json`.
#[derive(Debug)]
#[allow(dead_code)] // fields used by tests and future enrichment
pub(crate) struct MacGpuInfo {
    pub name: String,
    pub vendor: String,
    pub metal_family: Option<String>,
    pub cores: Option<u32>,
    /// VRAM in bytes (discrete GPUs report this; integrated use system RAM).
    pub vram_bytes: Option<u64>,
}

/// Parse `system_profiler SPDisplaysDataType -json` output.
///
/// Returns a list of detected GPUs with Metal feature set information.
/// On Apple Silicon, the integrated GPU shares system memory.
/// On Intel Macs with discrete GPUs, VRAM is reported separately.
pub(crate) fn parse_displays_json(json_str: &str) -> Vec<MacGpuInfo> {
    let parsed: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut gpus = Vec::new();

    // Top-level: { "SPDisplaysDataType": [ { ... }, ... ] }
    let displays = match parsed.get("SPDisplaysDataType").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return gpus,
    };

    for display in displays {
        let name = display
            .get("sppci_model")
            .or_else(|| display.get("_name"))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown GPU")
            .to_string();

        let vendor = display
            .get("sppci_vendor")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        let metal_family = display
            .get("spdisplays_metal_family")
            .and_then(|v| v.as_str())
            .map(String::from);

        let cores = display
            .get("sppci_cores")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<u32>().ok())
            .or_else(|| {
                display
                    .get("sppci_cores")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32)
            });

        // VRAM: discrete GPUs report "sppci_vram" as "8192 MB" or similar.
        let vram_bytes = display
            .get("sppci_vram")
            .and_then(|v| v.as_str())
            .and_then(parse_memory_string);

        gpus.push(MacGpuInfo {
            name,
            vendor,
            metal_family,
            cores,
            vram_bytes,
        });
    }

    gpus
}

// ---------------------------------------------------------------------------
// sysctl-based CPU topology (macOS)
// ---------------------------------------------------------------------------

/// CPU topology extracted from macOS `sysctl` output.
#[derive(Debug)]
pub(crate) struct MacCpuInfo {
    /// Total system memory in bytes.
    pub memory_bytes: Option<u64>,
    /// Total number of logical CPUs.
    pub cpu_count: Option<u32>,
    /// CPU frequency in Hz.
    pub cpu_freq_hz: Option<u64>,
    /// Number of performance cores (Apple Silicon).
    pub perf_cores: Option<u32>,
    /// Number of efficiency cores (Apple Silicon).
    pub eff_cores: Option<u32>,
}

/// Parse key=value output from `sysctl` on macOS.
///
/// Expected input is the combined output of:
/// ```text
/// hw.memsize: 34359738368
/// hw.ncpu: 12
/// hw.cpufrequency: 3200000000
/// hw.perflevel0.logicalcpu: 8
/// hw.perflevel1.logicalcpu: 4
/// ```
pub(crate) fn parse_sysctl_output(output: &str) -> MacCpuInfo {
    let mut info = MacCpuInfo {
        memory_bytes: None,
        cpu_count: None,
        cpu_freq_hz: None,
        perf_cores: None,
        eff_cores: None,
    };

    for line in output.lines() {
        let trimmed = line.trim();
        if let Some((key, val)) = trimmed.split_once(':') {
            let key = key.trim();
            let val = val.trim();
            match key {
                "hw.memsize" => info.memory_bytes = val.parse().ok(),
                "hw.ncpu" => info.cpu_count = val.parse().ok(),
                "hw.cpufrequency" => info.cpu_freq_hz = val.parse().ok(),
                "hw.perflevel0.logicalcpu" => info.perf_cores = val.parse().ok(),
                "hw.perflevel1.logicalcpu" => info.eff_cores = val.parse().ok(),
                _ => {}
            }
        }
    }

    info
}
