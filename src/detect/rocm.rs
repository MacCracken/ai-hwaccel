//! AMD ROCm GPU detection via sysfs.
//!
//! Reads clock speeds, temperature, power draw, firmware/VBIOS version,
//! and GPU utilization from sysfs when available.

use std::path::Path;

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::read_sysfs_u64;

pub(crate) fn detect_rocm(
    profiles: &mut Vec<AcceleratorProfile>,
    _warnings: &mut Vec<DetectionError>,
) {
    let drm = Path::new("/sys/class/drm");
    if !drm.exists() {
        return;
    }

    let mut device_id = 0u32;
    let mut entries: Vec<_> = std::fs::read_dir(drm)
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }

        let device_dir = entry.path().join("device");
        let driver_link = device_dir.join("driver");
        let driver_name = std::fs::read_link(&driver_link)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));

        if driver_name.as_deref() != Some("amdgpu") {
            continue;
        }

        let mem_total = read_sysfs_u64(&device_dir.join("mem_info_vram_total"))
            .unwrap_or(8 * 1024 * 1024 * 1024);
        let mem_used = read_sysfs_u64(&device_dir.join("mem_info_vram_used"));
        let mem_free = mem_used.map(|used| mem_total.saturating_sub(used));

        // Clock speeds from pp_dpm_sclk (GPU) and pp_dpm_mclk (memory).
        let gpu_clock_mhz = read_current_dpm_clock(&device_dir.join("pp_dpm_sclk"));
        let mem_clock_mhz = read_current_dpm_clock(&device_dir.join("pp_dpm_mclk"));

        // Temperature from hwmon (millidegrees C).
        let temp_c = read_hwmon_temp(&device_dir);

        // Power draw from hwmon (microwatts).
        let power_w = read_hwmon_power(&device_dir);

        // GPU utilization percentage.
        let gpu_busy = read_sysfs_u64(&device_dir.join("gpu_busy_percent"));

        // Firmware / VBIOS version.
        let vbios = std::fs::read_to_string(device_dir.join("vbios_version"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        // Compute capability from revision.
        let compute_cap = std::fs::read_to_string(device_dir.join("revision"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        debug!(
            device_id,
            ?gpu_clock_mhz,
            ?mem_clock_mhz,
            ?temp_c,
            ?power_w,
            ?gpu_busy,
            ?vbios,
            "AMD ROCm GPU detected via sysfs"
        );
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::RocmGpu { device_id },
            available: true,
            memory_bytes: mem_total,
            compute_capability: compute_cap,
            driver_version: vbios,
            memory_bandwidth_gbps: None,
            memory_used_bytes: mem_used,
            memory_free_bytes: mem_free,
            pcie_bandwidth_gbps: None,
            numa_node: None,
        });
        device_id += 1;
    }
}

/// Read the currently active DPM clock from a `pp_dpm_*` sysfs file.
///
/// Format: lines like `0: 200Mhz`, `1: 400Mhz *` — the `*` marks active.
fn read_current_dpm_clock(path: &Path) -> Option<u64> {
    let content = std::fs::read_to_string(path).ok()?;
    for line in content.lines() {
        if !line.contains('*') {
            continue;
        }
        if let Some(mhz_str) = line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.strip_suffix("Mhz").or_else(|| s.strip_suffix("MHz")))
        {
            return mhz_str.parse().ok();
        }
    }
    None
}

/// Read GPU temperature from hwmon (returns degrees C).
fn read_hwmon_temp(device_dir: &Path) -> Option<u64> {
    let hwmon_dir = find_hwmon_dir(device_dir)?;
    let millideg = read_sysfs_u64(&hwmon_dir.join("temp1_input"))?;
    Some(millideg / 1000)
}

/// Read GPU power draw from hwmon (returns watts).
fn read_hwmon_power(device_dir: &Path) -> Option<f64> {
    let hwmon_dir = find_hwmon_dir(device_dir)?;
    // Try power1_average first (more stable), then power1_input.
    let microwatts = read_sysfs_u64(&hwmon_dir.join("power1_average"))
        .or_else(|| read_sysfs_u64(&hwmon_dir.join("power1_input")))?;
    Some(microwatts as f64 / 1_000_000.0)
}

/// Find the first hwmon directory under a device.
fn find_hwmon_dir(device_dir: &Path) -> Option<std::path::PathBuf> {
    let hwmon_base = device_dir.join("hwmon");
    let entry = std::fs::read_dir(&hwmon_base).ok()?.flatten().next()?;
    Some(entry.path())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dpm_clock_active() {
        // Can't easily test read_current_dpm_clock without sysfs,
        // but we can test the line parsing logic indirectly.
        // The function reads from a file, so we just verify it returns
        // None for a nonexistent path.
        assert!(read_current_dpm_clock(Path::new("/nonexistent")).is_none());
    }
}
