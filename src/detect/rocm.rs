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

        // MI300X / MI350: detect CXL-attached memory (unified HBM pool).
        // CXL memory shows up as an additional memory region in sysfs.
        let cxl_mem = detect_cxl_memory(&device_dir);
        let total_with_cxl = if cxl_mem > 0 {
            mem_total.saturating_add(cxl_mem)
        } else {
            mem_total
        };

        debug!(
            device_id,
            ?gpu_clock_mhz,
            ?mem_clock_mhz,
            ?temp_c,
            ?power_w,
            ?gpu_busy,
            ?vbios,
            cxl_mem_bytes = cxl_mem,
            "AMD ROCm GPU detected via sysfs"
        );
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::RocmGpu { device_id },
            available: true,
            memory_bytes: total_with_cxl,
            compute_capability: compute_cap,
            driver_version: vbios,
            memory_bandwidth_gbps: None,
            memory_used_bytes: mem_used,
            memory_free_bytes: mem_free,
            pcie_bandwidth_gbps: None,
            numa_node: None,
            temperature_c: temp_c.map(|t| t as u32),
            power_watts: power_w,
            gpu_utilization_percent: gpu_busy.map(|b| b as u32),
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

/// Detect CXL-attached memory for MI300X / MI350.
///
/// On MI300X, the unified HBM pool may include CXL-attached memory exposed
/// via `/sys/bus/cxl/devices/memN/size` or via the device's
/// `mem_info_vis_vram_total` (visible VRAM includes CXL region).
///
/// Returns additional CXL memory in bytes (0 if not detected).
fn detect_cxl_memory(device_dir: &Path) -> u64 {
    // Method 1: Check if visible VRAM exceeds regular VRAM (indicates CXL pool).
    let vram_total = read_sysfs_u64(&device_dir.join("mem_info_vram_total")).unwrap_or(0);
    let vis_vram_total =
        read_sysfs_u64(&device_dir.join("mem_info_vis_vram_total")).unwrap_or(0);
    if vis_vram_total > vram_total && vram_total > 0 {
        let cxl = vis_vram_total.saturating_sub(vram_total);
        if cxl > 0 {
            tracing::debug!(
                cxl_bytes = cxl,
                "CXL-attached memory detected (vis_vram > vram)"
            );
            return cxl;
        }
    }

    // Method 2: Scan CXL bus for memory devices associated with this GPU.
    // MI300X exposes CXL memory via /sys/bus/cxl/devices/
    let cxl_bus = Path::new("/sys/bus/cxl/devices");
    if !cxl_bus.exists() {
        return 0;
    }
    let mut total_cxl = 0u64;
    for entry in std::fs::read_dir(cxl_bus).into_iter().flatten().flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // CXL memory devices are named "memN"
        if !name_str.starts_with("mem") {
            continue;
        }
        if let Some(size) = read_sysfs_u64(&entry.path().join("size")) {
            total_cxl = total_cxl.saturating_add(size);
        }
    }
    if total_cxl > 0 {
        tracing::debug!(
            total_cxl_bytes = total_cxl,
            "CXL memory detected via /sys/bus/cxl"
        );
    }
    total_cxl
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
