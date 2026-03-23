//! VRAM bandwidth probing for NVIDIA and AMD GPUs.
//!
//! Calculates theoretical memory bandwidth from memory clock speed and bus
//! width. Clock speed comes from `nvidia-smi` or AMD sysfs; bus width comes
//! from a lookup table keyed by compute capability (NVIDIA) or PCI device ID
//! (AMD).
//!
//! Formula: `bandwidth_gbps = memory_clock_mhz * bus_width_bits * 2 / 8 / 1000`
//!
//! The ×2 accounts for DDR (double data rate), which applies to both GDDR
//! and HBM memory types.

use std::path::Path;

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::command::{DEFAULT_TIMEOUT, run_tool};

const NVIDIA_BW_ARGS: &[&str] = &[
    "--query-gpu=clocks.max.memory,compute_cap",
    "--format=csv,noheader,nounits",
];

/// Enrich CUDA and ROCm profiles with memory bandwidth.
pub(crate) fn enrich_bandwidth(
    profiles: &mut [AcceleratorProfile],
    warnings: &mut Vec<DetectionError>,
) {
    // Skip the separate nvidia-smi call if all CUDA GPUs already have bandwidth
    // (set during the batched CUDA detection pass).
    let all_cuda_have_bw = profiles.iter().all(|p| {
        !matches!(p.accelerator, AcceleratorType::CudaGpu { .. })
            || p.memory_bandwidth_gbps.is_some()
    });
    let cuda_bw = if all_cuda_have_bw {
        Vec::new()
    } else {
        query_nvidia_bandwidth(warnings)
    };
    let count = apply_bandwidth(profiles, &cuda_bw);
    debug!(enriched = count, "memory bandwidth enrichment complete");
}

#[cfg(feature = "async-detect")]
pub(crate) async fn enrich_bandwidth_async(
    profiles: &mut [AcceleratorProfile],
    warnings: &mut Vec<DetectionError>,
) {
    let cuda_bw =
        match super::command::run_tool_async("nvidia-smi", NVIDIA_BW_ARGS, DEFAULT_TIMEOUT).await {
            Ok(o) => parse_nvidia_bandwidth_output(&o.stdout),
            Err(_) => Vec::new(),
        };
    let count = apply_bandwidth(profiles, &cuda_bw);
    debug!(enriched = count, "memory bandwidth enrichment complete");
    let _ = warnings; // no additional warnings from async path
}

/// Apply bandwidth values to profiles (shared by sync and async paths).
/// Returns the number of profiles that were enriched with bandwidth data.
fn apply_bandwidth(profiles: &mut [AcceleratorProfile], cuda_bw: &[Option<f64>]) -> usize {
    let mut nvidia_idx = 0usize;
    let mut count = 0usize;
    for profile in profiles.iter_mut() {
        match &profile.accelerator {
            AcceleratorType::CudaGpu { device_id } => {
                // Skip CUDA GPUs that already got bandwidth from the batched parse.
                if profile.memory_bandwidth_gbps.is_none() {
                    if let Some(bw) = cuda_bw.get(nvidia_idx).copied().flatten() {
                        profile.memory_bandwidth_gbps = Some(bw);
                        count += 1;
                    } else if let Some(cc) = &profile.compute_capability {
                        // Fallback: estimate from compute capability alone
                        if let Some(bw) = estimate_nvidia_bandwidth_from_cc(cc) {
                            profile.memory_bandwidth_gbps = Some(bw);
                            count += 1;
                        } else {
                            let device_id = *device_id;
                            debug!(device_id, "no memory bandwidth data available for CUDA GPU");
                        }
                    } else {
                        let device_id = *device_id;
                        debug!(device_id, "no memory bandwidth data available for CUDA GPU");
                    }
                } else {
                    count += 1;
                }
                nvidia_idx += 1;
            }
            AcceleratorType::RocmGpu { device_id } => {
                if let Some(bw) = probe_rocm_bandwidth(*device_id) {
                    profile.memory_bandwidth_gbps = Some(bw);
                    count += 1;
                }
            }
            _ => {}
        }
    }
    count
}

// ---------------------------------------------------------------------------
// NVIDIA bandwidth via nvidia-smi
// ---------------------------------------------------------------------------

/// Query `nvidia-smi` for max memory clock per GPU, calculate bandwidth.
fn query_nvidia_bandwidth(_warnings: &mut Vec<DetectionError>) -> Vec<Option<f64>> {
    let output = match run_tool("nvidia-smi", NVIDIA_BW_ARGS, DEFAULT_TIMEOUT) {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };
    parse_nvidia_bandwidth_output(&output.stdout)
}

/// Parse nvidia-smi bandwidth query output into per-GPU bandwidth values.
pub fn parse_nvidia_bandwidth_output(stdout: &str) -> Vec<Option<f64>> {
    stdout
        .lines()
        .map(|line| {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() < 2 {
                return None;
            }
            let max_mem_clock_mhz: f64 = parts[0].parse().ok()?;
            let cc = parts[1];
            let bus_width = nvidia_bus_width_bits(cc)?;
            let bw = calculate_bandwidth(max_mem_clock_mhz, bus_width);
            debug!(
                compute_cap = cc,
                mem_clock_mhz = max_mem_clock_mhz,
                bus_width,
                bandwidth_gbps = bw,
                "NVIDIA memory bandwidth calculated"
            );
            Some(bw)
        })
        .collect()
}

/// Calculate theoretical memory bandwidth in GB/s.
///
/// `clock_mhz * bus_width_bits * 2 (DDR) / 8 (bits→bytes) / 1000 (MB→GB)`
fn calculate_bandwidth(clock_mhz: f64, bus_width_bits: u32) -> f64 {
    let bw = clock_mhz * bus_width_bits as f64 * crate::units::DDR_MULTIPLIER
        / crate::units::BITS_PER_BYTE
        / crate::units::MHZ_PER_GHZ;
    (bw * 10.0).round() / 10.0 // round to 1 decimal
}

/// Look up NVIDIA GPU memory bus width from compute capability.
///
/// Compute capability → typical memory bus width in bits.
/// When a CC has multiple bus widths (e.g. consumer vs. data center),
/// we return the data center (wider) variant. The error from using the
/// wrong width is bounded — consumer cards usually have narrower buses
/// but also lower memory clocks, so the bandwidth estimate stays in the
/// right ballpark.
pub fn nvidia_bus_width_bits(cc: &str) -> Option<u32> {
    match cc {
        // Hopper (H100/H200)
        "9.0" => Some(5120), // HBM3, 5120-bit

        // Ada Lovelace (RTX 4090/4080/4070 etc.)
        "8.9" => Some(384), // RTX 4090 = 384-bit GDDR6X
        // Ampere data center (A100/A30)
        "8.0" => Some(5120), // HBM2e, 5120-bit
        // Ampere consumer (RTX 3090/3080/3070)
        "8.6" => Some(384), // RTX 3090 = 384-bit GDDR6X

        // Turing (RTX 2080 Ti / T4)
        "7.5" => Some(352), // RTX 2080 Ti = 352-bit GDDR6

        // Volta (V100)
        "7.0" => Some(4096), // HBM2, 4096-bit

        // Pascal data center (P100)
        "6.0" => Some(4096), // HBM2, 4096-bit
        // Pascal consumer (GTX 1080 Ti)
        "6.1" => Some(352), // 352-bit GDDR5X

        // Blackwell (B100/B200)
        "10.0" => Some(8192), // HBM3e, 8192-bit

        _ => None,
    }
}

/// Fallback: estimate bandwidth from compute capability using known
/// published specs (GB/s). Used when nvidia-smi doesn't report clock speed.
pub fn estimate_nvidia_bandwidth_from_cc(cc: &str) -> Option<f64> {
    match cc {
        "10.0" => Some(8000.0), // B200: ~8 TB/s
        "9.0" => Some(3350.0),  // H100 SXM: 3.35 TB/s
        "8.9" => Some(1008.0),  // RTX 4090: 1 TB/s
        "8.6" => Some(936.0),   // RTX 3090: 936 GB/s
        "8.0" => Some(2039.0),  // A100 SXM: 2 TB/s
        "7.5" => Some(616.0),   // RTX 2080 Ti: 616 GB/s
        "7.0" => Some(900.0),   // V100 SXM: 900 GB/s
        "6.1" => Some(484.0),   // GTX 1080 Ti: 484 GB/s
        "6.0" => Some(732.0),   // P100: 732 GB/s
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// AMD ROCm bandwidth via sysfs
// ---------------------------------------------------------------------------

/// Probe ROCm GPU memory bandwidth from sysfs memory clock info.
fn probe_rocm_bandwidth(device_id: u32) -> Option<f64> {
    // Find the card's sysfs device directory
    let drm = Path::new("/sys/class/drm");
    let mut card_idx = 0u32;
    for entry in std::fs::read_dir(drm).ok()?.flatten() {
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

        if card_idx == device_id {
            return read_rocm_bandwidth(&device_dir);
        }
        card_idx += 1;
    }
    None
}

/// Read memory bandwidth from a ROCm device's sysfs directory.
///
/// Reads max memory clock from `pp_dpm_mclk` and bus width from
/// `mem_info_vram_total` + PCI device ID heuristics.
fn read_rocm_bandwidth(device_dir: &Path) -> Option<f64> {
    // Read max memory clock from pp_dpm_mclk
    // Format: "0: 96Mhz\n1: 1000Mhz *\n" — last entry is typically max
    let dpm_mclk = super::read_sysfs_string(&device_dir.join("pp_dpm_mclk"), 4096)?;
    let max_mclk_mhz = parse_max_dpm_clock(&dpm_mclk)?;

    // Read memory bus width from mem_info_vram_total + device heuristic
    let bus_width = read_amd_bus_width(device_dir);

    let bw = calculate_bandwidth(max_mclk_mhz, bus_width);
    debug!(
        mem_clock_mhz = max_mclk_mhz,
        bus_width,
        bandwidth_gbps = bw,
        "AMD memory bandwidth calculated"
    );
    Some(bw)
}

/// Parse the highest clock entry from `pp_dpm_mclk`.
///
/// Format: lines like `0: 96Mhz`, `1: 1000Mhz *`
pub fn parse_max_dpm_clock(content: &str) -> Option<f64> {
    let mut max_clock = 0.0f64;
    for line in content.lines() {
        // e.g. "1: 1000Mhz *"
        if let Some(mhz_str) = line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.strip_suffix("Mhz").or_else(|| s.strip_suffix("MHz")))
            && let Ok(mhz) = mhz_str.parse::<f64>()
        {
            max_clock = max_clock.max(mhz);
        }
    }
    if max_clock > 0.0 {
        Some(max_clock)
    } else {
        None
    }
}

/// Determine AMD GPU memory bus width from sysfs device info.
///
/// Reads the PCI device ID and uses a lookup table. Falls back to
/// heuristics based on VRAM size.
fn read_amd_bus_width(device_dir: &Path) -> u32 {
    // Try PCI device ID lookup first
    if let Some(width) = read_amd_bus_width_from_device_id(device_dir) {
        return width;
    }

    // Fallback: estimate from VRAM size
    let vram = super::read_sysfs_u64(&device_dir.join("mem_info_vram_total")).unwrap_or(0);
    let vram_gb = vram / (1024 * 1024 * 1024);
    match vram_gb {
        0..=4 => 128,     // Low-end: 128-bit
        5..=8 => 256,     // Mid-range: 256-bit
        9..=16 => 256,    // High-end consumer: 256-bit
        17..=24 => 384,   // RTX 3090 class: 384-bit
        25..=48 => 4096,  // MI250: 4096-bit HBM2e
        49..=96 => 4096,  // MI250X: 4096-bit HBM2e
        97..=192 => 8192, // MI300X: 8192-bit HBM3
        _ => 256,
    }
}

/// Look up bus width from PCI device ID.
fn read_amd_bus_width_from_device_id(device_dir: &Path) -> Option<u32> {
    let device_id = super::read_sysfs_string(&device_dir.join("device"), 64)?;
    let device_id = device_id
        .trim()
        .strip_prefix("0x")
        .unwrap_or(device_id.trim());

    // AMD PCI device IDs → bus width
    match device_id {
        // MI300X
        "740c" | "740f" => Some(8192),
        // MI250X / MI250
        "740a" | "7408" => Some(4096),
        // MI210 / MI200 series
        "7400" | "7401" | "7402" | "7403" | "7404" | "7405" => Some(4096),
        // MI100
        "738c" | "738e" => Some(4096),
        // MI60 / MI50
        // MI60 / MI50 / Radeon VII
        "66a1" | "66a0" | "66af" => Some(4096),
        // RX 7900 XTX / 7900 XT
        "744c" | "7448" => Some(384),
        // RX 7800 XT / 7700 XT
        "7480" | "7470" => Some(256),
        // RX 6900 XT / 6800 XT
        "73bf" | "73a5" => Some(256),
        // RX 6700 XT
        "73df" => Some(192),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_bandwidth_h100() {
        // H100 SXM: 2619 MHz memory clock, 5120-bit bus
        // 2619 * 5120 * 2 / 8 / 1000 = 3352.3 GB/s
        let bw = calculate_bandwidth(2619.0, 5120);
        assert!((bw - 3352.3).abs() < 1.0, "H100 bandwidth: {}", bw);
    }

    #[test]
    fn calculate_bandwidth_a100() {
        // A100 SXM: 1593 MHz, 5120-bit bus
        // 1593 * 5120 * 2 / 8 / 1000 = 2039.0 GB/s
        let bw = calculate_bandwidth(1593.0, 5120);
        assert!((bw - 2039.0).abs() < 1.0, "A100 bandwidth: {}", bw);
    }

    #[test]
    fn calculate_bandwidth_rtx4090() {
        // RTX 4090: 10501 MHz effective (actual clock ~1313 MHz for GDDR6X ×8)
        // nvidia-smi reports 10501 MHz; 384-bit bus
        // 10501 * 384 * 2 / 8 / 1000 = 1008.1 GB/s
        let bw = calculate_bandwidth(10501.0, 384);
        assert!((bw - 1008.1).abs() < 1.0, "RTX 4090 bandwidth: {}", bw);
    }

    #[test]
    fn parse_max_dpm_clock_normal() {
        let input = "0: 96Mhz\n1: 1000Mhz *\n";
        assert_eq!(parse_max_dpm_clock(input), Some(1000.0));
    }

    #[test]
    fn parse_max_dpm_clock_hbm() {
        let input = "0: 500Mhz\n1: 900Mhz\n2: 1600Mhz *\n";
        assert_eq!(parse_max_dpm_clock(input), Some(1600.0));
    }

    #[test]
    fn parse_max_dpm_clock_empty() {
        assert_eq!(parse_max_dpm_clock(""), None);
    }
}
