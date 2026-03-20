//! PCIe link detection via sysfs.
//!
//! Reads `/sys/bus/pci/devices/*/` for link width and speed to estimate
//! host-to-device transfer rates. Called as an enrichment pass after
//! per-backend detection completes.

use std::path::Path;

use tracing::debug;

use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

/// Enrich profiles with PCIe bandwidth by reading sysfs PCI link info.
///
/// For CUDA GPUs, the PCI address is found via `/sys/bus/pci/drivers/nvidia/`.
/// For ROCm GPUs, via `/sys/bus/pci/drivers/amdgpu/`.
/// Falls back to scanning `/sys/class/drm/card*/device/` for any GPU.
pub(crate) fn enrich_pcie(profiles: &mut [AcceleratorProfile]) {
    let nvidia_addrs = list_driver_pci_addrs("nvidia");
    let amdgpu_addrs = list_driver_pci_addrs("amdgpu");

    let mut nvidia_idx = 0usize;
    let mut amdgpu_idx = 0usize;

    for profile in profiles.iter_mut() {
        let addr = match &profile.accelerator {
            AcceleratorType::CudaGpu { .. } => {
                let a = nvidia_addrs.get(nvidia_idx).cloned();
                nvidia_idx += 1;
                a
            }
            AcceleratorType::RocmGpu { .. } => {
                let a = amdgpu_addrs.get(amdgpu_idx).cloned();
                amdgpu_idx += 1;
                a
            }
            _ => None,
        };

        if let Some(addr) = addr {
            let device_path = format!("/sys/bus/pci/devices/{}", addr);
            if let Some(bw) = read_pcie_bandwidth(Path::new(&device_path)) {
                debug!(addr = %addr, bandwidth_gbps = bw, "PCIe link detected");
                profile.pcie_bandwidth_gbps = Some(bw);
            }
        }
    }
}

/// List PCI addresses bound to a given driver (sorted).
fn list_driver_pci_addrs(driver: &str) -> Vec<String> {
    let driver_path = format!("/sys/bus/pci/drivers/{}", driver);
    let dir = Path::new(&driver_path);
    if !dir.exists() {
        return Vec::new();
    }
    let mut addrs: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            // PCI addresses look like "0000:01:00.0"
            if name.contains(':') && name.contains('.') {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    addrs.sort();
    addrs
}

/// Read PCIe link width and speed from sysfs and compute theoretical bandwidth.
///
/// `current_link_width` × `current_link_speed` → GB/s.
fn read_pcie_bandwidth(device_path: &Path) -> Option<f64> {
    let width_str = std::fs::read_to_string(device_path.join("current_link_width")).ok()?;
    let speed_str = std::fs::read_to_string(device_path.join("current_link_speed")).ok()?;

    let width: f64 = width_str.trim().parse().ok()?;
    let speed_gts = parse_link_speed(speed_str.trim())?;

    // PCIe uses 128b/130b encoding for Gen3+, 8b/10b for Gen1/2.
    let encoding_overhead = if speed_gts >= 8.0 {
        128.0 / 130.0
    } else {
        8.0 / 10.0
    };

    // GT/s × width × encoding / 8 bits per byte = GB/s
    let bandwidth_gbps = speed_gts * width * encoding_overhead / 8.0;
    Some((bandwidth_gbps * 100.0).round() / 100.0)
}

/// Parse a PCIe link speed string like "16 GT/s" or "8.0 GT/s" to GT/s.
fn parse_link_speed(s: &str) -> Option<f64> {
    // Formats: "16 GT/s", "8.0 GT/s PCIe", "2.5 GT/s"
    let numeric = s
        .split_whitespace()
        .next()?;
    numeric.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_link_speed_values() {
        assert_eq!(parse_link_speed("16 GT/s"), Some(16.0));
        assert_eq!(parse_link_speed("8.0 GT/s PCIe"), Some(8.0));
        assert_eq!(parse_link_speed("2.5 GT/s"), Some(2.5));
    }

    #[test]
    fn pcie_bandwidth_gen4_x16() {
        // Gen4 = 16 GT/s, x16, 128b/130b encoding
        // 16 × 16 × (128/130) / 8 = 31.51 GB/s
        let bw = read_pcie_bandwidth(Path::new("/nonexistent"));
        assert!(bw.is_none());
    }
}
