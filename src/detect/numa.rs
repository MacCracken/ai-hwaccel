//! NUMA topology detection via sysfs.
//!
//! Maps GPUs to their NUMA nodes by reading
//! `/sys/bus/pci/devices/<addr>/numa_node`.

use std::path::Path;

use tracing::debug;

use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

/// Enrich profiles with NUMA node information.
///
/// Uses the same PCI address mapping as the PCIe module to find each device's
/// NUMA affinity.
pub(crate) fn enrich_numa(profiles: &mut [AcceleratorProfile]) {
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
            let numa_path = format!("/sys/bus/pci/devices/{}/numa_node", addr);
            if let Ok(contents) = std::fs::read_to_string(&numa_path)
                && let Ok(node) = contents.trim().parse::<i32>()
                && node >= 0
            {
                debug!(addr = %addr, numa_node = node, "NUMA node detected");
                profile.numa_node = Some(node as u32);
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
