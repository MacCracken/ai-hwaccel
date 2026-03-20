//! NUMA topology detection via sysfs.
//!
//! Maps GPUs to their NUMA nodes by reading
//! `/sys/bus/pci/devices/<addr>/numa_node`.

use tracing::debug;

use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

/// Enrich profiles with NUMA node information.
///
/// Uses the same PCI address mapping as the PCIe module to find each device's
/// NUMA affinity.
pub(crate) fn enrich_numa(
    profiles: &mut [AcceleratorProfile],
    nvidia_addrs: &[String],
    amdgpu_addrs: &[String],
) {
    let mut count = 0usize;
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
            if let Ok(canonical) = std::fs::canonicalize(&device_path) {
                if !canonical.starts_with("/sys/") {
                    continue;
                }
                let numa_path = canonical.join("numa_node");
                if let Some(contents) = super::read_sysfs_string(&numa_path, 64)
                    && let Ok(node) = contents.trim().parse::<i32>()
                    && node >= 0
                {
                    debug!(addr = %addr, numa_node = node, "NUMA node detected");
                    profile.numa_node = Some(node as u32);
                    count += 1;
                }
            }
        }
    }
    debug!(enriched = count, "NUMA topology enrichment complete");
}
