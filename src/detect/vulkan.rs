//! Vulkan compute device detection via `vulkaninfo`.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::command::{DEFAULT_TIMEOUT, run_tool};

/// Detect Vulkan devices.
///
/// Note: deduplication against CUDA/ROCm GPUs is handled by the orchestrator
/// in `detect/mod.rs` after all backends complete (since detection is parallel).
pub(crate) fn detect_vulkan(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    let output = match run_tool("vulkaninfo", &["--summary"], DEFAULT_TIMEOUT) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => return,
        Err(e) => {
            warnings.push(e);
            return;
        }
    };

    let devices = parse_vulkan_summary(&output.stdout);

    if devices.is_empty() {
        debug!("vulkaninfo found but no devices parsed, registering generic Vulkan GPU");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::VulkanGpu {
                device_id: 0,
                device_name: "Unknown Vulkan Device".into(),
            },
            available: true,
            memory_bytes: 4 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
            memory_bandwidth_gbps: None,
            memory_used_bytes: None,
            memory_free_bytes: None,
            pcie_bandwidth_gbps: None,
            numa_node: None,
        });
    } else {
        for (i, dev) in devices.into_iter().enumerate() {
            debug!(device_id = i, name = %dev.name, mem_mb = dev.memory_mb, "Vulkan GPU detected");
            profiles.push(AcceleratorProfile {
                accelerator: AcceleratorType::VulkanGpu {
                    device_id: i as u32,
                    device_name: dev.name,
                },
                available: true,
                memory_bytes: dev.memory_mb * 1024 * 1024,
                compute_capability: dev.api_version,
                driver_version: dev.driver_version,
                memory_bandwidth_gbps: None,
                memory_used_bytes: None,
                memory_free_bytes: None,
                pcie_bandwidth_gbps: None,
                numa_node: None,
            });
        }
    }
}

struct VulkanDevice {
    name: String,
    memory_mb: u64,
    api_version: Option<String>,
    driver_version: Option<String>,
}

/// Parse `vulkaninfo --summary` output for device details.
fn parse_vulkan_summary(output: &str) -> Vec<VulkanDevice> {
    let mut devices = Vec::new();
    let mut current_name = String::new();
    let mut current_mem: u64 = 0;
    let mut current_api = None;
    let mut current_driver = None;
    let mut in_device = false;

    for line in output.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("GPU") && trimmed.ends_with(':') {
            if in_device && !current_name.is_empty() {
                devices.push(VulkanDevice {
                    name: std::mem::take(&mut current_name),
                    memory_mb: if current_mem > 0 {
                        current_mem
                    } else {
                        4 * 1024
                    },
                    api_version: current_api.take(),
                    driver_version: current_driver.take(),
                });
                current_mem = 0;
            }
            in_device = true;
            continue;
        }

        if !in_device {
            continue;
        }

        if let Some((key, value)) = trimmed.split_once('=') {
            let key = key.trim();
            let value = value.trim();

            match key {
                "deviceName" => current_name = value.to_string(),
                "apiVersion" => current_api = Some(value.to_string()),
                "driverVersion" => current_driver = Some(value.to_string()),
                _ => {}
            }
        }

        if trimmed.starts_with("size")
            && let Some((_, rest)) = trimmed.split_once('=')
        {
            let rest = rest.trim();
            if let Some(mb_str) = rest.split_whitespace().next()
                && let Ok(mb) = mb_str.parse::<u64>()
                && mb > current_mem
            {
                current_mem = mb;
            }
        }
    }

    if in_device && !current_name.is_empty() {
        devices.push(VulkanDevice {
            name: current_name,
            memory_mb: if current_mem > 0 {
                current_mem
            } else {
                4 * 1024
            },
            api_version: current_api,
            driver_version: current_driver,
        });
    }

    devices
}
