//! Vulkan compute device detection via `vulkaninfo`.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_vulkan(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    // Skip if we already found a CUDA or ROCm GPU (avoid double-counting).
    let has_dedicated_gpu = profiles.iter().any(|p| {
        matches!(
            p.accelerator,
            AcceleratorType::CudaGpu { .. } | AcceleratorType::RocmGpu { .. }
        )
    });
    if has_dedicated_gpu {
        return;
    }

    let output = match std::process::Command::new("vulkaninfo")
        .arg("--summary")
        .output()
    {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            warnings.push(DetectionError::ToolFailed {
                tool: "vulkaninfo".into(),
                exit_code: o.status.code(),
                stderr: String::from_utf8_lossy(&o.stderr).to_string(),
            });
            return;
        }
        Err(_) => return,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let devices = parse_vulkan_summary(&stdout);

    if devices.is_empty() {
        // Fallback: register a generic device if vulkaninfo ran but we couldn't parse
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
///
/// The summary format has blocks like:
/// ```text
/// GPU0:
///     apiVersion    = 1.3.277
///     driverVersion = 545.29.6
///     deviceName    = NVIDIA GeForce RTX 4090
///     ...
///     deviceMemoryHeap[0]:
///         size = 24564 MB (...)
/// ```
fn parse_vulkan_summary(output: &str) -> Vec<VulkanDevice> {
    let mut devices = Vec::new();
    let mut current_name = String::new();
    let mut current_mem: u64 = 0;
    let mut current_api = None;
    let mut current_driver = None;
    let mut in_device = false;

    for line in output.lines() {
        let trimmed = line.trim();

        // New GPU block
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

        // Parse memory heap size: "size = 24564 MB (...)"
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

    // Flush last device
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
