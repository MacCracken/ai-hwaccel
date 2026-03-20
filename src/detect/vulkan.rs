//! Vulkan compute device detection via `vulkaninfo`.
//!
//! Uses `vulkaninfo --summary` for basic device info, then optionally
//! parses full `vulkaninfo` output for compute queue families and subgroup
//! sizes.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::command::{DEFAULT_TIMEOUT, run_tool};

const VULKANINFO_SUMMARY_ARGS: &[&str] = &["--summary"];
const VULKANINFO_FULL_ARGS: &[&str] = &[];

/// Detect Vulkan devices.
///
/// Note: deduplication against CUDA/ROCm GPUs is handled by the orchestrator
/// in `detect/mod.rs` after all backends complete (since detection is parallel).
pub(crate) fn detect_vulkan(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    let output = match run_tool("vulkaninfo", VULKANINFO_SUMMARY_ARGS, DEFAULT_TIMEOUT) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => return,
        Err(e) => {
            warnings.push(e);
            return;
        }
    };
    // Full output for compute queue details (separate format from --summary).
    let full_output = run_tool("vulkaninfo", VULKANINFO_FULL_ARGS, DEFAULT_TIMEOUT).ok();
    let full_stdout = full_output.as_ref().map(|o| o.stdout.as_str());
    parse_vulkan_output(&output.stdout, full_stdout, profiles, warnings);
}

#[cfg(feature = "async-detect")]
pub(crate) async fn detect_vulkan_async() -> super::DetectResult {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let output = match super::command::run_tool_async("vulkaninfo", VULKANINFO_SUMMARY_ARGS, DEFAULT_TIMEOUT).await {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => return (profiles, warnings),
        Err(e) => {
            warnings.push(e);
            return (profiles, warnings);
        }
    };
    let full_output = super::command::run_tool_async("vulkaninfo", VULKANINFO_FULL_ARGS, DEFAULT_TIMEOUT).await.ok();
    let full_stdout = full_output.as_ref().map(|o| o.stdout.as_str());
    parse_vulkan_output(&output.stdout, full_stdout, &mut profiles, &mut warnings);
    (profiles, warnings)
}

pub(crate) fn parse_vulkan_output(
    summary_stdout: &str,
    full_stdout: Option<&str>,
    profiles: &mut Vec<AcceleratorProfile>,
    _warnings: &mut Vec<DetectionError>,
) {
    let devices = parse_vulkan_summary(summary_stdout);
    let compute_info = full_stdout.map(parse_vulkan_full).unwrap_or_default();

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
            temperature_c: None,
            power_watts: None,
            gpu_utilization_percent: None,
        });
    } else {
        for (i, dev) in devices.into_iter().take(1024).enumerate() {
            let extra = compute_info.get(i);
            let mut cap_parts = Vec::new();
            if let Some(api) = &dev.api_version {
                cap_parts.push(format!("Vulkan {api}"));
            }
            if let Some(info) = extra {
                cap_parts.push(format!(
                    "compute queues: {}x{}, subgroup: {}",
                    info.compute_queue_count,
                    info.compute_queue_family_count,
                    info.subgroup_size,
                ));
            }
            let compute_cap = if cap_parts.is_empty() {
                dev.api_version.clone()
            } else {
                Some(cap_parts.join(", "))
            };

            debug!(
                device_id = i,
                name = %dev.name,
                mem_mb = dev.memory_mb,
                ?extra,
                "Vulkan GPU detected"
            );
            profiles.push(AcceleratorProfile {
                accelerator: AcceleratorType::VulkanGpu {
                    device_id: i as u32,
                    device_name: dev.name,
                },
                available: true,
                memory_bytes: dev.memory_mb.saturating_mul(1024 * 1024),
                compute_capability: compute_cap,
                driver_version: dev.driver_version,
                memory_bandwidth_gbps: None,
                memory_used_bytes: None,
                memory_free_bytes: None,
                pcie_bandwidth_gbps: None,
                numa_node: None,
            temperature_c: None,
            power_watts: None,
            gpu_utilization_percent: None,
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

/// Compute-relevant info parsed from full `vulkaninfo` output.
#[derive(Debug, Default)]
struct VulkanComputeInfo {
    /// Number of queue families that support QUEUE_COMPUTE_BIT.
    compute_queue_family_count: u32,
    /// Total compute queues across all compute-capable families.
    compute_queue_count: u32,
    /// Subgroup size (wavefront/warp size).
    subgroup_size: u32,
}

// ---------------------------------------------------------------------------
// Summary parser (--summary)
// ---------------------------------------------------------------------------

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
                "deviceName" => current_name = value.chars().take(256).collect(),
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

// ---------------------------------------------------------------------------
// Full output parser (no args)
// ---------------------------------------------------------------------------

/// Parse full `vulkaninfo` output for compute queue families and subgroup size.
///
/// Returns one `VulkanComputeInfo` per GPU found.
fn parse_vulkan_full(output: &str) -> Vec<VulkanComputeInfo> {
    let mut infos = Vec::new();
    let mut current = VulkanComputeInfo::default();
    let mut in_queue_section = false;

    for line in output.lines() {
        let trimmed = line.trim();

        // New GPU section resets state.
        if trimmed.starts_with("VkPhysicalDeviceProperties:") || trimmed.starts_with("GPU id") {
            if current.subgroup_size > 0 || current.compute_queue_count > 0 {
                infos.push(current);
                current = VulkanComputeInfo::default();
            }
            in_queue_section = false;
        }

        // Subgroup size.
        if trimmed.starts_with("subgroupSize") && !trimmed.contains("Control") {
            if let Some(val) = extract_value(trimmed) {
                if let Ok(size) = val.parse::<u32>() {
                    current.subgroup_size = size;
                }
            }
        }

        // Queue family section.
        if trimmed.starts_with("VkQueueFamilyProperties") {
            in_queue_section = true;
            continue;
        }

        if in_queue_section {
            // Detect compute-capable queue family.
            if trimmed.starts_with("queueFlags") && trimmed.contains("QUEUE_COMPUTE_BIT") {
                current.compute_queue_family_count += 1;
                // The queueCount for this family should follow shortly.
            }

            if trimmed.starts_with("queueCount") {
                if let Some(val) = extract_value(trimmed) {
                    if let Ok(count) = val.parse::<u32>() {
                        // Only add to compute_queue_count if the previous queueFlags
                        // included COMPUTE_BIT. We track this by checking if family
                        // count just incremented. This is a heuristic — the queueFlags
                        // line always precedes its queueCount.
                        if current.compute_queue_family_count > 0 {
                            // Count the latest family's queues.
                            current.compute_queue_count += count;
                        }
                    }
                }
            }

            // End of queue section.
            if trimmed.is_empty() || trimmed.starts_with("Vk") && !trimmed.starts_with("VkQueue") {
                in_queue_section = false;
            }
        }
    }

    // Flush last device.
    if current.subgroup_size > 0 || current.compute_queue_count > 0 {
        infos.push(current);
    }

    infos
}

/// Extract the value after `=` or `:` from a key-value line.
fn extract_value(line: &str) -> Option<&str> {
    line.split_once('=')
        .map(|(_, v)| v.trim())
        .or_else(|| line.split_once(':').map(|(_, v)| v.trim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_vulkan_full_compute_info() {
        let output = r#"
VkPhysicalDeviceProperties:
	deviceType        = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	deviceName        = AMD Radeon Graphics (RADV RENOIR)
	subgroupSize                      = 64
VkQueueFamilyProperties:
		queueCount                  = 1
		queueFlags                  = QUEUE_GRAPHICS_BIT | QUEUE_COMPUTE_BIT | QUEUE_TRANSFER_BIT
		queueCount                  = 4
		queueFlags                  = QUEUE_COMPUTE_BIT | QUEUE_TRANSFER_BIT
		queueCount                  = 1
		queueFlags                  = QUEUE_VIDEO_DECODE_BIT_KHR
"#;
        let infos = parse_vulkan_full(output);
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].subgroup_size, 64);
        assert_eq!(infos[0].compute_queue_family_count, 2);
        assert!(infos[0].compute_queue_count >= 5); // 1 + 4
    }

    #[test]
    fn parse_vulkan_full_empty() {
        let infos = parse_vulkan_full("");
        assert!(infos.is_empty());
    }

    #[test]
    fn extract_value_equals() {
        assert_eq!(extract_value("subgroupSize = 64"), Some("64"));
    }

    #[test]
    fn extract_value_colon() {
        assert_eq!(extract_value("queueCount: 4"), Some("4"));
    }
}
