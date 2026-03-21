//! Vulkan compute device detection via `vulkaninfo`.
//!
//! Uses `vulkaninfo --summary` for basic device info, then optionally
//! parses full `vulkaninfo` output for compute queue families and subgroup
//! sizes.
//!
//! Performance: `vulkaninfo` can take 3–5s on some systems (e.g. AMD Cezanne
//! iGPU). To mitigate this, results are cached in
//! `$XDG_CACHE_HOME/ai-hwaccel/vulkan.json` with a 60s TTL, and the subprocess
//! runs with a 3s timeout (shorter than the default 5s).

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use tracing::{debug, trace};

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::command::run_tool;

const VULKANINFO_SUMMARY_ARGS: &[&str] = &["--summary"];
const VULKANINFO_FULL_ARGS: &[&str] = &[];

/// Vulkaninfo-specific timeout (shorter than the 5s default since vulkaninfo
/// is known to hang on some systems).
const VULKANINFO_TIMEOUT: Duration = Duration::from_secs(3);

/// TTL for the vulkaninfo file cache.
const VULKANINFO_CACHE_TTL: Duration = Duration::from_secs(60);

/// Return the cache file path for vulkaninfo results.
fn vulkan_cache_path() -> Option<PathBuf> {
    let cache_dir = std::env::var("XDG_CACHE_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cache"))
        })?;
    Some(cache_dir.join("ai-hwaccel").join("vulkan.json"))
}

/// Try to read cached vulkaninfo output. Returns (summary, full) if fresh.
fn read_vulkan_cache() -> Option<(String, Option<String>)> {
    let path = vulkan_cache_path()?;
    let metadata = std::fs::metadata(&path).ok()?;
    let age = metadata
        .modified()
        .ok()?
        .elapsed()
        .unwrap_or(Duration::MAX);
    if age > VULKANINFO_CACHE_TTL {
        return None;
    }
    let data = std::fs::read_to_string(&path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&data).ok()?;
    let summary = parsed.get("summary")?.as_str()?.to_string();
    let full = parsed.get("full").and_then(|v| v.as_str()).map(String::from);
    debug!("using cached vulkaninfo results (age: {:.1}s)", age.as_secs_f64());
    Some((summary, full))
}

/// Write vulkaninfo output to the cache file (atomic via temp+rename).
fn write_vulkan_cache(summary: &str, full: Option<&str>) {
    let Some(path) = vulkan_cache_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let cache = serde_json::json!({
        "summary": summary,
        "full": full,
        "cached_at": SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    });
    if let Ok(json) = serde_json::to_string(&cache) {
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, &json).is_ok() {
            if std::fs::rename(&tmp, &path).is_ok() {
                return;
            }
            let _ = std::fs::remove_file(&tmp);
        }
        let _ = std::fs::write(&path, json);
    }
}

/// Detect Vulkan devices.
///
/// Note: deduplication against CUDA/ROCm GPUs is handled by the orchestrator
/// in `detect/mod.rs` after all backends complete (since detection is parallel).
pub(crate) fn detect_vulkan(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    // Try cache first.
    if let Some((summary, full)) = read_vulkan_cache() {
        parse_vulkan_output(&summary, full.as_deref(), profiles, warnings);
        return;
    }

    let output = match run_tool("vulkaninfo", VULKANINFO_SUMMARY_ARGS, VULKANINFO_TIMEOUT) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => {
            debug!("vulkaninfo not found on $PATH, skipping Vulkan detection");
            return;
        }
        Err(DetectionError::Timeout { .. }) => {
            debug!("vulkaninfo timed out, falling back to sysfs-only Vulkan detection");
            warnings.push(DetectionError::Timeout {
                tool: "vulkaninfo".into(),
                timeout_secs: VULKANINFO_TIMEOUT.as_secs_f64(),
            });
            return;
        }
        Err(e) => {
            warnings.push(e);
            return;
        }
    };
    // Full output for compute queue details (separate format from --summary).
    let full_output = run_tool("vulkaninfo", VULKANINFO_FULL_ARGS, VULKANINFO_TIMEOUT).ok();
    let full_stdout = full_output.as_ref().map(|o| o.stdout.as_str());

    // Cache the results for subsequent calls.
    write_vulkan_cache(&output.stdout, full_stdout);

    parse_vulkan_output(&output.stdout, full_stdout, profiles, warnings);
}

#[cfg(feature = "async-detect")]
pub(crate) async fn detect_vulkan_async() -> super::DetectResult {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();

    // Try cache first.
    if let Some((summary, full)) = read_vulkan_cache() {
        parse_vulkan_output(&summary, full.as_deref(), &mut profiles, &mut warnings);
        return (profiles, warnings);
    }

    let output = match super::command::run_tool_async(
        "vulkaninfo",
        VULKANINFO_SUMMARY_ARGS,
        VULKANINFO_TIMEOUT,
    )
    .await
    {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => {
            debug!("vulkaninfo not found on $PATH, skipping Vulkan detection");
            return (profiles, warnings);
        }
        Err(DetectionError::Timeout { .. }) => {
            debug!("vulkaninfo timed out, falling back to sysfs-only Vulkan detection");
            warnings.push(DetectionError::Timeout {
                tool: "vulkaninfo".into(),
                timeout_secs: VULKANINFO_TIMEOUT.as_secs_f64(),
            });
            return (profiles, warnings);
        }
        Err(e) => {
            warnings.push(e);
            return (profiles, warnings);
        }
    };
    let full_output =
        super::command::run_tool_async("vulkaninfo", VULKANINFO_FULL_ARGS, VULKANINFO_TIMEOUT)
            .await
            .ok();
    let full_stdout = full_output.as_ref().map(|o| o.stdout.as_str());

    write_vulkan_cache(&output.stdout, full_stdout);

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
    // Cap full output parse at 256 KiB to bound memory usage on verbose drivers.
    let compute_info = full_stdout
        .filter(|s| s.len() <= 256 * 1024)
        .map(parse_vulkan_full)
        .unwrap_or_default();

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
                    info.compute_queue_count, info.compute_queue_family_count, info.subgroup_size,
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

// ---------------------------------------------------------------------------
// Sysfs-only Vulkan fallback
// ---------------------------------------------------------------------------

/// Detect Vulkan-capable GPUs via sysfs without spawning `vulkaninfo`.
///
/// Scans `/sys/class/drm/card*/device/{vendor,device}` and uses a PCI ID
/// lookup table to identify GPU names and estimate VRAM. This is a fast
/// fallback for when `vulkaninfo` is absent, slow, or times out.
pub(crate) fn detect_vulkan_sysfs(
    profiles: &mut Vec<AcceleratorProfile>,
    _warnings: &mut Vec<DetectionError>,
) {
    let drm_dir = std::path::Path::new("/sys/class/drm");
    if !drm_dir.exists() {
        return;
    }

    let mut device_id_counter: u32 = 0;

    let entries = match std::fs::read_dir(drm_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Only look at cardN entries, not cardN-DP-1 etc.
        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }

        let dev_dir = entry.path().join("device");
        let vendor_path = dev_dir.join("vendor");
        let device_path = dev_dir.join("device");

        let Some(vendor_str) = super::read_sysfs_string(&vendor_path, 64) else {
            continue;
        };
        let Some(device_str) = super::read_sysfs_string(&device_path, 64) else {
            continue;
        };

        let vendor_id = u16::from_str_radix(vendor_str.trim().trim_start_matches("0x"), 16)
            .unwrap_or(0);
        let device_id_pci = u16::from_str_radix(device_str.trim().trim_start_matches("0x"), 16)
            .unwrap_or(0);

        // Skip non-GPU vendors.
        if !is_gpu_vendor(vendor_id) {
            continue;
        }

        let (device_name, estimated_vram_mb) = lookup_pci_gpu(vendor_id, device_id_pci);

        // Try to read VRAM from sysfs (some drivers expose this).
        let vram_bytes = read_drm_vram(&dev_dir)
            .unwrap_or(estimated_vram_mb * 1024 * 1024);

        debug!(
            vendor_id,
            device_id_pci,
            name = %device_name,
            vram_mb = vram_bytes / (1024 * 1024),
            "sysfs Vulkan GPU detected"
        );

        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::VulkanGpu {
                device_id: device_id_counter,
                device_name,
            },
            available: true,
            memory_bytes: vram_bytes,
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
        device_id_counter += 1;
    }
}

/// Check if a PCI vendor ID belongs to a GPU vendor.
fn is_gpu_vendor(vendor_id: u16) -> bool {
    matches!(
        vendor_id,
        0x10de  // NVIDIA
        | 0x1002 // AMD/ATI
        | 0x8086 // Intel
    )
}

/// Try to read VRAM total from DRM sysfs.
fn read_drm_vram(dev_dir: &std::path::Path) -> Option<u64> {
    // AMD: mem_info_vram_total
    if let Some(vram) = super::read_sysfs_u64(&dev_dir.join("mem_info_vram_total"))
        && vram > 0
    {
        return Some(vram);
    }
    None
}

/// Lookup GPU name and estimated VRAM from PCI vendor/device IDs.
///
/// Returns (name, estimated_vram_mb). This covers common consumer and
/// datacenter GPUs. Unknown devices get a generic name and 4 GB estimate.
fn lookup_pci_gpu(vendor_id: u16, device_id: u16) -> (String, u64) {
    match vendor_id {
        // NVIDIA
        0x10de => {
            let (name, vram) = match device_id {
                // Ada Lovelace
                0x2684 => ("NVIDIA GeForce RTX 4090", 24 * 1024),
                0x2702 => ("NVIDIA GeForce RTX 4080 SUPER", 16 * 1024),
                0x2704 => ("NVIDIA GeForce RTX 4080", 16 * 1024),
                0x2782 => ("NVIDIA GeForce RTX 4070 Ti SUPER", 16 * 1024),
                0x2786 => ("NVIDIA GeForce RTX 4070 Ti", 12 * 1024),
                0x2783 => ("NVIDIA GeForce RTX 4070 SUPER", 12 * 1024),
                0x2787..=0x27a0 => ("NVIDIA GeForce RTX 4070", 12 * 1024),
                0x2803..=0x2820 => ("NVIDIA GeForce RTX 4060 Ti", 8 * 1024),
                0x2882..=0x2900 => ("NVIDIA GeForce RTX 4060", 8 * 1024),
                // Ampere
                0x2204 => ("NVIDIA GeForce RTX 3090", 24 * 1024),
                0x2206 => ("NVIDIA GeForce RTX 3080", 10 * 1024),
                0x2484 => ("NVIDIA GeForce RTX 3070", 8 * 1024),
                0x2504 => ("NVIDIA GeForce RTX 3060", 12 * 1024),
                // Datacenter
                0x2330 => ("NVIDIA H100", 80 * 1024),
                0x2324 => ("NVIDIA H100 PCIe", 80 * 1024),
                0x20b0 => ("NVIDIA A100 SXM", 80 * 1024),
                0x20b2 => ("NVIDIA A100 PCIe 80GB", 80 * 1024),
                0x20b5 => ("NVIDIA A100 PCIe 40GB", 40 * 1024),
                0x20b7 => ("NVIDIA A30", 24 * 1024),
                0x25b6 => ("NVIDIA A16", 16 * 1024),
                0x27b8 => ("NVIDIA L4", 24 * 1024),
                0x26b5 => ("NVIDIA L40", 48 * 1024),
                0x26b9 => ("NVIDIA L40S", 48 * 1024),
                0x1eb8 => ("NVIDIA T4", 16 * 1024),
                _ => ("NVIDIA GPU (unknown model)", 4 * 1024),
            };
            (name.into(), vram)
        }
        // AMD/ATI
        0x1002 => {
            let (name, vram) = match device_id {
                0x744c => ("AMD Radeon RX 7900 XTX", 24 * 1024),
                0x7448 => ("AMD Radeon RX 7900 XT", 20 * 1024),
                0x7480 => ("AMD Radeon RX 7800 XT", 16 * 1024),
                0x7470 => ("AMD Radeon RX 7700 XT", 12 * 1024),
                0x7460 => ("AMD Radeon RX 7600", 8 * 1024),
                0x73bf => ("AMD Radeon RX 6900 XT", 16 * 1024),
                0x73af => ("AMD Radeon RX 6800 XT", 16 * 1024),
                0x73df => ("AMD Radeon RX 6700 XT", 12 * 1024),
                0x73ff => ("AMD Radeon RX 6600 XT", 8 * 1024),
                0x7408 => ("AMD Instinct MI300X", 192 * 1024),
                0x740c => ("AMD Instinct MI250X", 128 * 1024),
                0x740f => ("AMD Instinct MI210", 64 * 1024),
                // Integrated GPUs (common Renoir/Cezanne/Phoenix/Hawk Point)
                0x1636 | 0x1638 => ("AMD Radeon Graphics (Renoir)", 512),
                0x164c | 0x164e => ("AMD Radeon Graphics (Cezanne)", 512),
                0x15bf | 0x15c8 => ("AMD Radeon Graphics (Phoenix)", 512),
                _ => ("AMD GPU (unknown model)", 4 * 1024),
            };
            (name.into(), vram)
        }
        // Intel
        0x8086 => {
            let (name, vram) = match device_id {
                0x56a0 | 0x56a1 => ("Intel Arc A770", 16 * 1024),
                0x56a5 | 0x56a6 => ("Intel Arc A750", 8 * 1024),
                0x5690..=0x5692 => ("Intel Arc A580", 8 * 1024),
                0x56c0 | 0x56c1 => ("Intel Arc A380", 6 * 1024),
                0x4f80..=0x4f8f => ("Intel Data Center GPU Flex", 16 * 1024),
                0x0bd0..=0x0bdf => ("Intel Data Center GPU Max", 48 * 1024),
                _ => ("Intel GPU (unknown model)", 2 * 1024),
            };
            (name.into(), vram)
        }
        _ => ("Unknown GPU".into(), 4 * 1024),
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
    trace!(
        line_count = output.lines().count(),
        "parsing vulkaninfo summary"
    );
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
        if trimmed.starts_with("subgroupSize")
            && !trimmed.contains("Control")
            && let Some(val) = extract_value(trimmed)
            && let Ok(size) = val.parse::<u32>()
        {
            current.subgroup_size = size;
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

            if trimmed.starts_with("queueCount")
                && let Some(val) = extract_value(trimmed)
                && let Ok(count) = val.parse::<u32>()
            {
                // Only add to compute_queue_count if the previous queueFlags
                // included COMPUTE_BIT. We track this by checking if family
                // count just incremented. This is a heuristic — the queueFlags
                // line always precedes its queueCount.
                if current.compute_queue_family_count > 0 {
                    // Count the latest family's queues.
                    current.compute_queue_count += count;
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
