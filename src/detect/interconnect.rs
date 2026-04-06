//! Network interconnect detection: InfiniBand, RoCE, NVLink, NVSwitch,
//! XGMI/Infinity Fabric, and Google ICI.
//!
//! Probes sysfs and CLI tools to discover high-speed interconnects
//! used for multi-node training and inter-GPU communication.

use tracing::debug;

use crate::error::DetectionError;
use crate::system_io::{Interconnect, InterconnectKind};

use super::command::{DEFAULT_TIMEOUT, run_tool};

const NVLINK_ARGS: &[&str] = &["nvlink", "-s"];
const TOPO_ARGS: &[&str] = &["topo", "-m"];
const ROCM_TOPO_ARGS: &[&str] = &["--showtopo"];

/// Detect all network interconnects on the system.
pub(crate) fn detect_interconnects(warnings: &mut Vec<DetectionError>) -> Vec<Interconnect> {
    let mut interconnects = Vec::new();

    detect_infiniband(&mut interconnects, warnings);
    detect_nvlink(&mut interconnects, warnings);
    detect_nvswitch(&mut interconnects, warnings);
    detect_xgmi(&mut interconnects, warnings);
    detect_tpu_ici(&mut interconnects);

    interconnects
}

#[cfg(feature = "async-detect")]
pub(crate) async fn detect_interconnects_async() -> (Vec<Interconnect>, Vec<DetectionError>) {
    let mut interconnects = Vec::new();
    let mut warnings = Vec::new();

    // InfiniBand/RoCE: sync sysfs probing, runs inline.
    detect_infiniband(&mut interconnects, &mut warnings);

    // NVLink: async CLI call.
    if let Ok(output) =
        super::command::run_tool_async("nvidia-smi", NVLINK_ARGS, DEFAULT_TIMEOUT).await
    {
        parse_nvlink_output(&output.stdout, &mut interconnects);
    }

    // NVSwitch: async CLI call (nvidia-smi topo -m).
    if let Ok(output) =
        super::command::run_tool_async("nvidia-smi", TOPO_ARGS, DEFAULT_TIMEOUT).await
    {
        parse_nvswitch_topo(&output.stdout, &mut interconnects);
    } else {
        // Fallback to sysfs probing.
        detect_nvswitch_sysfs(&mut interconnects);
    }

    // XGMI: async CLI call (rocm-smi --showtopo).
    if let Ok(output) =
        super::command::run_tool_async("rocm-smi", ROCM_TOPO_ARGS, DEFAULT_TIMEOUT).await
    {
        parse_xgmi_topo(&output.stdout, &mut interconnects);
    } else {
        // Fallback to sysfs probing.
        detect_xgmi_sysfs(&mut interconnects);
    }

    // TPU ICI: sync sysfs probing.
    detect_tpu_ici(&mut interconnects);

    (interconnects, warnings)
}

/// Detect InfiniBand / RoCE devices via sysfs (`/sys/class/infiniband/`).
fn detect_infiniband(interconnects: &mut Vec<Interconnect>, _warnings: &mut Vec<DetectionError>) {
    let ib_dir = std::path::Path::new("/sys/class/infiniband");
    let Ok(entries) = std::fs::read_dir(ib_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let dev_name = entry.file_name().to_string_lossy().to_string();

        // Read port 1 state and rate
        let port_dir = entry.path().join("ports/1");
        if !port_dir.exists() {
            continue;
        }

        let state =
            super::read_sysfs_string(&port_dir.join("state"), 256).map(|s| s.trim().to_string());

        let rate_str = super::read_sysfs_string(&port_dir.join("rate"), 256).unwrap_or_default();

        let bandwidth_gbps = parse_ib_rate(rate_str.trim());

        // Determine if IB or RoCE from link_layer file
        let link_layer =
            super::read_sysfs_string(&port_dir.join("link_layer"), 256).unwrap_or_default();
        let kind = if link_layer.trim().eq_ignore_ascii_case("Ethernet") {
            // Distinguish RoCE v1 vs v2 via gid_attrs/types.
            // RoCE v2 uses "RoCE v2" type entries; v1 uses "IB/RoCE v1".
            let is_v2 = detect_roce_version(&port_dir);
            if is_v2 {
                InterconnectKind::RoCEv2
            } else {
                InterconnectKind::RoCE
            }
        } else {
            InterconnectKind::InfiniBand
        };

        debug!(name = %dev_name, %kind, bandwidth_gbps, "interconnect detected");
        interconnects.push(Interconnect {
            kind,
            name: dev_name,
            bandwidth_gbps,
            state,
        });
    }
}

/// Parse InfiniBand rate string like "200 Gb/sec (4X HDR)" → GB/s.
pub fn parse_ib_rate(s: &str) -> f64 {
    // Format: "<number> Gb/sec (...)"
    if let Some(gb_str) = s.split_whitespace().next()
        && let Ok(gbits) = gb_str.parse::<f64>()
    {
        gbits / crate::units::GBITS_PER_GBYTE // Gb/s → GB/s
    } else {
        0.0
    }
}

/// Detect NVLink topology via `nvidia-smi nvlink -s`.
fn detect_nvlink(interconnects: &mut Vec<Interconnect>, warnings: &mut Vec<DetectionError>) {
    let output = match run_tool("nvidia-smi", NVLINK_ARGS, DEFAULT_TIMEOUT) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => return,
        Err(e) => {
            // NVLink failure is non-fatal — the GPU may not have NVLink.
            warnings.push(e);
            return;
        }
    };
    parse_nvlink_output(&output.stdout, interconnects);
}

/// Parse `nvidia-smi nvlink -s` output into interconnect entries.
///
/// Example lines:
///   GPU 0: NVIDIA H100 (UUID: GPU-xxx)
///       Link 0: 25 GB/s
pub fn parse_nvlink_output(stdout: &str, interconnects: &mut Vec<Interconnect>) {
    let mut current_gpu = String::new();
    let mut link_count = 0u32;
    let mut link_bw = 0.0f64;

    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("GPU ") {
            // Flush previous GPU's NVLinks
            if link_count > 0 {
                interconnects.push(Interconnect {
                    kind: InterconnectKind::NVLink,
                    name: current_gpu.clone(),
                    bandwidth_gbps: link_bw * link_count as f64,
                    state: Some(format!("{} links", link_count)),
                });
            }
            current_gpu = trimmed.to_string();
            link_count = 0;
            link_bw = 0.0;
        } else if trimmed.starts_with("Link ") {
            // Cap at 256 links per GPU to bound bandwidth accumulation from malformed output.
            link_count = link_count.saturating_add(1).min(256);
            // Parse "Link N: <bw> GB/s"
            if let Some(bw_part) = trimmed.split(':').nth(1)
                && let Some(bw_str) = bw_part.split_whitespace().next()
                && let Ok(bw) = bw_str.parse::<f64>()
            {
                link_bw = bw;
            }
        }
    }

    // Flush last GPU
    if link_count > 0 {
        interconnects.push(Interconnect {
            kind: InterconnectKind::NVLink,
            name: current_gpu,
            bandwidth_gbps: link_bw * link_count as f64,
            state: Some(format!("{} links", link_count)),
        });
    }
}

// ---------------------------------------------------------------------------
// NVSwitch detection
// ---------------------------------------------------------------------------

/// Detect NVSwitch via sysfs or `nvidia-smi topo -m`.
fn detect_nvswitch(interconnects: &mut Vec<Interconnect>, warnings: &mut Vec<DetectionError>) {
    // Try sysfs first (faster, no subprocess).
    if detect_nvswitch_sysfs(interconnects) {
        return;
    }

    // Fallback: parse `nvidia-smi topo -m` output.
    let output = match run_tool("nvidia-smi", TOPO_ARGS, DEFAULT_TIMEOUT) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => return,
        Err(e) => {
            warnings.push(e);
            return;
        }
    };
    parse_nvswitch_topo(&output.stdout, interconnects);
}

/// Probe `/sys/devices/virtual/nvidia-nvswitch/` for NVSwitch devices.
///
/// Returns `true` if at least one NVSwitch was found.
fn detect_nvswitch_sysfs(interconnects: &mut Vec<Interconnect>) -> bool {
    let nvswitch_dir = std::path::Path::new("/sys/devices/virtual/nvidia-nvswitch");
    let Ok(nv_entries) = std::fs::read_dir(nvswitch_dir) else {
        return false;
    };

    let mut found = false;
    for entry in nv_entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("nvswitch") {
            continue;
        }

        // NVSwitch 3.0 (H100): 900 GB/s bisection bandwidth per switch.
        // NVSwitch 2.0 (A100): 600 GB/s. We use a conservative default.
        let bandwidth_gbps = 900.0;

        debug!(name = %name, bandwidth_gbps, "NVSwitch detected via sysfs");
        interconnects.push(Interconnect {
            kind: InterconnectKind::NVSwitch,
            name,
            bandwidth_gbps,
            state: Some("sysfs".into()),
        });
        found = true;
    }
    found
}

/// Parse `nvidia-smi topo -m` output to detect NVSwitch presence.
///
/// NVSwitch-connected GPUs show "NV#" entries (e.g. "NV12", "NV18") in the
/// topology matrix, where the number is the NVSwitch link count. Non-NVSwitch
/// connections show "SYS", "NODE", "PIX", "PXB", "PHB", or "NV#" with lower
/// numbers.
///
/// Example matrix row (8x H100 DGX):
///   GPU0  X   NV18 NV18 NV18 NV18 NV18 NV18 NV18
pub fn parse_nvswitch_topo(stdout: &str, interconnects: &mut Vec<Interconnect>) {
    // Look for "NV" entries with high link counts (≥ 8 typically indicates NVSwitch).
    let mut max_nv_links = 0u32;
    let mut gpu_count = 0u32;

    for line in stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("GPU") {
            continue;
        }
        gpu_count += 1;

        for token in trimmed.split_whitespace() {
            if let Some(n_str) = token.strip_prefix("NV")
                && let Ok(n) = n_str.parse::<u32>()
            {
                max_nv_links = max_nv_links.max(n);
            }
        }
    }

    // NVSwitch is indicated by high NVLink counts (≥ 8 per GPU pair) and
    // all GPUs having uniform NV connections. DGX H100 shows NV18, A100 NV12.
    if max_nv_links >= 8 && gpu_count >= 2 {
        // Estimate bandwidth: each NVLink 4.0 lane is ~25 GB/s bidirectional.
        let bandwidth_gbps = max_nv_links as f64 * 25.0;

        debug!(
            max_nv_links,
            gpu_count, bandwidth_gbps, "NVSwitch detected via nvidia-smi topo"
        );
        interconnects.push(Interconnect {
            kind: InterconnectKind::NVSwitch,
            name: format!("NVSwitch ({gpu_count} GPUs)"),
            bandwidth_gbps,
            state: Some(format!("NV{max_nv_links}")),
        });
    }
}

// ---------------------------------------------------------------------------
// AMD XGMI / Infinity Fabric detection
// ---------------------------------------------------------------------------

/// Detect AMD XGMI / Infinity Fabric interconnect.
fn detect_xgmi(interconnects: &mut Vec<Interconnect>, warnings: &mut Vec<DetectionError>) {
    // Try sysfs first.
    if detect_xgmi_sysfs(interconnects) {
        return;
    }

    // Fallback: parse `rocm-smi --showtopo` output.
    let output = match run_tool("rocm-smi", ROCM_TOPO_ARGS, DEFAULT_TIMEOUT) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => return,
        Err(e) => {
            warnings.push(e);
            return;
        }
    };
    parse_xgmi_topo(&output.stdout, interconnects);
}

/// Probe sysfs for XGMI links between AMD GPUs.
///
/// XGMI hive ID is exposed at `/sys/class/drm/card*/device/xgmi_hive_info`
/// or `/sys/class/drm/card*/device/xgmi_hive_id`. GPUs in the same hive are
/// interconnected via XGMI.
fn detect_xgmi_sysfs(interconnects: &mut Vec<Interconnect>) -> bool {
    let drm = std::path::Path::new("/sys/class/drm");
    let Ok(drm_entries) = std::fs::read_dir(drm) else {
        return false;
    };

    let mut hive_gpus: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for entry in drm_entries.flatten() {
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

        // Check for XGMI hive ID.
        let hive_id = super::read_sysfs_string(&device_dir.join("xgmi_hive_info"), 256)
            .or_else(|| super::read_sysfs_string(&device_dir.join("xgmi_hive_id"), 256));

        if let Some(hive) = hive_id {
            let hive = hive.trim().to_string();
            // Hive ID of "0" or empty means no XGMI.
            if !hive.is_empty() && hive != "0" && hive != "0x0" {
                hive_gpus
                    .entry(hive)
                    .or_default()
                    .push(name_str.to_string());
            }
        }
    }

    if hive_gpus.is_empty() {
        return false;
    }

    for (hive_id, gpus) in &hive_gpus {
        if gpus.len() < 2 {
            continue;
        }

        // MI300X XGMI: ~896 GB/s per link direction. MI250X: ~400 GB/s.
        // We use a conservative default since we can't easily distinguish.
        let bandwidth_gbps = 400.0;

        debug!(
            hive_id,
            gpu_count = gpus.len(),
            bandwidth_gbps,
            "AMD XGMI hive detected via sysfs"
        );
        interconnects.push(Interconnect {
            kind: InterconnectKind::XgmiInfinityFabric,
            name: format!("XGMI hive {} ({} GPUs)", hive_id, gpus.len()),
            bandwidth_gbps,
            state: Some(format!("{} GPUs", gpus.len())),
        });
    }
    true
}

/// Parse `rocm-smi --showtopo` output to detect XGMI links.
///
/// Example output:
/// ```text
/// ========================= Topology Information =========================
///          GPU0         GPU1
/// GPU0     0            XGMI
/// GPU1     XGMI         0
/// ```
///
/// Or weight-based format:
/// ```text
/// ===================== Inter Node Access (different P2P protocols) ===========
///          GPU0         GPU1
/// GPU0     0            15
/// GPU1     15           0
///
/// ========================= Link Type between two GPUs =========================
///          GPU0         GPU1
/// GPU0     0            XGMI
/// GPU1     XGMI         0
/// ```
pub fn parse_xgmi_topo(stdout: &str, interconnects: &mut Vec<Interconnect>) {
    let mut xgmi_gpu_count = 0u32;
    let mut in_link_section = false;

    for line in stdout.lines() {
        let trimmed = line.trim();

        // Detect the link-type section header.
        if trimmed.contains("Link Type") || trimmed.contains("Topology Information") {
            in_link_section = true;
            continue;
        }

        // A section of "====" after we're in link section means new section.
        if in_link_section && trimmed.starts_with("====") {
            // If we already counted GPUs, we're done.
            if xgmi_gpu_count > 0 {
                break;
            }
            continue;
        }

        if !in_link_section {
            continue;
        }

        // Skip header row (starts with spaces then GPU0, GPU1, ...)
        if trimmed.starts_with("GPU") && trimmed.contains("XGMI") {
            xgmi_gpu_count += 1;
        }
    }

    if xgmi_gpu_count >= 2 {
        let bandwidth_gbps = 400.0; // Conservative default for XGMI.
        debug!(
            xgmi_gpu_count,
            bandwidth_gbps, "XGMI detected via rocm-smi --showtopo"
        );
        interconnects.push(Interconnect {
            kind: InterconnectKind::XgmiInfinityFabric,
            name: format!("XGMI ({xgmi_gpu_count} GPUs)"),
            bandwidth_gbps,
            state: Some(format!("{xgmi_gpu_count} GPUs")),
        });
    }
}

// ---------------------------------------------------------------------------
// Google TPU ICI detection
// ---------------------------------------------------------------------------

/// Detect Google ICI (Inter-Chip Interconnect) for TPU pod slices.
///
/// TPU chips within a pod slice communicate via ICI. The number of chips is
/// read from sysfs; ICI is present when multiple chips are detected.
fn detect_tpu_ici(interconnects: &mut Vec<Interconnect>) {
    let mut total_chips = 0u32;
    let mut version_str = String::new();

    for device_id in super::iter_dev_devices("accel") {
        // Skip AMD XDNA devices that also appear under /dev/accel.
        let driver_link = format!("/sys/class/accel/accel{device_id}/device/driver");
        if let Ok(target) = std::fs::read_link(&driver_link)
            && target.to_string_lossy().contains("amdxdna")
        {
            continue;
        }

        let chip_path = format!("/sys/class/accel/accel{device_id}/device/chip_count");
        if let Some(count_str) = super::read_sysfs_string(std::path::Path::new(&chip_path), 64)
            && let Ok(n) = count_str.trim().parse::<u32>()
            && n > 0
        {
            total_chips += n;
        } else {
            total_chips += 1;
        }

        if version_str.is_empty() {
            let ver_path = format!("/sys/class/accel/accel{device_id}/device/tpu_version");
            if let Some(v) = super::read_sysfs_string(std::path::Path::new(&ver_path), 256) {
                version_str = v.trim().to_string();
            }
        }
    }

    // ICI only matters with multi-chip configurations.
    if total_chips < 2 {
        return;
    }

    // ICI bandwidth varies by TPU version:
    // v4: 4x ICI links @ 48 GB/s each = 192 GB/s per chip
    // v5e: 4x ICI links @ 51.2 GB/s each = 204.8 GB/s per chip
    // v5p: 4x ICI links @ 102.4 GB/s each = 409.6 GB/s per chip
    let per_chip_gbps = if version_str.contains("v5p") {
        409.6
    } else if version_str.contains("v5e") || version_str.contains("v5litepod") {
        204.8
    } else if version_str.contains("v4") {
        192.0
    } else {
        204.8 // Default to v5e.
    };

    let bandwidth_gbps = per_chip_gbps * total_chips as f64;

    debug!(
        total_chips,
        version = %version_str,
        bandwidth_gbps,
        "Google ICI detected"
    );
    interconnects.push(Interconnect {
        kind: InterconnectKind::Ici,
        name: format!("ICI ({total_chips} chips)"),
        bandwidth_gbps,
        state: Some(format!("{total_chips} chips, {version_str}")),
    });
}

// ---------------------------------------------------------------------------
// RoCE v2 detection
// ---------------------------------------------------------------------------

/// Check if a RoCE port supports v2 by reading sysfs `gid_attrs/types`.
///
/// RoCE v2 entries in the GID table have type "RoCE v2" while v1 entries
/// have "IB/RoCE v1". If any v2 entry exists, the port supports RoCE v2.
fn detect_roce_version(port_dir: &std::path::Path) -> bool {
    let types_dir = port_dir.join("gid_attrs").join("types");
    let Ok(type_entries) = std::fs::read_dir(&types_dir) else {
        return false;
    };

    for entry in type_entries.flatten() {
        if let Some(content) = super::read_sysfs_string(&entry.path(), 256)
            && content.trim().contains("RoCE v2")
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ib_rate_hdr() {
        assert!((parse_ib_rate("200 Gb/sec (4X HDR)") - 25.0).abs() < 0.01);
    }

    #[test]
    fn parse_ib_rate_ndr() {
        assert!((parse_ib_rate("400 Gb/sec (4X NDR)") - 50.0).abs() < 0.01);
    }

    #[test]
    fn parse_ib_rate_empty() {
        assert_eq!(parse_ib_rate(""), 0.0);
    }

    // -----------------------------------------------------------------------
    // NVSwitch topo parsing
    // -----------------------------------------------------------------------

    #[test]
    fn parse_nvswitch_topo_dgx_h100() {
        let output = "\
\tGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\t
GPU0\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\t
GPU1\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\t
GPU2\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tNV18\t
GPU3\tNV18\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tNV18\t
GPU4\tNV18\tNV18\tNV18\tNV18\t X \tNV18\tNV18\tNV18\t
GPU5\tNV18\tNV18\tNV18\tNV18\tNV18\t X \tNV18\tNV18\t
GPU6\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\t X \tNV18\t
GPU7\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\t X \t
";
        let mut interconnects = Vec::new();
        parse_nvswitch_topo(output, &mut interconnects);
        assert_eq!(interconnects.len(), 1);
        assert_eq!(interconnects[0].kind, InterconnectKind::NVSwitch);
        assert_eq!(interconnects[0].bandwidth_gbps, 18.0 * 25.0);
        assert_eq!(interconnects[0].state.as_deref(), Some("NV18"));
    }

    #[test]
    fn parse_nvswitch_topo_dgx_a100() {
        let output = "\
\tGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\t
GPU0\t X \tNV12\tNV12\tNV12\tNV12\tNV12\tNV12\tNV12\t
GPU1\tNV12\t X \tNV12\tNV12\tNV12\tNV12\tNV12\tNV12\t
GPU2\tNV12\tNV12\t X \tNV12\tNV12\tNV12\tNV12\tNV12\t
GPU3\tNV12\tNV12\tNV12\t X \tNV12\tNV12\tNV12\tNV12\t
GPU4\tNV12\tNV12\tNV12\tNV12\t X \tNV12\tNV12\tNV12\t
GPU5\tNV12\tNV12\tNV12\tNV12\tNV12\t X \tNV12\tNV12\t
GPU6\tNV12\tNV12\tNV12\tNV12\tNV12\tNV12\t X \tNV12\t
GPU7\tNV12\tNV12\tNV12\tNV12\tNV12\tNV12\tNV12\t X \t
";
        let mut interconnects = Vec::new();
        parse_nvswitch_topo(output, &mut interconnects);
        assert_eq!(interconnects.len(), 1);
        assert_eq!(interconnects[0].kind, InterconnectKind::NVSwitch);
        assert_eq!(interconnects[0].bandwidth_gbps, 12.0 * 25.0);
    }

    #[test]
    fn parse_nvswitch_topo_no_nvswitch() {
        let output = "\
\tGPU0\tGPU1\t
GPU0\t X \tSYS\t
GPU1\tSYS\t X \t
";
        let mut interconnects = Vec::new();
        parse_nvswitch_topo(output, &mut interconnects);
        assert!(interconnects.is_empty());
    }

    #[test]
    fn parse_nvswitch_topo_low_nvlink_no_switch() {
        let output = "\
\tGPU0\tGPU1\t
GPU0\t X \tNV2\t
GPU1\tNV2\t X \t
";
        let mut interconnects = Vec::new();
        parse_nvswitch_topo(output, &mut interconnects);
        assert!(interconnects.is_empty());
    }

    #[test]
    fn parse_nvswitch_topo_empty() {
        let mut interconnects = Vec::new();
        parse_nvswitch_topo("", &mut interconnects);
        assert!(interconnects.is_empty());
    }

    #[test]
    fn parse_nvswitch_topo_single_gpu() {
        let output = "GPU0\t X \t\n";
        let mut interconnects = Vec::new();
        parse_nvswitch_topo(output, &mut interconnects);
        assert!(interconnects.is_empty());
    }

    // -----------------------------------------------------------------------
    // XGMI topology parsing
    // -----------------------------------------------------------------------

    #[test]
    fn parse_xgmi_topo_two_gpus() {
        let output = "\
========================= Link Type between two GPUs =========================
         GPU0         GPU1
GPU0     0            XGMI
GPU1     XGMI         0
";
        let mut interconnects = Vec::new();
        parse_xgmi_topo(output, &mut interconnects);
        assert_eq!(interconnects.len(), 1);
        assert_eq!(interconnects[0].kind, InterconnectKind::XgmiInfinityFabric);
        assert_eq!(interconnects[0].state.as_deref(), Some("2 GPUs"));
    }

    #[test]
    fn parse_xgmi_topo_four_gpus() {
        let output = "\
========================= Topology Information =========================
         GPU0         GPU1         GPU2         GPU3
GPU0     0            XGMI         XGMI         XGMI
GPU1     XGMI         0            XGMI         XGMI
GPU2     XGMI         XGMI         0            XGMI
GPU3     XGMI         XGMI         XGMI         0
";
        let mut interconnects = Vec::new();
        parse_xgmi_topo(output, &mut interconnects);
        assert_eq!(interconnects.len(), 1);
        assert_eq!(interconnects[0].state.as_deref(), Some("4 GPUs"));
    }

    #[test]
    fn parse_xgmi_topo_no_xgmi() {
        let output = "\
========================= Topology Information =========================
         GPU0         GPU1
GPU0     0            PCIE
GPU1     PCIE         0
";
        let mut interconnects = Vec::new();
        parse_xgmi_topo(output, &mut interconnects);
        assert!(interconnects.is_empty());
    }

    #[test]
    fn parse_xgmi_topo_empty() {
        let mut interconnects = Vec::new();
        parse_xgmi_topo("", &mut interconnects);
        assert!(interconnects.is_empty());
    }

    #[test]
    fn parse_xgmi_topo_single_gpu() {
        let output = "\
========================= Link Type between two GPUs =========================
         GPU0
GPU0     0
";
        let mut interconnects = Vec::new();
        parse_xgmi_topo(output, &mut interconnects);
        assert!(interconnects.is_empty());
    }
}
