//! Network interconnect detection: InfiniBand, RoCE, NVLink.
//!
//! Probes sysfs and CLI tools to discover high-speed interconnects
//! used for multi-node training and inter-GPU communication.

use tracing::debug;

use crate::error::DetectionError;
use crate::system_io::{Interconnect, InterconnectKind};

use super::command::{DEFAULT_TIMEOUT, run_tool};

const NVLINK_ARGS: &[&str] = &["nvlink", "-s"];

/// Detect all network interconnects on the system.
pub(crate) fn detect_interconnects(warnings: &mut Vec<DetectionError>) -> Vec<Interconnect> {
    let mut interconnects = Vec::new();

    detect_infiniband(&mut interconnects, warnings);
    detect_nvlink(&mut interconnects, warnings);

    interconnects
}

#[cfg(feature = "async-detect")]
pub(crate) async fn detect_interconnects_async() -> (Vec<Interconnect>, Vec<DetectionError>) {
    let mut interconnects = Vec::new();
    let mut warnings = Vec::new();

    // InfiniBand: sync sysfs probing, runs inline.
    detect_infiniband(&mut interconnects, &mut warnings);

    // NVLink: async CLI call.
    if let Ok(output) =
        super::command::run_tool_async("nvidia-smi", NVLINK_ARGS, DEFAULT_TIMEOUT).await
    {
        parse_nvlink_output(&output.stdout, &mut interconnects);
    }

    (interconnects, warnings)
}

/// Detect InfiniBand / RoCE devices via sysfs (`/sys/class/infiniband/`).
fn detect_infiniband(interconnects: &mut Vec<Interconnect>, _warnings: &mut Vec<DetectionError>) {
    let ib_dir = std::path::Path::new("/sys/class/infiniband");
    if !ib_dir.exists() {
        return;
    }

    for entry in std::fs::read_dir(ib_dir).into_iter().flatten().flatten() {
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
            InterconnectKind::RoCE
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
pub(crate) fn parse_ib_rate(s: &str) -> f64 {
    // Format: "<number> Gb/sec (...)"
    if let Some(gb_str) = s.split_whitespace().next()
        && let Ok(gbits) = gb_str.parse::<f64>()
    {
        gbits / 8.0 // Gb/s → GB/s
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
pub(crate) fn parse_nvlink_output(stdout: &str, interconnects: &mut Vec<Interconnect>) {
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
            link_count += 1;
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
}
