//! Network interconnect detection: InfiniBand, RoCE, NVLink.
//!
//! Probes sysfs and CLI tools to discover high-speed interconnects
//! used for multi-node training and inter-GPU communication.

use tracing::debug;

use crate::error::DetectionError;
use crate::system_io::{Interconnect, InterconnectKind};

use super::command::{DEFAULT_TIMEOUT, run_tool};

/// Detect all network interconnects on the system.
pub(crate) fn detect_interconnects(warnings: &mut Vec<DetectionError>) -> Vec<Interconnect> {
    let mut interconnects = Vec::new();

    detect_infiniband(&mut interconnects, warnings);
    detect_nvlink(&mut interconnects, warnings);

    interconnects
}

/// Detect InfiniBand / RoCE devices via sysfs (`/sys/class/infiniband/`).
fn detect_infiniband(
    interconnects: &mut Vec<Interconnect>,
    _warnings: &mut Vec<DetectionError>,
) {
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

        let state = std::fs::read_to_string(port_dir.join("state"))
            .ok()
            .map(|s| s.trim().to_string());

        let rate_str = std::fs::read_to_string(port_dir.join("rate"))
            .ok()
            .unwrap_or_default();

        let bandwidth_gbps = parse_ib_rate(rate_str.trim());

        // Determine if IB or RoCE from link_layer file
        let link_layer = std::fs::read_to_string(port_dir.join("link_layer"))
            .ok()
            .unwrap_or_default();
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
fn parse_ib_rate(s: &str) -> f64 {
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
fn detect_nvlink(
    interconnects: &mut Vec<Interconnect>,
    warnings: &mut Vec<DetectionError>,
) {
    let output = match run_tool("nvidia-smi", &["nvlink", "-s"], DEFAULT_TIMEOUT) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => return,
        Err(e) => {
            // NVLink failure is non-fatal — the GPU may not have NVLink.
            warnings.push(e);
            return;
        }
    };

    // Parse nvidia-smi nvlink -s output.
    // Example lines:
    //   GPU 0: NVIDIA H100 (UUID: GPU-xxx)
    //       Link 0: 25 GB/s
    let mut current_gpu = String::new();
    let mut link_count = 0u32;
    let mut link_bw = 0.0f64;

    for line in output.stdout.lines() {
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
            if let Some(bw_part) = trimmed.split(':').nth(1) {
                if let Some(bw_str) = bw_part.trim().split_whitespace().next()
                    && let Ok(bw) = bw_str.parse::<f64>()
                {
                    link_bw = bw;
                }
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
