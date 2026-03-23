//! Intel Arc / Data Center GPU Max (oneAPI) detection via `xpu-smi`.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::command::{DEFAULT_TIMEOUT, run_tool, validate_device_id, validate_memory_mb};

const XPU_SMI_ARGS: &[&str] = &["discovery", "--dump", "1,2,18,19"];

pub(crate) fn detect_intel_oneapi(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    let output = match run_tool("xpu-smi", XPU_SMI_ARGS, DEFAULT_TIMEOUT) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => {
            debug!("xpu-smi not found on $PATH, skipping Intel oneAPI detection");
            return;
        }
        Err(e) => {
            warnings.push(e);
            return;
        }
    };
    parse_xpu_smi_output(&output.stdout, profiles, warnings);
}

#[cfg(feature = "async-detect")]
pub(crate) async fn detect_intel_oneapi_async() -> super::DetectResult {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let output =
        match super::command::run_tool_async("xpu-smi", XPU_SMI_ARGS, DEFAULT_TIMEOUT).await {
            Ok(o) => o,
            Err(DetectionError::ToolNotFound { .. }) => {
                debug!("xpu-smi not found on $PATH, skipping Intel oneAPI detection");
                return (profiles, warnings);
            }
            Err(e) => {
                warnings.push(e);
                return (profiles, warnings);
            }
        };
    parse_xpu_smi_output(&output.stdout, &mut profiles, &mut warnings);
    (profiles, warnings)
}

pub(crate) fn parse_xpu_smi_output(
    stdout: &str,
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    for line in stdout.lines() {
        if line.starts_with("DeviceId") || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').take(20).map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }
        let device_id = match validate_device_id(parts[0], "intel-oneapi") {
            Ok(id) => id,
            Err(e) => {
                warnings.push(e);
                continue;
            }
        };
        let _name = parts[1].to_string();
        let mem_total_mb = match validate_memory_mb(parts[2], "intel-oneapi") {
            Ok(mb) => mb,
            Err(e) => {
                warnings.push(e);
                continue;
            }
        };

        debug!(device_id, "Intel oneAPI GPU detected via xpu-smi");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::IntelOneApi { device_id },
            available: true,
            memory_bytes: mem_total_mb * 1024 * 1024,
            ..Default::default()
        });
    }
}
