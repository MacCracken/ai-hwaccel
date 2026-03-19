//! Intel Arc / Data Center GPU Max (oneAPI) detection via `xpu-smi`.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_intel_oneapi(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    let output = std::process::Command::new("xpu-smi")
        .args(["discovery", "--dump", "1,2,18,19"])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            warnings.push(DetectionError::ToolFailed {
                tool: "xpu-smi".into(),
                exit_code: o.status.code(),
                stderr: String::from_utf8_lossy(&o.stderr).to_string(),
            });
            return;
        }
        Err(_) => return,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.starts_with("DeviceId") || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }
        let device_id: u32 = parts[0].parse().unwrap_or(0);
        let _name = parts[1].to_string();
        let mem_total_mb: u64 = parts[2].parse().unwrap_or(0);

        debug!(device_id, "Intel oneAPI GPU detected via xpu-smi");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::IntelOneApi { device_id },
            available: true,
            memory_bytes: mem_total_mb * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        });
    }
}
