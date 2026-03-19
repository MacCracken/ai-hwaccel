//! NVIDIA CUDA GPU detection via `nvidia-smi`.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_cuda(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.total,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            warnings.push(DetectionError::ToolFailed {
                tool: "nvidia-smi".into(),
                exit_code: o.status.code(),
                stderr,
            });
            return;
        }
        Err(_) => {
            // nvidia-smi not found — not an error on systems without NVIDIA GPUs
            return;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            warnings.push(DetectionError::ParseError {
                backend: "cuda".into(),
                message: format!("expected 4 CSV fields, got {}: {}", parts.len(), line),
            });
            continue;
        }
        let device_id: u32 = match parts[0].parse() {
            Ok(id) => id,
            Err(e) => {
                warnings.push(DetectionError::ParseError {
                    backend: "cuda".into(),
                    message: format!("invalid device id '{}': {}", parts[0], e),
                });
                continue;
            }
        };
        let mem_total_mb: u64 = parts[1].parse().unwrap_or(8192);
        let compute_cap = parts[2].to_string();
        let driver_version = parts[3].to_string();

        debug!(device_id, mem_total_mb, %driver_version, "NVIDIA CUDA GPU detected");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::CudaGpu { device_id },
            available: true,
            memory_bytes: mem_total_mb * 1024 * 1024,
            compute_capability: if compute_cap.is_empty() {
                None
            } else {
                Some(compute_cap)
            },
            driver_version: if driver_version.is_empty() {
                None
            } else {
                Some(driver_version)
            },
        });
    }
}
