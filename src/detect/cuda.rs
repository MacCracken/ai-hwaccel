//! NVIDIA CUDA GPU detection via `nvidia-smi`.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::command::{run_tool, validate_device_id, validate_memory_mb, DEFAULT_TIMEOUT};

pub(crate) fn detect_cuda(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    let output = match run_tool(
        "nvidia-smi",
        &[
            "--query-gpu=index,memory.total,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        DEFAULT_TIMEOUT,
    ) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => return,
        Err(e) => {
            warnings.push(e);
            return;
        }
    };

    for line in output.stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            warnings.push(DetectionError::ParseError {
                backend: "cuda".into(),
                message: format!("expected 4 CSV fields, got {}: {}", parts.len(), line),
            });
            continue;
        }

        let device_id = match validate_device_id(parts[0], "cuda") {
            Ok(id) => id,
            Err(e) => {
                warnings.push(e);
                continue;
            }
        };
        let mem_total_mb = match validate_memory_mb(parts[1], "cuda") {
            Ok(mb) => mb,
            Err(e) => {
                warnings.push(e);
                continue;
            }
        };
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
