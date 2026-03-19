//! NVIDIA CUDA GPU detection via `nvidia-smi`.

use tracing::debug;

use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_cuda(profiles: &mut Vec<AcceleratorProfile>) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 3 {
            continue;
        }
        let device_id: u32 = parts[0].parse().unwrap_or(0);
        let mem_total_mb: u64 = parts[1].parse().unwrap_or(8192);
        let compute_cap = parts[2].to_string();

        debug!(device_id, mem_total_mb, "NVIDIA CUDA GPU detected");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::CudaGpu { device_id },
            available: true,
            memory_bytes: mem_total_mb * 1024 * 1024,
            compute_capability: if compute_cap.is_empty() {
                None
            } else {
                Some(compute_cap)
            },
            driver_version: None,
        });
    }
}
