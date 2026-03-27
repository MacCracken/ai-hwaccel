//! NVIDIA CUDA GPU detection via `nvidia-smi`.

use tracing::{debug, trace};

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::bandwidth::{estimate_nvidia_bandwidth_from_cc, nvidia_bus_width_bits};
use super::command::{DEFAULT_TIMEOUT, run_tool, validate_device_id, validate_memory_mb};

const NVIDIA_SMI_ARGS: &[&str] = &[
    "--query-gpu=index,memory.total,memory.used,memory.free,compute_cap,driver_version,name,temperature.gpu,power.draw,utilization.gpu,clocks.max.memory",
    "--format=csv,noheader,nounits",
];

pub(crate) fn detect_cuda(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    let output = match run_tool("nvidia-smi", NVIDIA_SMI_ARGS, DEFAULT_TIMEOUT) {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => {
            debug!("nvidia-smi not found on $PATH, skipping CUDA detection");
            return;
        }
        Err(e) => {
            warnings.push(e);
            return;
        }
    };
    parse_cuda_output(&output.stdout, profiles, warnings);
}

#[cfg(feature = "async-detect")]
pub(crate) async fn detect_cuda_async() -> super::DetectResult {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let output = match super::command::run_tool_async(
        "nvidia-smi",
        NVIDIA_SMI_ARGS,
        DEFAULT_TIMEOUT,
    )
    .await
    {
        Ok(o) => o,
        Err(DetectionError::ToolNotFound { .. }) => {
            debug!("nvidia-smi not found on $PATH, skipping CUDA detection");
            return (profiles, warnings);
        }
        Err(e) => {
            warnings.push(e);
            return (profiles, warnings);
        }
    };
    parse_cuda_output(&output.stdout, &mut profiles, &mut warnings);
    (profiles, warnings)
}

pub fn parse_cuda_output(
    stdout: &str,
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    for line in stdout.lines() {
        trace!(line, "parsing nvidia-smi CSV line");
        let parts: Vec<&str> = line.split(',').take(20).map(|s| s.trim()).collect();
        // Accept 6 fields (legacy) or 7 (with gpu name).
        if parts.len() < 6 {
            warnings.push(DetectionError::ParseError {
                backend: "cuda".into(),
                message: format!("expected 6+ CSV fields, got {}: {}", parts.len(), line),
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
        let mem_used_mb: Option<u64> = parts[2].parse().ok().filter(|&v| v <= 16 * 1024 * 1024);
        let mem_free_mb: Option<u64> = parts[3].parse().ok().filter(|&v| v <= 16 * 1024 * 1024);
        let compute_cap_str = *parts.get(4).unwrap_or(&"");
        let driver_version_str = *parts.get(5).unwrap_or(&"");
        let gpu_name = if parts.len() > 6 { parts[6] } else { "" };
        let temp_c: Option<u32> = parts.get(7).and_then(|s| s.parse().ok());
        let power_w: Option<f64> = parts.get(8).and_then(|s| s.parse().ok());
        let gpu_util: Option<u32> = parts.get(9).and_then(|s| s.parse().ok());
        let max_mem_clock_mhz: Option<f64> = parts.get(10).and_then(|s| s.parse().ok());

        // Grace Hopper detection: GH200 has unified CPU+GPU memory via NVLink-C2C.
        // The GPU can access system memory seamlessly, so the effective memory
        // pool is much larger than the reported GPU VRAM.
        let is_grace_hopper = gpu_name.contains("GH200")
            || gpu_name.contains("GH100")
            || gpu_name.contains("Grace Hopper");
        if is_grace_hopper {
            debug!(
                device_id,
                gpu_name, "Grace Hopper detected — unified CPU+GPU memory via NVLink-C2C"
            );
        }

        let mut effective_mem = mem_total_mb.saturating_mul(1024 * 1024);
        // Grace Hopper GH200 has 96 GB HBM3 + up to 480 GB LPDDR5X unified.
        // nvidia-smi reports only HBM. Add system memory estimate for planning.
        if is_grace_hopper
            && (80 * 1024 * 1024 * 1024..100 * 1024 * 1024 * 1024).contains(&effective_mem)
        {
            // HBM reported, add unified CPU memory (typical GH200: 480 GB).
            effective_mem = effective_mem.saturating_add(480 * 1024 * 1024 * 1024);
        }

        // Calculate memory bandwidth inline from max memory clock + bus width lookup.
        let memory_bandwidth_gbps = max_mem_clock_mhz
            .and_then(|clock_mhz| {
                nvidia_bus_width_bits(compute_cap_str).map(|bus_width| {
                    let bw = clock_mhz * bus_width as f64 * 2.0 / 8.0 / 1000.0;
                    (bw * 10.0).round() / 10.0
                })
            })
            .or_else(|| estimate_nvidia_bandwidth_from_cc(compute_cap_str));

        debug!(
            device_id,
            mem_total_mb,
            ?mem_used_mb,
            ?mem_free_mb,
            driver_version = driver_version_str,
            gpu_name,
            "NVIDIA CUDA GPU detected"
        );
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::CudaGpu { device_id },
            available: true,
            memory_bytes: effective_mem,
            compute_capability: if compute_cap_str.is_empty() {
                None
            } else {
                Some(compute_cap_str.to_string())
            },
            driver_version: if driver_version_str.is_empty() {
                None
            } else {
                Some(driver_version_str.to_string())
            },
            device_name: if gpu_name.is_empty() {
                None
            } else {
                Some(gpu_name.to_string())
            },
            memory_bandwidth_gbps,
            memory_used_bytes: mem_used_mb.map(|mb| mb.saturating_mul(1024 * 1024)),
            memory_free_bytes: mem_free_mb.map(|mb| mb.saturating_mul(1024 * 1024)),
            pcie_bandwidth_gbps: None,
            numa_node: None,
            temperature_c: temp_c,
            power_watts: power_w,
            gpu_utilization_percent: gpu_util,
        });
    }
}
