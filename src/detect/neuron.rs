//! AWS Inferentia / Trainium (Neuron SDK) detection.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::{AcceleratorType, NeuronChipType};
use crate::profile::AcceleratorProfile;

use super::command::{DEFAULT_TIMEOUT, run_tool};

const NEURON_LS_ARGS: &[&str] = &["--json-output"];

pub(crate) fn detect_aws_neuron(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    // Use neuron-ls JSON output via safe runner.
    if let Ok(output) = run_tool("neuron-ls", NEURON_LS_ARGS, DEFAULT_TIMEOUT) {
        if parse_neuron_output(&output.stdout, profiles, warnings) {
            return;
        }
    }

    // Fallback: probe /dev/neuron* devices
    detect_neuron_dev_fallback(profiles);
}

#[cfg(feature = "async-detect")]
pub(crate) async fn detect_aws_neuron_async() -> super::DetectResult {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    if let Ok(output) = super::command::run_tool_async("neuron-ls", NEURON_LS_ARGS, DEFAULT_TIMEOUT).await {
        if parse_neuron_output(&output.stdout, &mut profiles, &mut warnings) {
            return (profiles, warnings);
        }
    }

    // Fallback: probe /dev/neuron* devices (sync sysfs, runs inline)
    detect_neuron_dev_fallback(&mut profiles);
    (profiles, warnings)
}

/// Parse `neuron-ls --json-output` stdout. Returns `true` if devices were
/// successfully parsed (even if the list was empty), `false` if JSON parsing
/// failed so the caller should fall back to `/dev` probing.
fn parse_neuron_output(
    stdout: &str,
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) -> bool {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(stdout)
        && let Some(devices) = json.as_array()
    {
        for (i, device) in devices.iter().enumerate() {
            let model = device["model"].as_str().unwrap_or("Neuron Device");
            let nc_count = device["nc_count"].as_u64().unwrap_or(2) as u32;
            let mem_per_nc = device["memory_per_nc_mb"].as_u64().unwrap_or(8192);
            let mem_total = (nc_count as u64)
                .saturating_mul(mem_per_nc)
                .saturating_mul(1024 * 1024);

            let chip_type = if model.contains("trn") || model.contains("Trainium") {
                NeuronChipType::Trainium
            } else {
                NeuronChipType::Inferentia
            };

            debug!(device_id = i, %chip_type, nc_count, "AWS Neuron device detected");
            profiles.push(AcceleratorProfile {
                accelerator: AcceleratorType::AwsNeuron {
                    device_id: i as u32,
                    chip_type,
                    core_count: nc_count,
                },
                available: true,
                memory_bytes: mem_total,
                compute_capability: Some(format!("Neuron {}", chip_type)),
                driver_version: None,
                memory_bandwidth_gbps: None,
                memory_used_bytes: None,
                memory_free_bytes: None,
                pcie_bandwidth_gbps: None,
                numa_node: None,
            });
        }
        true
    } else {
        warnings.push(DetectionError::ParseError {
            backend: "aws-neuron".into(),
            message: "neuron-ls JSON output could not be parsed".into(),
        });
        false
    }
}

/// Fallback: probe /dev/neuron* devices.
fn detect_neuron_dev_fallback(profiles: &mut Vec<AcceleratorProfile>) {
    for entry in std::fs::read_dir("/dev").into_iter().flatten().flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("neuron") {
            continue;
        }
        let suffix = &name_str[6..];
        if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let device_id: u32 = suffix.parse().unwrap_or(0);

        let chip_type = if std::fs::read_to_string("/sys/devices/virtual/dmi/id/product_name")
            .unwrap_or_default()
            .contains("trn")
        {
            NeuronChipType::Trainium
        } else {
            NeuronChipType::Inferentia
        };

        let core_count = 2u32;
        let mem = chip_type.hbm_per_core_bytes() * core_count as u64;

        debug!(device_id, %chip_type, "AWS Neuron device detected via /dev");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::AwsNeuron {
                device_id,
                chip_type,
                core_count,
            },
            available: true,
            memory_bytes: mem,
            compute_capability: Some(format!("Neuron {}", chip_type)),
            driver_version: None,
            memory_bandwidth_gbps: None,
            memory_used_bytes: None,
            memory_free_bytes: None,
            pcie_bandwidth_gbps: None,
            numa_node: None,
        });
    }
}
