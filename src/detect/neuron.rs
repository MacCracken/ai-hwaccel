//! AWS Inferentia / Trainium (Neuron SDK) detection.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::{AcceleratorType, NeuronChipType};
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_aws_neuron(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    // Use neuron-ls JSON output
    let output = std::process::Command::new("neuron-ls")
        .args(["--json-output"])
        .output();

    if let Ok(o) = output
        && o.status.success()
    {
        let stdout = String::from_utf8_lossy(&o.stdout);
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout)
            && let Some(devices) = json.as_array()
        {
            for (i, device) in devices.iter().enumerate() {
                let model = device["model"].as_str().unwrap_or("Neuron Device");
                let nc_count = device["nc_count"].as_u64().unwrap_or(2) as u32;
                let mem_per_nc = device["memory_per_nc_mb"].as_u64().unwrap_or(8192);
                let mem_total = nc_count as u64 * mem_per_nc * 1024 * 1024;

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
                });
            }
            return;
        } else {
            warnings.push(DetectionError::ParseError {
                backend: "aws-neuron".into(),
                message: "neuron-ls JSON output could not be parsed".into(),
            });
        }
    }

    // Fallback: probe /dev/neuron* devices
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
        });
    }
}
