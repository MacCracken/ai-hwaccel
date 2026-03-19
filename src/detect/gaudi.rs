//! Intel Gaudi (Habana Labs HPU) detection via `hl-smi`.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::{AcceleratorType, GaudiGeneration};
use crate::profile::AcceleratorProfile;

pub(crate) fn detect_gaudi(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    let output = std::process::Command::new("hl-smi")
        .args([
            "--query-aip=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            warnings.push(DetectionError::ToolFailed {
                tool: "hl-smi".into(),
                exit_code: o.status.code(),
                stderr: String::from_utf8_lossy(&o.stderr).to_string(),
            });
            return;
        }
        Err(_) => return,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }
        let device_id: u32 = parts[0].parse().unwrap_or(0);
        let name = parts[1].to_lowercase();
        let mem_total_mb: u64 = parts[2].parse().unwrap_or(0);

        let generation = if name.contains("gaudi3") || name.contains("hl-325") {
            GaudiGeneration::Gaudi3
        } else {
            GaudiGeneration::Gaudi2
        };

        debug!(device_id, %generation, "Intel Gaudi HPU detected");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::Gaudi {
                device_id,
                generation,
            },
            available: true,
            memory_bytes: mem_total_mb * 1024 * 1024,
            compute_capability: Some(generation.to_string()),
            driver_version: None,
        });
    }
}
