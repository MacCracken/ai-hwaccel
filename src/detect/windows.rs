//! Windows GPU detection via WMI (`wmic`) and PowerShell.
//!
//! On Windows, `nvidia-smi.exe` lives at `C:\Windows\System32\nvidia-smi.exe`
//! (not on `$PATH` in all environments). This module handles:
//!
//! 1. WMIC `Win32_VideoController` parsing for GPU enumeration
//! 2. PowerShell `Get-CimInstance` fallback (wmic deprecated on newer Windows)
//! 3. `nvidia-smi.exe` path resolution at known Windows paths
//!
//! Parsing functions are platform-independent for testing.

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;

use super::command::{DEFAULT_TIMEOUT, run_tool};

/// Known Windows paths for `nvidia-smi.exe`.
#[cfg(target_os = "windows")]
const NVIDIA_SMI_PATHS: &[&str] = &[
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
];

/// Detect GPUs via WMI on Windows.
///
/// Tries `wmic` first, then PowerShell `Get-CimInstance` fallback.
/// On non-Windows, these tools won't exist and detection short-circuits.
pub(crate) fn detect_windows_gpu(
    profiles: &mut Vec<AcceleratorProfile>,
    warnings: &mut Vec<DetectionError>,
) {
    // Try wmic first (available on Windows 10 and earlier).
    if let Ok(output) = run_tool(
        "wmic",
        &[
            "path",
            "Win32_VideoController",
            "get",
            "Name,AdapterRAM,DriverVersion,VideoProcessor",
            "/format:csv",
        ],
        DEFAULT_TIMEOUT,
    ) && parse_wmic_output(&output.stdout, profiles)
    {
        return;
    }

    // PowerShell fallback (wmic deprecated on Windows 11+).
    if let Ok(output) = run_tool(
        "powershell",
        &[
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Csv -NoTypeInformation",
        ],
        DEFAULT_TIMEOUT,
    ) && parse_powershell_csv(&output.stdout, profiles)
    {
        return;
    }

    debug!("no Windows GPU detection method available");
    let _ = warnings;
}

/// Parse `wmic path Win32_VideoController get ... /format:csv` output.
///
/// CSV format: `Node,AdapterRAM,DriverVersion,Name,VideoProcessor`
/// First line is header, subsequent lines are data.
///
/// Returns `true` if at least one GPU was parsed.
pub(crate) fn parse_wmic_output(output: &str, profiles: &mut Vec<AcceleratorProfile>) -> bool {
    let mut found = false;
    let mut device_id: u32 = 0;

    for line in output.lines().skip(1) {
        // skip header
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        // Expected: Node, AdapterRAM, DriverVersion, Name, VideoProcessor
        if fields.len() < 4 {
            continue;
        }

        let adapter_ram: u64 = fields[1].trim().parse().unwrap_or(0);
        let driver_version = {
            let v = fields[2].trim();
            if v.is_empty() {
                None
            } else {
                Some(v.to_string())
            }
        };
        let name = fields[3].trim().to_string();

        // Skip Microsoft Basic Display Adapter and similar virtual devices.
        let name_lower = name.to_lowercase();
        if name_lower.contains("basic display")
            || name_lower.contains("remote desktop")
            || name_lower.contains("microsoft")
        {
            continue;
        }

        debug!(
            device_id,
            name = %name,
            adapter_ram_mb = adapter_ram / (1024 * 1024),
            "Windows GPU detected via WMI"
        );

        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::VulkanGpu { device_id },
            available: true,
            memory_bytes: adapter_ram,
            driver_version,
            device_name: Some(name),
            ..Default::default()
        });
        device_id += 1;
        found = true;
    }

    found
}

/// Parse PowerShell `Get-CimInstance ... | ConvertTo-Csv` output.
///
/// CSV format: `"Name","AdapterRAM","DriverVersion"`
/// First line is header (quoted), subsequent lines are data (quoted).
///
/// Returns `true` if at least one GPU was parsed.
pub(crate) fn parse_powershell_csv(output: &str, profiles: &mut Vec<AcceleratorProfile>) -> bool {
    let mut found = false;
    let mut device_id: u32 = 0;

    for line in output.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Strip quotes and split.
        let fields: Vec<&str> = line
            .split(',')
            .map(|f| f.trim().trim_matches('"'))
            .collect();
        // Expected: Name, AdapterRAM, DriverVersion
        if fields.len() < 3 {
            continue;
        }

        let name = fields[0].to_string();
        let adapter_ram: u64 = fields[1].parse().unwrap_or(0);
        let driver_version = {
            let v = fields[2];
            if v.is_empty() {
                None
            } else {
                Some(v.to_string())
            }
        };

        let name_lower = name.to_lowercase();
        if name_lower.contains("basic display")
            || name_lower.contains("remote desktop")
            || name_lower.contains("microsoft")
        {
            continue;
        }

        debug!(
            device_id,
            name = %name,
            adapter_ram_mb = adapter_ram / (1024 * 1024),
            "Windows GPU detected via PowerShell"
        );

        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::VulkanGpu { device_id },
            available: true,
            memory_bytes: adapter_ram,
            driver_version,
            device_name: Some(name),
            ..Default::default()
        });
        device_id += 1;
        found = true;
    }

    found
}

/// Resolve `nvidia-smi.exe` on Windows from known installation paths.
///
/// Returns the first existing path, or `None` if not found.
#[cfg(target_os = "windows")]
#[must_use]
pub(crate) fn find_nvidia_smi_windows() -> Option<&'static str> {
    NVIDIA_SMI_PATHS
        .iter()
        .find(|path| std::path::Path::new(path).exists())
        .copied()
}
