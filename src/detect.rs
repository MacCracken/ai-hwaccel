//! Hardware detection: probes sysfs, /dev, and PATH tools to discover accelerators.

use std::path::Path;

use tracing::debug;

use crate::types::*;

// ---------------------------------------------------------------------------
// AcceleratorRegistry
// ---------------------------------------------------------------------------

/// Registry of detected hardware accelerators with planning helpers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AcceleratorRegistry {
    profiles: Vec<AcceleratorProfile>,
}

impl AcceleratorRegistry {
    /// Creates a registry containing only a default CPU profile.
    pub fn new() -> Self {
        Self {
            profiles: vec![cpu_profile()],
        }
    }

    /// Probes the system for all available accelerators.
    ///
    /// Detection is best-effort: missing tools or sysfs entries simply mean
    /// the corresponding accelerator is not registered.
    pub fn detect() -> Self {
        let mut profiles = vec![cpu_profile()];

        detect_cuda(&mut profiles);
        detect_rocm(&mut profiles);
        detect_metal_and_ane(&mut profiles);
        detect_vulkan(&mut profiles);
        detect_intel_npu(&mut profiles);
        detect_amd_xdna(&mut profiles);
        detect_tpu(&mut profiles);
        detect_gaudi(&mut profiles);
        detect_aws_neuron(&mut profiles);
        detect_intel_oneapi(&mut profiles);
        detect_qualcomm_ai100(&mut profiles);

        debug!(count = profiles.len(), "accelerator detection complete");
        Self { profiles }
    }

    /// Build a registry from a pre-built list of profiles (for testing or config-driven setups).
    pub fn from_profiles(profiles: Vec<AcceleratorProfile>) -> Self {
        Self { profiles }
    }

    /// All registered profiles (including unavailable ones).
    pub fn all_profiles(&self) -> &[AcceleratorProfile] {
        &self.profiles
    }

    /// Only the available accelerator profiles.
    pub fn available(&self) -> Vec<&AcceleratorProfile> {
        self.profiles.iter().filter(|p| p.available).collect()
    }

    /// The highest-ranked available device.
    pub fn best_available(&self) -> Option<&AcceleratorProfile> {
        self.profiles
            .iter()
            .filter(|p| p.available)
            .max_by_key(|p| p.accelerator.rank())
    }

    /// Total memory across all **available** devices.
    pub fn total_memory(&self) -> u64 {
        self.profiles
            .iter()
            .filter(|p| p.available)
            .map(|p| p.memory_bytes)
            .sum()
    }

    /// Total memory across all available non-CPU devices (GPU + NPU + TPU + ASIC).
    pub fn total_accelerator_memory(&self) -> u64 {
        self.profiles
            .iter()
            .filter(|p| p.available && !matches!(p.accelerator, AcceleratorType::Cpu))
            .map(|p| p.memory_bytes)
            .sum()
    }

    /// Whether any non-CPU accelerator is available.
    pub fn has_accelerator(&self) -> bool {
        self.profiles
            .iter()
            .any(|p| p.available && !matches!(p.accelerator, AcceleratorType::Cpu))
    }

    /// All profiles matching a given [`AcceleratorFamily`].
    pub fn by_family(&self, family: AcceleratorFamily) -> Vec<&AcceleratorProfile> {
        self.profiles
            .iter()
            .filter(|p| p.available && p.accelerator.family() == family)
            .collect()
    }

    /// All profiles satisfying an [`AcceleratorRequirement`].
    pub fn satisfying(&self, req: &AcceleratorRequirement) -> Vec<&AcceleratorProfile> {
        self.profiles
            .iter()
            .filter(|p| req.satisfied_by(p))
            .collect()
    }

    /// Add a profile manually (for testing or manual config).
    pub fn add_profile(&mut self, profile: AcceleratorProfile) {
        self.profiles.push(profile);
    }

    /// Estimate memory required for `model_params` parameters at the given quantisation.
    ///
    /// Formula: `params * (bits / 8)` plus 20% overhead for activations/KV cache.
    pub fn estimate_memory(model_params: u64, quant: &QuantizationLevel) -> u64 {
        let bytes_per_param = quant.bits_per_param() as u64;
        let raw = model_params * bytes_per_param / 8;
        raw + raw / 5
    }

    /// Suggest a quantisation level based on available hardware and model size.
    pub fn suggest_quantization(&self, model_params: u64) -> QuantizationLevel {
        // Check for TPU first — TPUs strongly prefer BFloat16
        if let Some(tpu_mem) = self.best_memory_for(AcceleratorFamily::Tpu) {
            if Self::estimate_memory(model_params, &QuantizationLevel::BFloat16) <= tpu_mem {
                return QuantizationLevel::BFloat16;
            }
            if Self::estimate_memory(model_params, &QuantizationLevel::Int8) <= tpu_mem {
                return QuantizationLevel::Int8;
            }
        }

        // Check for Gaudi — also prefers BFloat16
        if let Some(gaudi_mem) = self.best_memory_for(AcceleratorFamily::AiAsic)
            && Self::estimate_memory(model_params, &QuantizationLevel::BFloat16) <= gaudi_mem
        {
            return QuantizationLevel::BFloat16;
        }

        // Check GPU
        if let Some(gpu_mem) = self.best_memory_for(AcceleratorFamily::Gpu) {
            for quant in &[
                QuantizationLevel::Float16,
                QuantizationLevel::Int8,
                QuantizationLevel::Int4,
            ] {
                if Self::estimate_memory(model_params, quant) <= gpu_mem {
                    return *quant;
                }
            }
        }

        // Check NPU (INT8/INT4 only)
        if let Some(npu_mem) = self.best_memory_for(AcceleratorFamily::Npu) {
            for quant in &[QuantizationLevel::Int8, QuantizationLevel::Int4] {
                if Self::estimate_memory(model_params, quant) <= npu_mem {
                    return *quant;
                }
            }
        }

        // Fallback: FP16 on CPU
        QuantizationLevel::Float16
    }

    /// Returns the largest device memory for available devices of a given family.
    fn best_memory_for(&self, family: AcceleratorFamily) -> Option<u64> {
        self.profiles
            .iter()
            .filter(|p| p.available && p.accelerator.family() == family)
            .map(|p| p.memory_bytes)
            .max()
    }
}

impl Default for AcceleratorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Detection helpers
// ---------------------------------------------------------------------------

fn cpu_profile() -> AcceleratorProfile {
    AcceleratorProfile {
        accelerator: AcceleratorType::Cpu,
        available: true,
        memory_bytes: detect_cpu_memory(),
        compute_capability: None,
        driver_version: None,
    }
}

/// System memory from /proc/meminfo (fallback: 16 GiB).
fn detect_cpu_memory() -> u64 {
    if let Ok(info) = std::fs::read_to_string("/proc/meminfo") {
        for line in info.lines() {
            if line.starts_with("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(kb_str) = parts.get(1)
                    && let Ok(kb) = kb_str.parse::<u64>()
                {
                    return kb * 1024;
                }
            }
        }
    }
    16 * 1024 * 1024 * 1024
}

/// Check if an executable is on `$PATH`.
fn which_exists(name: &str) -> bool {
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            if Path::new(dir).join(name).exists() {
                return true;
            }
        }
    }
    false
}

// -- NVIDIA CUDA --

fn detect_cuda(profiles: &mut Vec<AcceleratorProfile>) {
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

// -- AMD ROCm --

fn detect_rocm(profiles: &mut Vec<AcceleratorProfile>) {
    let drm = Path::new("/sys/class/drm");
    if !drm.exists() {
        return;
    }

    let mut device_id = 0u32;
    for entry in std::fs::read_dir(drm).into_iter().flatten().flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }

        let device_dir = entry.path().join("device");
        let driver_link = device_dir.join("driver");
        let driver_name = std::fs::read_link(&driver_link)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));

        if driver_name.as_deref() != Some("amdgpu") {
            continue;
        }

        let mem_total = read_sysfs_u64(&device_dir.join("mem_info_vram_total"))
            .unwrap_or(8 * 1024 * 1024 * 1024);

        debug!(device_id, "AMD ROCm GPU detected via sysfs");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::RocmGpu { device_id },
            available: true,
            memory_bytes: mem_total,
            compute_capability: None,
            driver_version: None,
        });
        device_id += 1;
    }
}

fn read_sysfs_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

// -- Apple Metal + ANE --

fn detect_metal_and_ane(profiles: &mut Vec<AcceleratorProfile>) {
    if let Ok(compat) = std::fs::read_to_string("/proc/device-tree/compatible")
        && compat.contains("apple")
    {
        debug!("Apple device detected, registering Metal GPU + ANE");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::MetalGpu,
            available: true,
            memory_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        });
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::AppleNpu,
            available: true,
            memory_bytes: 4 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        });
    }
}

// -- Vulkan --

fn detect_vulkan(profiles: &mut Vec<AcceleratorProfile>) {
    if which_exists("vulkaninfo") {
        // In a real implementation, we'd parse vulkaninfo output for device names
        // and memory. For now, register a generic Vulkan device if the tool exists
        // but only if we didn't already find a CUDA or ROCm GPU (avoid double-counting).
        let has_dedicated_gpu = profiles.iter().any(|p| {
            matches!(
                p.accelerator,
                AcceleratorType::CudaGpu { .. } | AcceleratorType::RocmGpu { .. }
            )
        });

        if !has_dedicated_gpu {
            debug!("vulkaninfo found (no CUDA/ROCm), registering Vulkan GPU");
            profiles.push(AcceleratorProfile {
                accelerator: AcceleratorType::VulkanGpu {
                    device_id: 0,
                    device_name: "Unknown Vulkan Device".into(),
                },
                available: true,
                memory_bytes: 4 * 1024 * 1024 * 1024,
                compute_capability: None,
                driver_version: None,
            });
        }
    }
}

// -- Intel NPU --

fn detect_intel_npu(profiles: &mut Vec<AcceleratorProfile>) {
    if Path::new("/sys/class/misc/intel_npu").exists() {
        debug!("Intel NPU detected via sysfs");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::IntelNpu,
            available: true,
            memory_bytes: 2 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        });
    }
}

// -- AMD XDNA / Ryzen AI NPU --

fn detect_amd_xdna(profiles: &mut Vec<AcceleratorProfile>) {
    // AMD XDNA NPUs appear under /sys/class/accel/ with driver name "amdxdna"
    let accel_dir = Path::new("/sys/class/accel");
    if !accel_dir.exists() {
        return;
    }
    for entry in std::fs::read_dir(accel_dir).into_iter().flatten().flatten() {
        let driver_link = entry.path().join("device/driver");
        if let Ok(target) = std::fs::read_link(&driver_link) {
            let driver_name = target
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if driver_name == "amdxdna" {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                let device_id: u32 = name_str
                    .strip_prefix("accel")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                debug!(device_id, "AMD XDNA NPU detected");
                profiles.push(AcceleratorProfile {
                    accelerator: AcceleratorType::AmdXdnaNpu { device_id },
                    available: true,
                    memory_bytes: 2 * 1024 * 1024 * 1024, // shared system memory
                    compute_capability: None,
                    driver_version: None,
                });
            }
        }
    }
}

// -- Google TPU --

fn detect_tpu(profiles: &mut Vec<AcceleratorProfile>) {
    // TPU kernel driver exposes /dev/accel0, /dev/accel1, etc.
    // We filter out AMD XDNA devices by checking the driver is NOT amdxdna.
    let dev_dir = Path::new("/dev");
    for entry in std::fs::read_dir(dev_dir).into_iter().flatten().flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("accel") {
            continue;
        }
        let suffix = &name_str[5..];
        if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let device_id: u32 = suffix.parse().unwrap_or(0);

        // Skip if this is an AMD XDNA device
        let driver_link = format!("/sys/class/accel/accel{}/device/driver", device_id);
        if let Ok(target) = std::fs::read_link(&driver_link)
            && target.to_string_lossy().contains("amdxdna")
        {
            continue;
        }

        let version = detect_tpu_version(device_id);
        let chip_count = detect_tpu_chip_count(device_id);
        let hbm = version.hbm_per_chip_bytes() * chip_count as u64;

        debug!(device_id, %version, chip_count, "Google TPU detected");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::Tpu {
                device_id,
                chip_count,
                version,
            },
            available: true,
            memory_bytes: hbm,
            compute_capability: Some(format!("TPU {}", version)),
            driver_version: None,
        });
    }
}

fn detect_tpu_version(device_id: u32) -> TpuVersion {
    let path = format!("/sys/class/accel/accel{}/device/tpu_version", device_id);
    if let Ok(ver) = std::fs::read_to_string(&path) {
        let ver = ver.trim();
        if ver.contains("v5p") {
            return TpuVersion::V5p;
        } else if ver.contains("v5e") || ver.contains("v5litepod") {
            return TpuVersion::V5e;
        } else if ver.contains("v4") {
            return TpuVersion::V4;
        }
    }
    TpuVersion::V5e // default — most common cloud TPU
}

fn detect_tpu_chip_count(device_id: u32) -> u32 {
    let path = format!("/sys/class/accel/accel{}/device/chip_count", device_id);
    if let Ok(count) = std::fs::read_to_string(&path)
        && let Ok(n) = count.trim().parse::<u32>()
        && n > 0
    {
        return n;
    }
    1
}

// -- Intel Gaudi (Habana Labs HPU) --

fn detect_gaudi(profiles: &mut Vec<AcceleratorProfile>) {
    let output = std::process::Command::new("hl-smi")
        .args([
            "--query-aip=index,name,memory.total,memory.free",
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

// -- AWS Inferentia / Trainium (Neuron SDK) --

fn detect_aws_neuron(profiles: &mut Vec<AcceleratorProfile>) {
    // Use neuron-ls JSON output like Synapse does
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

        // Check DMI for instance type hint
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

// -- Intel Arc / Data Center GPU Max (oneAPI) --

fn detect_intel_oneapi(profiles: &mut Vec<AcceleratorProfile>) {
    // xpu-smi is the Intel GPU management tool for oneAPI
    let output = std::process::Command::new("xpu-smi")
        .args(["discovery", "--dump", "1,2,18,19"])
        .output();

    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return,
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

// -- Qualcomm Cloud AI 100 --

fn detect_qualcomm_ai100(profiles: &mut Vec<AcceleratorProfile>) {
    // Qualcomm AI 100 appears as /dev/qaic_* or under /sys/class/qaic
    if Path::new("/sys/class/qaic").exists() {
        debug!("Qualcomm Cloud AI 100 detected via sysfs");
        profiles.push(AcceleratorProfile {
            accelerator: AcceleratorType::QualcommAi100 { device_id: 0 },
            available: true,
            memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB DDR
            compute_capability: Some("AI 100".into()),
            driver_version: None,
        });
        return;
    }

    // Fallback: check for /dev/qaic_* devices
    for entry in std::fs::read_dir("/dev").into_iter().flatten().flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("qaic_") {
            debug!("Qualcomm Cloud AI 100 detected via /dev");
            profiles.push(AcceleratorProfile {
                accelerator: AcceleratorType::QualcommAi100 { device_id: 0 },
                available: true,
                memory_bytes: 32 * 1024 * 1024 * 1024,
                compute_capability: Some("AI 100".into()),
                driver_version: None,
            });
            return;
        }
    }
}
