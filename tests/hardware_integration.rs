//! Hardware integration tests — run on real systems to verify detection.
//!
//! These tests probe actual hardware and sysfs, so results depend on the
//! machine. Each test is gated on conditions that detect whether the
//! relevant hardware/tool is present, and skips gracefully if not.

use ai_hwaccel::{
    AcceleratorFamily, AcceleratorProfile, AcceleratorRegistry, AcceleratorType, DetectBuilder,
    SystemIo,
};

// ---------------------------------------------------------------------------
// CPU (always present)
// ---------------------------------------------------------------------------

#[test]
fn cpu_always_detected() {
    let reg = AcceleratorRegistry::detect();
    let cpus: Vec<_> = reg
        .all_profiles()
        .iter()
        .filter(|p| matches!(p.accelerator, AcceleratorType::Cpu))
        .collect();
    assert_eq!(cpus.len(), 1, "exactly one CPU profile expected");
    assert!(cpus[0].available);
    assert!(cpus[0].memory_bytes > 0, "CPU memory should be > 0");
}

#[test]
fn detect_returns_at_least_cpu() {
    let reg = DetectBuilder::none().detect();
    assert!(!reg.all_profiles().is_empty());
    assert!(matches!(
        reg.all_profiles()[0].accelerator,
        AcceleratorType::Cpu
    ));
}

// ---------------------------------------------------------------------------
// ROCm (present if /sys/class/drm/card*/device/driver -> amdgpu)
// ---------------------------------------------------------------------------

fn has_rocm() -> bool {
    let drm = std::path::Path::new("/sys/class/drm");
    if !drm.exists() {
        return false;
    }
    for entry in std::fs::read_dir(drm).into_iter().flatten().flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("card") || name_str.contains('-') {
            continue;
        }
        if let Ok(target) = std::fs::read_link(entry.path().join("device/driver")) {
            if target.to_string_lossy().contains("amdgpu") {
                return true;
            }
        }
    }
    false
}

#[test]
fn rocm_detection_when_present() {
    if !has_rocm() {
        eprintln!("SKIP: no amdgpu driver detected");
        return;
    }
    let reg = DetectBuilder::none().with_rocm().detect();
    let gpus: Vec<_> = reg.by_family(AcceleratorFamily::Gpu);
    assert!(!gpus.is_empty(), "ROCm GPU should be detected");

    for gpu in &gpus {
        assert!(gpu.memory_bytes > 0, "VRAM should be > 0");
        assert!(gpu.available);
    }
}

#[test]
fn rocm_vram_usage_when_present() {
    if !has_rocm() {
        eprintln!("SKIP: no amdgpu driver detected");
        return;
    }
    let reg = DetectBuilder::none().with_rocm().detect();
    for gpu in reg.by_family(AcceleratorFamily::Gpu) {
        // At minimum, used bytes should be reported.
        assert!(
            gpu.memory_used_bytes.is_some(),
            "ROCm should report memory_used_bytes"
        );
        assert!(
            gpu.memory_free_bytes.is_some(),
            "ROCm should report memory_free_bytes"
        );
        // Free + used should approximately equal total.
        if let (Some(used), Some(free)) = (gpu.memory_used_bytes, gpu.memory_free_bytes) {
            let sum = used + free;
            let diff = (sum as i64 - gpu.memory_bytes as i64).unsigned_abs();
            assert!(
                diff < 64 * 1024 * 1024,
                "used + free should be ~total (diff: {} bytes)",
                diff
            );
        }
    }
}

#[test]
fn rocm_firmware_version_when_present() {
    if !has_rocm() {
        eprintln!("SKIP: no amdgpu driver detected");
        return;
    }
    let reg = DetectBuilder::none().with_rocm().detect();
    for gpu in reg.by_family(AcceleratorFamily::Gpu) {
        // VBIOS version should be populated as driver_version.
        assert!(
            gpu.driver_version.is_some(),
            "ROCm should report VBIOS as driver_version"
        );
    }
}

// ---------------------------------------------------------------------------
// Vulkan (present if vulkaninfo is on PATH)
// ---------------------------------------------------------------------------

fn has_vulkaninfo() -> bool {
    std::process::Command::new("vulkaninfo")
        .arg("--summary")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
fn vulkan_detection_when_present() {
    if !has_vulkaninfo() {
        eprintln!("SKIP: vulkaninfo not found");
        return;
    }
    // Use vulkan-only builder to avoid ROCm dedup removing vulkan.
    let reg = DetectBuilder::none().with_vulkan().detect();
    let gpus: Vec<_> = reg
        .all_profiles()
        .iter()
        .filter(|p| matches!(p.accelerator, AcceleratorType::VulkanGpu { .. }))
        .collect();
    assert!(!gpus.is_empty(), "Vulkan GPU should be detected");

    for gpu in &gpus {
        assert!(gpu.memory_bytes > 0);
        // compute_capability should now contain compute queue info.
        if let Some(cc) = &gpu.compute_capability {
            assert!(
                cc.contains("Vulkan") || cc.contains("compute"),
                "compute_capability should have Vulkan or compute info, got: {cc}"
            );
        }
    }
}

#[test]
fn vulkan_compute_queues_when_present() {
    if !has_vulkaninfo() {
        eprintln!("SKIP: vulkaninfo not found");
        return;
    }
    let reg = DetectBuilder::none().with_vulkan().detect();
    for p in reg.all_profiles() {
        if let AcceleratorType::VulkanGpu { .. } = &p.accelerator {
            if let Some(cc) = &p.compute_capability {
                // Should mention subgroup size if full vulkaninfo worked.
                if cc.contains("subgroup") {
                    assert!(
                        cc.contains("subgroup: "),
                        "subgroup info should be present: {cc}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PCIe link detection
// ---------------------------------------------------------------------------

#[test]
fn pcie_bandwidth_for_gpu() {
    let reg = AcceleratorRegistry::detect();
    for p in reg.all_profiles() {
        match &p.accelerator {
            AcceleratorType::CudaGpu { .. } | AcceleratorType::RocmGpu { .. } => {
                // GPUs should have PCIe bandwidth on systems with sysfs.
                if std::path::Path::new("/sys/bus/pci").exists() {
                    // Note: may be None if the PCI address can't be mapped.
                    // We just verify it doesn't panic.
                    let _ = p.pcie_bandwidth_gbps;
                }
            }
            _ => {}
        }
    }
}

#[test]
fn pcie_bandwidth_is_reasonable() {
    let reg = AcceleratorRegistry::detect();
    for p in reg.all_profiles() {
        if let Some(bw) = p.pcie_bandwidth_gbps {
            // PCIe bandwidth should be between 0.5 GB/s (Gen1 x1) and 128 GB/s (Gen5 x16).
            assert!(
                bw >= 0.5 && bw <= 128.0,
                "PCIe bandwidth {bw} GB/s out of range for {}",
                p.accelerator
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Memory bandwidth
// ---------------------------------------------------------------------------

#[test]
fn memory_bandwidth_is_reasonable() {
    let reg = AcceleratorRegistry::detect();
    for p in reg.all_profiles() {
        if let Some(bw) = p.memory_bandwidth_gbps {
            // Memory bandwidth should be between 10 GB/s and 10 TB/s.
            assert!(
                bw >= 10.0 && bw <= 10000.0,
                "Memory bandwidth {bw} GB/s out of range for {}",
                p.accelerator
            );
        }
    }
}

// ---------------------------------------------------------------------------
// System I/O: storage
// ---------------------------------------------------------------------------

#[test]
fn storage_detected_on_linux() {
    if !std::path::Path::new("/sys/block").exists() {
        eprintln!("SKIP: /sys/block not found");
        return;
    }
    let reg = AcceleratorRegistry::detect();
    let sio = reg.system_io();
    assert!(
        !sio.storage.is_empty(),
        "at least one storage device should be detected"
    );

    for dev in &sio.storage {
        assert!(!dev.name.is_empty());
        assert!(dev.bandwidth_gbps > 0.0);
    }
}

#[test]
fn storage_bandwidth_is_reasonable() {
    let reg = AcceleratorRegistry::detect();
    for dev in &reg.system_io().storage {
        assert!(
            dev.bandwidth_gbps >= 0.05 && dev.bandwidth_gbps <= 20.0,
            "Storage bandwidth {:.2} GB/s out of range for {}",
            dev.bandwidth_gbps,
            dev.name
        );
    }
}

#[test]
fn ingestion_estimate_returns_value() {
    let reg = AcceleratorRegistry::detect();
    let sio = reg.system_io();
    if sio.storage.is_empty() {
        eprintln!("SKIP: no storage detected");
        return;
    }
    let secs = sio.estimate_ingestion_secs(100 * 1024 * 1024 * 1024);
    assert!(secs.is_some(), "ingestion estimate should return a value");
    let secs = secs.unwrap();
    assert!(
        secs > 0.0 && secs < 100_000.0,
        "100 GB ingestion estimate should be reasonable: {secs:.1}s"
    );
}

// ---------------------------------------------------------------------------
// System I/O: interconnects
// ---------------------------------------------------------------------------

#[test]
fn interconnect_detection_does_not_panic() {
    // Just verify the detection doesn't crash, even without IB/NVLink.
    let reg = AcceleratorRegistry::detect();
    let _ = reg.system_io().interconnects.len();
    let _ = reg.system_io().total_interconnect_bandwidth_gbps();
}

// ---------------------------------------------------------------------------
// Full detection pipeline sanity
// ---------------------------------------------------------------------------

#[test]
fn full_detect_no_panics() {
    let reg = AcceleratorRegistry::detect();
    assert!(!reg.all_profiles().is_empty());
    let _ = reg.best_available();
    let _ = reg.total_memory();
    let _ = reg.total_accelerator_memory();
    let _ = reg.has_accelerator();
    let _ = reg.system_io();
    let _ = serde_json::to_string(&reg).unwrap();
}

#[test]
fn full_detect_json_roundtrip() {
    let reg = AcceleratorRegistry::detect();
    let json = serde_json::to_string(&reg).unwrap();
    let back: AcceleratorRegistry = serde_json::from_str(&json).unwrap();
    assert_eq!(reg.all_profiles().len(), back.all_profiles().len());
    assert_eq!(reg.schema_version(), back.schema_version());
}

#[test]
fn concurrent_detect_is_safe() {
    // Run detection from 4 threads simultaneously.
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| s.spawn(AcceleratorRegistry::detect))
            .collect();
        for h in handles {
            let reg = h.join().expect("detection should not panic");
            assert!(!reg.all_profiles().is_empty());
        }
    });
}
