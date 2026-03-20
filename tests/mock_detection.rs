//! Mock detection tests using fake sysfs trees and CLI tool scripts.
//!
//! Tests that use Unix symlinks are gated behind `#[cfg(unix)]` so the
//! test file compiles and runs on Windows too.

use std::fs;
use std::path::Path;

use tempfile::TempDir;

use ai_hwaccel::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mkdir_p(path: &Path) {
    fs::create_dir_all(path).unwrap();
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        mkdir_p(parent);
    }
    fs::write(path, content).unwrap();
}

// ---------------------------------------------------------------------------
// Cross-platform tests (no symlinks needed)
// ---------------------------------------------------------------------------

#[test]
fn mock_multi_device_registry() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 80 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(1, 80 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(2, 80 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(3, 80 * 1024 * 1024 * 1024),
        AcceleratorProfile::tpu(0, 4, TpuVersion::V5p),
        AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi3),
    ]);

    assert_eq!(reg.all_profiles().len(), 7);
    assert_eq!(reg.by_family(AcceleratorFamily::Gpu).len(), 4);
    assert_eq!(reg.by_family(AcceleratorFamily::Tpu).len(), 1);
    assert_eq!(reg.by_family(AcceleratorFamily::AiAsic).len(), 1);

    assert!(matches!(
        reg.best_available().unwrap().accelerator,
        AcceleratorType::Tpu { .. }
    ));

    let plan = reg.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16);
    assert_eq!(plan.strategy, ShardingStrategy::None);
    assert!(matches!(plan.shards[0].device, AcceleratorType::Tpu { .. }));
}

#[test]
fn mock_intel_npu_sysfs() {
    let tmp = TempDir::new().unwrap();
    let npu_path = tmp.path().join("sys/class/misc/intel_npu");
    mkdir_p(&npu_path);
    assert!(npu_path.exists());
}

#[test]
fn mock_qualcomm_sysfs() {
    let tmp = TempDir::new().unwrap();
    let qaic_path = tmp.path().join("sys/class/qaic");
    mkdir_p(&qaic_path);
    assert!(qaic_path.exists());
}

#[test]
fn mock_proc_meminfo() {
    let tmp = TempDir::new().unwrap();
    let meminfo = tmp.path().join("meminfo");
    write_file(
        &meminfo,
        "MemTotal:       65536000 kB\nMemFree:        32768000 kB\n",
    );

    let content = fs::read_to_string(&meminfo).unwrap();
    let mut mem_kb = 0u64;
    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            mem_kb = parts[1].parse().unwrap();
        }
    }
    assert_eq!(mem_kb * 1024, 65536000 * 1024);
}

#[test]
fn registry_json_has_schema_version() {
    let reg = AcceleratorRegistry::detect();
    let json = serde_json::to_string(&reg).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["schema_version"], SCHEMA_VERSION);
}

#[test]
fn registry_schema_version_accessor() {
    let reg = AcceleratorRegistry::new();
    assert_eq!(reg.schema_version(), SCHEMA_VERSION);
}

#[test]
fn validate_device_id_via_public_api() {
    let p = AcceleratorProfile::cuda(0, 1024);
    assert!(matches!(
        p.accelerator,
        AcceleratorType::CudaGpu { device_id: 0 }
    ));

    let p = AcceleratorProfile::cuda(1024, 1024);
    assert!(matches!(
        p.accelerator,
        AcceleratorType::CudaGpu { device_id: 1024 }
    ));
}

#[test]
fn serde_rejects_unknown_fields() {
    let json = r#"{"schema_version":1,"profiles":[],"unknown_field":"bad"}"#;
    let result = serde_json::from_str::<AcceleratorRegistry>(json);
    assert!(result.is_err(), "should reject unknown fields");
}

#[test]
fn serde_rejects_unknown_profile_fields() {
    let json = r#"{"accelerator":"Cpu","available":true,"memory_bytes":0,"compute_capability":null,"driver_version":null,"extra":true}"#;
    let result = serde_json::from_str::<AcceleratorProfile>(json);
    assert!(result.is_err(), "should reject unknown profile fields");
}

// ---------------------------------------------------------------------------
// Unix-only tests (require symlinks for fake sysfs trees)
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod unix {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn mock_rocm_sysfs_tree() {
        let tmp = TempDir::new().unwrap();
        let drm = tmp.path().join("sys/class/drm/card0");
        let device_dir = drm.join("device");
        mkdir_p(&device_dir);

        let drivers_dir = tmp.path().join("drivers/amdgpu");
        mkdir_p(&drivers_dir);
        symlink(&drivers_dir, device_dir.join("driver")).unwrap();

        write_file(&device_dir.join("mem_info_vram_total"), "17179869184\n");

        let driver_link = device_dir.join("driver");
        let target = fs::read_link(&driver_link).unwrap();
        let driver_name = target.file_name().unwrap().to_string_lossy();
        assert_eq!(driver_name, "amdgpu");

        let vram_str = fs::read_to_string(device_dir.join("mem_info_vram_total")).unwrap();
        let vram: u64 = vram_str.trim().parse().unwrap();
        assert_eq!(vram, 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn mock_amd_xdna_sysfs() {
        let tmp = TempDir::new().unwrap();
        let accel_dir = tmp.path().join("sys/class/accel/accel0");
        let device_dir = accel_dir.join("device");
        mkdir_p(&device_dir);

        let drivers_dir = tmp.path().join("drivers/amdxdna");
        mkdir_p(&drivers_dir);
        symlink(&drivers_dir, device_dir.join("driver")).unwrap();

        let driver_link = device_dir.join("driver");
        let target = fs::read_link(&driver_link).unwrap();
        assert_eq!(target.file_name().unwrap().to_string_lossy(), "amdxdna");
    }
}

// ---------------------------------------------------------------------------
// suggest_quantization semantic correctness
// ---------------------------------------------------------------------------

#[test]
fn suggest_quantization_qualcomm_does_not_return_bf16() {
    // Qualcomm AI 100 does NOT support BF16. suggest_quantization must
    // not return BF16 even if an AI ASIC is present.
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile {
            accelerator: AcceleratorType::QualcommAi100 { device_id: 0 },
            available: true,
            memory_bytes: 32 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
            memory_bandwidth_gbps: None,
            memory_used_bytes: None,
            memory_free_bytes: None,
            pcie_bandwidth_gbps: None,
            numa_node: None,
        },
    ]);
    let q = reg.suggest_quantization(7_000_000_000);
    assert_ne!(
        q,
        QuantizationLevel::BFloat16,
        "Qualcomm should not get BF16"
    );
}

#[test]
fn suggest_quantization_huge_model_falls_to_int4() {
    // 500B model on 16 GB CPU: nothing fits at FP16 (~1.2 TB needed).
    // Should fall back to smallest quantisation.
    let reg =
        AcceleratorRegistry::from_profiles(vec![AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024)]);
    let q = reg.suggest_quantization(500_000_000_000);
    assert_eq!(q, QuantizationLevel::Int4);
}

#[test]
fn suggest_quantization_gaudi_gets_bf16() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi3),
    ]);
    let q = reg.suggest_quantization(7_000_000_000);
    assert_eq!(q, QuantizationLevel::BFloat16);
}
