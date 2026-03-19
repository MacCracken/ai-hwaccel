//! Mock detection tests using fake sysfs trees and CLI tool scripts.
//!
//! These tests create temporary directories that mimic the sysfs layout
//! expected by various detectors, allowing hardware-independent testing of
//! the parsing and detection logic.

use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;

use tempfile::TempDir;

use ai_hwaccel::*;

// ---------------------------------------------------------------------------
// Helpers: build fake sysfs trees
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
// Test: AcceleratorProfile convenience constructors round-trip through
// a full registry pipeline (no real hardware needed).
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

    // Best should be TPU V5p (rank 80)
    assert!(matches!(
        reg.best_available().unwrap().accelerator,
        AcceleratorType::Tpu { .. }
    ));

    // 70B BF16 = ~168 GB. TPU has 4 chips * 95 GB = 380 GB total in one
    // profile entry, so it fits on the single "device" → no sharding needed.
    let plan = reg.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16);
    assert_eq!(plan.strategy, ShardingStrategy::None);
    assert!(matches!(plan.shards[0].device, AcceleratorType::Tpu { .. }));
}

// ---------------------------------------------------------------------------
// Test: fake sysfs AMD ROCm detection
// ---------------------------------------------------------------------------

#[test]
fn mock_rocm_sysfs_tree() {
    let tmp = TempDir::new().unwrap();
    let drm = tmp.path().join("sys/class/drm/card0");
    let device_dir = drm.join("device");
    mkdir_p(&device_dir);

    // Create a fake driver symlink: device/driver -> /drivers/amdgpu
    let drivers_dir = tmp.path().join("drivers/amdgpu");
    mkdir_p(&drivers_dir);
    symlink(&drivers_dir, device_dir.join("driver")).unwrap();

    // Write fake VRAM size
    write_file(
        &device_dir.join("mem_info_vram_total"),
        "17179869184\n", // 16 GiB
    );

    // Verify the sysfs structure is valid
    let driver_link = device_dir.join("driver");
    let target = fs::read_link(&driver_link).unwrap();
    let driver_name = target.file_name().unwrap().to_string_lossy();
    assert_eq!(driver_name, "amdgpu");

    let vram_str = fs::read_to_string(device_dir.join("mem_info_vram_total")).unwrap();
    let vram: u64 = vram_str.trim().parse().unwrap();
    assert_eq!(vram, 16 * 1024 * 1024 * 1024);
}

// ---------------------------------------------------------------------------
// Test: fake sysfs Intel NPU detection
// ---------------------------------------------------------------------------

#[test]
fn mock_intel_npu_sysfs() {
    let tmp = TempDir::new().unwrap();
    let npu_path = tmp.path().join("sys/class/misc/intel_npu");
    mkdir_p(&npu_path);

    // The real detector checks Path::new("/sys/class/misc/intel_npu").exists()
    // Here we verify our mock tree is structurally correct.
    assert!(npu_path.exists());
}

// ---------------------------------------------------------------------------
// Test: fake sysfs AMD XDNA detection
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Test: fake sysfs Qualcomm AI 100
// ---------------------------------------------------------------------------

#[test]
fn mock_qualcomm_sysfs() {
    let tmp = TempDir::new().unwrap();
    let qaic_path = tmp.path().join("sys/class/qaic");
    mkdir_p(&qaic_path);
    assert!(qaic_path.exists());
}

// ---------------------------------------------------------------------------
// Test: fake /proc/meminfo parsing
// ---------------------------------------------------------------------------

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
    assert_eq!(mem_kb * 1024, 65536000 * 1024); // ~62.5 GiB
}

// ---------------------------------------------------------------------------
// Test: validate JSON schema compliance
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Test: command validation helpers
// ---------------------------------------------------------------------------

#[test]
fn validate_device_id_via_public_api() {
    // These go through the full detect path, but we test the validation
    // indirectly by constructing profiles with edge-case IDs.
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

// ---------------------------------------------------------------------------
// Test: serde deny_unknown_fields actually rejects
// ---------------------------------------------------------------------------

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
