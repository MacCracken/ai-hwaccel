//! AcceleratorRegistry query and memory estimation tests.

use crate::*;

#[test]
fn registry_new_has_cpu() {
    let reg = AcceleratorRegistry::new();
    assert_eq!(reg.all_profiles().len(), 1);
    assert_eq!(reg.all_profiles()[0].accelerator, AcceleratorType::Cpu);
}

#[test]
fn registry_best_available_cpu_only() {
    let reg = AcceleratorRegistry::new();
    let best = reg.best_available().unwrap();
    assert_eq!(best.accelerator, AcceleratorType::Cpu);
}

#[test]
fn registry_best_prefers_tpu_v5p_over_cuda() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 4,
            version: TpuVersion::V5p,
        },
        available: true,
        memory_bytes: 380 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    assert!(matches!(
        reg.best_available().unwrap().accelerator,
        AcceleratorType::Tpu { .. }
    ));
}

#[test]
fn registry_best_prefers_cuda_over_tpu_v5e() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 1,
            version: TpuVersion::V5e,
        },
        available: true,
        memory_bytes: 16 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    assert!(matches!(
        reg.best_available().unwrap().accelerator,
        AcceleratorType::CudaGpu { .. }
    ));
}

#[test]
fn registry_total_accelerator_memory() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 8 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 1,
            version: TpuVersion::V4,
        },
        available: true,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    // GPU 8 + TPU 32 = 40 GiB (CPU excluded)
    assert_eq!(reg.total_accelerator_memory(), 40 * 1024 * 1024 * 1024);
}

#[test]
fn registry_by_family() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 8 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::IntelNpu,
        available: true,
        memory_bytes: 2 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    assert_eq!(reg.by_family(AcceleratorFamily::Gpu).len(), 1);
    assert_eq!(reg.by_family(AcceleratorFamily::Npu).len(), 1);
    assert_eq!(reg.by_family(AcceleratorFamily::Tpu).len(), 0);
}

#[test]
fn estimate_memory_fp32() {
    let est = AcceleratorRegistry::estimate_memory(1_000_000_000, &QuantizationLevel::None);
    assert_eq!(est, 4_800_000_000); // 4 bytes * 1B + 20%
}

#[test]
fn estimate_memory_int4() {
    let est = AcceleratorRegistry::estimate_memory(1_000_000_000, &QuantizationLevel::Int4);
    assert_eq!(est, 600_000_000); // 0.5 bytes * 1B + 20%
}

#[test]
fn suggest_quantization_tpu_prefers_bf16() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 4,
            version: TpuVersion::V5p,
        },
        available: true,
        memory_bytes: 380 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    assert_eq!(
        reg.suggest_quantization(7_000_000_000),
        QuantizationLevel::BFloat16
    );
}

#[test]
fn suggest_quantization_gpu_prefers_fp16() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    assert_eq!(
        reg.suggest_quantization(7_000_000_000),
        QuantizationLevel::Float16
    );
}

#[test]
fn detect_returns_at_least_cpu() {
    let reg = AcceleratorRegistry::detect();
    assert!(!reg.all_profiles().is_empty());
    assert!(
        reg.all_profiles()
            .iter()
            .any(|p| matches!(p.accelerator, AcceleratorType::Cpu))
    );
}
