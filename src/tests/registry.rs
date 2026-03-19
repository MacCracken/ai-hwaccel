//! AcceleratorRegistry query and memory estimation tests.

use crate::*;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn registry_new_has_cpu() {
    let reg = AcceleratorRegistry::new();
    assert_eq!(reg.all_profiles().len(), 1);
    assert_eq!(reg.all_profiles()[0].accelerator, AcceleratorType::Cpu);
    assert!(reg.warnings().is_empty());
}

#[test]
fn registry_from_profiles() {
    let profiles = vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024),
    ];
    let reg = AcceleratorRegistry::from_profiles(profiles);
    assert_eq!(reg.all_profiles().len(), 2);
    assert!(reg.warnings().is_empty());
}

// ---------------------------------------------------------------------------
// best_available
// ---------------------------------------------------------------------------

#[test]
fn registry_best_available_cpu_only() {
    let reg = AcceleratorRegistry::new();
    let best = reg.best_available().unwrap();
    assert_eq!(best.accelerator, AcceleratorType::Cpu);
}

#[test]
fn registry_best_prefers_tpu_v5p_over_cuda() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024),
        AcceleratorProfile::tpu(0, 4, TpuVersion::V5p),
    ]);
    assert!(matches!(
        reg.best_available().unwrap().accelerator,
        AcceleratorType::Tpu { .. }
    ));
}

#[test]
fn registry_best_prefers_cuda_over_tpu_v5e() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024),
        AcceleratorProfile::tpu(0, 1, TpuVersion::V5e),
    ]);
    assert!(matches!(
        reg.best_available().unwrap().accelerator,
        AcceleratorType::CudaGpu { .. }
    ));
}

#[test]
fn registry_best_skips_unavailable() {
    let mut fast_but_down = AcceleratorProfile::tpu(0, 4, TpuVersion::V5p);
    fast_but_down.available = false;
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        fast_but_down,
        AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024),
    ]);
    assert!(matches!(
        reg.best_available().unwrap().accelerator,
        AcceleratorType::CudaGpu { .. }
    ));
}

// ---------------------------------------------------------------------------
// Memory queries
// ---------------------------------------------------------------------------

#[test]
fn registry_total_accelerator_memory() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 8 * 1024 * 1024 * 1024),
        AcceleratorProfile::tpu(0, 1, TpuVersion::V4),
    ]);
    // GPU 8 + TPU 32 = 40 GiB (CPU excluded)
    assert_eq!(reg.total_accelerator_memory(), 40 * 1024 * 1024 * 1024);
}

#[test]
fn registry_total_memory_includes_cpu() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 8 * 1024 * 1024 * 1024),
    ]);
    assert_eq!(reg.total_memory(), 72 * 1024 * 1024 * 1024);
}

#[test]
fn registry_has_accelerator() {
    let cpu_only = AcceleratorRegistry::from_profiles(vec![AcceleratorProfile::cpu(
        16 * 1024 * 1024 * 1024,
    )]);
    assert!(!cpu_only.has_accelerator());

    let with_gpu = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 8 * 1024 * 1024 * 1024),
    ]);
    assert!(with_gpu.has_accelerator());
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

#[test]
fn registry_by_family() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 8 * 1024 * 1024 * 1024),
        AcceleratorProfile {
            accelerator: AcceleratorType::IntelNpu,
            available: true,
            memory_bytes: 2 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        },
    ]);
    assert_eq!(reg.by_family(AcceleratorFamily::Gpu).len(), 1);
    assert_eq!(reg.by_family(AcceleratorFamily::Npu).len(), 1);
    assert_eq!(reg.by_family(AcceleratorFamily::Tpu).len(), 0);
    assert_eq!(reg.by_family(AcceleratorFamily::Cpu).len(), 1);
}

#[test]
fn registry_satisfying() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 8 * 1024 * 1024 * 1024),
        AcceleratorProfile::tpu(0, 4, TpuVersion::V5p),
        AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi3),
    ]);
    assert_eq!(reg.satisfying(&AcceleratorRequirement::Gpu).len(), 1);
    assert_eq!(reg.satisfying(&AcceleratorRequirement::GpuOrTpu).len(), 2);
    assert_eq!(reg.satisfying(&AcceleratorRequirement::AnyAccelerator).len(), 3);
    assert_eq!(reg.satisfying(&AcceleratorRequirement::None).len(), 4);
    assert_eq!(reg.satisfying(&AcceleratorRequirement::Gaudi).len(), 1);
    assert_eq!(reg.satisfying(&AcceleratorRequirement::Tpu { min_chips: 4 }).len(), 1);
    assert_eq!(reg.satisfying(&AcceleratorRequirement::Tpu { min_chips: 8 }).len(), 0);
}

#[test]
fn registry_available_excludes_unavailable() {
    let mut down = AcceleratorProfile::cuda(0, 8 * 1024 * 1024 * 1024);
    down.available = false;
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        down,
    ]);
    assert_eq!(reg.available().len(), 1);
    assert_eq!(reg.all_profiles().len(), 2);
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

#[test]
fn estimate_memory_fp32() {
    let est = AcceleratorRegistry::estimate_memory(1_000_000_000, &QuantizationLevel::None);
    assert_eq!(est, 4_800_000_000); // 4 bytes * 1B + 20%
}

#[test]
fn estimate_memory_fp16() {
    let est = AcceleratorRegistry::estimate_memory(1_000_000_000, &QuantizationLevel::Float16);
    assert_eq!(est, 2_400_000_000);
}

#[test]
fn estimate_memory_bf16() {
    let est = AcceleratorRegistry::estimate_memory(1_000_000_000, &QuantizationLevel::BFloat16);
    assert_eq!(est, 2_400_000_000);
}

#[test]
fn estimate_memory_int8() {
    let est = AcceleratorRegistry::estimate_memory(1_000_000_000, &QuantizationLevel::Int8);
    assert_eq!(est, 1_200_000_000);
}

#[test]
fn estimate_memory_int4() {
    let est = AcceleratorRegistry::estimate_memory(1_000_000_000, &QuantizationLevel::Int4);
    assert_eq!(est, 600_000_000);
}

// ---------------------------------------------------------------------------
// suggest_quantization
// ---------------------------------------------------------------------------

#[test]
fn suggest_quantization_tpu_prefers_bf16() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::tpu(0, 4, TpuVersion::V5p),
    ]);
    assert_eq!(
        reg.suggest_quantization(7_000_000_000),
        QuantizationLevel::BFloat16
    );
}

#[test]
fn suggest_quantization_gpu_prefers_fp16() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024),
    ]);
    assert_eq!(
        reg.suggest_quantization(7_000_000_000),
        QuantizationLevel::Float16
    );
}

#[test]
fn suggest_quantization_gaudi_prefers_bf16() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi3),
    ]);
    assert_eq!(
        reg.suggest_quantization(7_000_000_000),
        QuantizationLevel::BFloat16
    );
}

#[test]
fn suggest_quantization_npu_prefers_int8() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile {
            accelerator: AcceleratorType::IntelNpu,
            available: true,
            memory_bytes: 16 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        },
    ]);
    assert_eq!(
        reg.suggest_quantization(7_000_000_000),
        QuantizationLevel::Int8
    );
}

#[test]
fn suggest_quantization_cpu_only_fallback() {
    let reg = AcceleratorRegistry::from_profiles(vec![AcceleratorProfile::cpu(
        16 * 1024 * 1024 * 1024,
    )]);
    // CPU-only falls back to FP16
    assert_eq!(
        reg.suggest_quantization(7_000_000_000),
        QuantizationLevel::Float16
    );
}

#[test]
fn suggest_quantization_gpu_too_small_drops_to_int8() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 4 * 1024 * 1024 * 1024), // only 4 GB
    ]);
    // 7B at FP16 needs ~16.8 GB, won't fit in 4 GB
    // 7B at INT8 needs ~8.4 GB, still won't fit
    // 7B at INT4 needs ~4.2 GB, still won't fit
    // Falls back to CPU FP16
    let q = reg.suggest_quantization(7_000_000_000);
    assert!(
        q == QuantizationLevel::Int4 || q == QuantizationLevel::Float16,
        "got {:?}",
        q
    );
}

// ---------------------------------------------------------------------------
// detect() smoke test
// ---------------------------------------------------------------------------

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
