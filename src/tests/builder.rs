//! DetectBuilder, convenience constructor, and error type tests.

use crate::*;

// ---------------------------------------------------------------------------
// DetectBuilder
// ---------------------------------------------------------------------------

#[test]
fn builder_default_returns_at_least_cpu() {
    let reg = AcceleratorRegistry::builder().detect();
    assert!(!reg.all_profiles().is_empty());
    assert!(
        reg.all_profiles()
            .iter()
            .any(|p| matches!(p.accelerator, AcceleratorType::Cpu))
    );
}

#[test]
fn builder_none_returns_only_cpu() {
    let reg = DetectBuilder::none().detect();
    assert_eq!(reg.all_profiles().len(), 1);
    assert_eq!(reg.all_profiles()[0].accelerator, AcceleratorType::Cpu);
}

#[test]
fn builder_none_with_one_backend() {
    let reg = DetectBuilder::none().with_cuda().detect();
    assert!(
        reg.all_profiles()
            .iter()
            .any(|p| matches!(p.accelerator, AcceleratorType::Cpu))
    );
}

#[test]
fn builder_without_disables_backend() {
    // Disable all backends one by one — should still have CPU
    let reg = AcceleratorRegistry::builder()
        .without_cuda()
        .without_rocm()
        .without_apple()
        .without_vulkan()
        .without_intel_npu()
        .without_amd_xdna()
        .without_tpu()
        .without_gaudi()
        .without_aws_neuron()
        .without_intel_oneapi()
        .without_qualcomm()
        .detect();
    // With all backends disabled, only CPU remains
    assert_eq!(reg.all_profiles().len(), 1);
    assert_eq!(reg.all_profiles()[0].accelerator, AcceleratorType::Cpu);
}

#[test]
fn builder_chaining_with_and_without() {
    // Enable just two
    let reg = DetectBuilder::none()
        .with_cuda()
        .with_rocm()
        .without_cuda() // then disable one
        .detect();
    // CPU always present; CUDA disabled, ROCm may or may not find hardware
    assert!(
        reg.all_profiles()
            .iter()
            .any(|p| matches!(p.accelerator, AcceleratorType::Cpu))
    );
}

#[test]
fn builder_from_registry_shortcut() {
    let reg = AcceleratorRegistry::builder().detect();
    assert!(!reg.all_profiles().is_empty());
}

#[test]
fn backend_all_constant() {
    assert_eq!(Backend::ALL.len(), 17);
}

#[test]
fn detect_builder_bitmask_none_has_zero_enabled() {
    let b = DetectBuilder::none();
    assert_eq!(b.enabled_count(), 0);
}

#[test]
fn detect_builder_bitmask_all_has_all_enabled() {
    let b = DetectBuilder::new();
    assert_eq!(b.enabled_count(), Backend::ALL.len());
}

#[test]
fn detect_builder_is_copy() {
    let a = DetectBuilder::none().with_cuda();
    let b = a; // Copy, not move
    let _c = a; // Still usable — proves Copy
    assert_eq!(b.enabled_count(), 1);
}

#[test]
fn detect_builder_with_without_roundtrip() {
    let b = DetectBuilder::none()
        .with(Backend::Cuda)
        .with(Backend::Rocm)
        .without(Backend::Cuda);
    assert_eq!(b.enabled_count(), 1);
    // ROCm should still be enabled
    assert!(b.backend_enabled(Backend::Rocm));
    assert!(!b.backend_enabled(Backend::Cuda));
}

// ---------------------------------------------------------------------------
// Warnings
// ---------------------------------------------------------------------------

#[test]
fn warnings_are_accessible() {
    let reg = AcceleratorRegistry::detect();
    let _warnings = reg.warnings();
}

#[test]
fn from_profiles_has_no_warnings() {
    let reg =
        AcceleratorRegistry::from_profiles(vec![AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024)]);
    assert!(reg.warnings().is_empty());
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

#[test]
fn convenience_cuda() {
    let p = AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024);
    assert!(matches!(
        p.accelerator,
        AcceleratorType::CudaGpu { device_id: 0 }
    ));
    assert_eq!(p.memory_bytes, 24 * 1024 * 1024 * 1024);
    assert!(p.available);
    assert!(p.compute_capability.is_none());
    assert!(p.driver_version.is_none());
}

#[test]
fn convenience_rocm() {
    let p = AcceleratorProfile::rocm(1, 16 * 1024 * 1024 * 1024);
    assert!(matches!(
        p.accelerator,
        AcceleratorType::RocmGpu { device_id: 1 }
    ));
    assert_eq!(p.memory_bytes, 16 * 1024 * 1024 * 1024);
}

#[test]
fn convenience_tpu() {
    let p = AcceleratorProfile::tpu(0, 4, TpuVersion::V5p);
    assert!(matches!(
        p.accelerator,
        AcceleratorType::Tpu {
            chip_count: 4,
            version: TpuVersion::V5p,
            ..
        }
    ));
    assert_eq!(p.memory_bytes, 4 * TpuVersion::V5p.hbm_per_chip_bytes());
    assert!(p.compute_capability.is_some());
}

#[test]
fn convenience_gaudi() {
    let p = AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi3);
    assert_eq!(p.memory_bytes, GaudiGeneration::Gaudi3.hbm_bytes());
    assert!(p.compute_capability.is_some());
}

#[test]
fn convenience_cpu() {
    let p = AcceleratorProfile::cpu(32 * 1024 * 1024 * 1024);
    assert_eq!(p.accelerator, AcceleratorType::Cpu);
    assert_eq!(p.memory_bytes, 32 * 1024 * 1024 * 1024);
}

// ---------------------------------------------------------------------------
// ShardingPlan Display
// ---------------------------------------------------------------------------

#[test]
fn sharding_plan_display_single_device() {
    let plan = ShardingPlan {
        shards: vec![ModelShard {
            shard_id: 0,
            layer_range: (0, 0),
            device: AcceleratorType::CudaGpu { device_id: 0 },
            memory_bytes: 14 * 1024 * 1024 * 1024,
        }],
        strategy: ShardingStrategy::None,
        total_memory_bytes: 14 * 1024 * 1024 * 1024,
        estimated_tokens_per_sec: Some(42.0),
    };
    let display = plan.to_string();
    assert!(display.contains("None"));
    assert!(display.contains("14.0 GB"));
    assert!(display.contains("42 tok/s"));
    assert!(display.contains("CUDA GPU"));
    // Single device: should show "Device:" not "Shards:"
    assert!(!display.contains("Shards:"));
    assert!(display.contains("Device:"));
}

#[test]
fn sharding_plan_display_multi_shard() {
    let plan = ShardingPlan {
        shards: vec![
            ModelShard {
                shard_id: 0,
                layer_range: (0, 15),
                device: AcceleratorType::CudaGpu { device_id: 0 },
                memory_bytes: 8 * 1024 * 1024 * 1024,
            },
            ModelShard {
                shard_id: 1,
                layer_range: (16, 31),
                device: AcceleratorType::CudaGpu { device_id: 1 },
                memory_bytes: 8 * 1024 * 1024 * 1024,
            },
        ],
        strategy: ShardingStrategy::PipelineParallel { num_stages: 2 },
        total_memory_bytes: 16 * 1024 * 1024 * 1024,
        estimated_tokens_per_sec: None,
    };
    let display = plan.to_string();
    assert!(display.contains("Pipeline Parallel"));
    assert!(display.contains("Shards:"));
    assert!(display.contains("[0]"));
    assert!(display.contains("[1]"));
    // No throughput line when None
    assert!(!display.contains("tok/s"));
}

#[test]
fn sharding_plan_display_empty() {
    let plan = ShardingPlan {
        shards: vec![],
        strategy: ShardingStrategy::DataParallel { num_replicas: 0 },
        total_memory_bytes: 0,
        estimated_tokens_per_sec: None,
    };
    let display = plan.to_string();
    assert!(display.contains("Data Parallel"));
}

// ---------------------------------------------------------------------------
// DetectionError is std::error::Error
// ---------------------------------------------------------------------------

#[test]
fn detection_error_is_std_error() {
    let e: Box<dyn std::error::Error> = Box::new(DetectionError::ToolNotFound {
        tool: "test".into(),
    });
    assert!(!e.to_string().is_empty());
}

// ---------------------------------------------------------------------------
// CachedRegistry
// ---------------------------------------------------------------------------

#[test]
fn cached_registry_debug() {
    let cache = CachedRegistry::new(std::time::Duration::from_secs(60));
    let debug = format!("{:?}", cache);
    assert!(debug.contains("CachedRegistry"));
    assert!(debug.contains("ttl"));
}

#[test]
fn cached_registry_returns_same_on_second_call() {
    let cache = CachedRegistry::new(std::time::Duration::from_secs(300));
    let first = cache.get();
    let second = cache.get();
    assert_eq!(first.all_profiles().len(), second.all_profiles().len());
}

#[test]
fn cached_registry_invalidate_forces_redetect() {
    let cache = CachedRegistry::new(std::time::Duration::from_secs(300));
    let _first = cache.get();
    cache.invalidate();
    let after = cache.get(); // should re-detect
    assert!(!after.all_profiles().is_empty());
}

#[test]
fn cached_registry_ttl_accessor() {
    let cache = CachedRegistry::new(std::time::Duration::from_secs(42));
    assert_eq!(cache.ttl(), std::time::Duration::from_secs(42));
}

// ---------------------------------------------------------------------------
// Overflow safety: large chip counts don't panic
// ---------------------------------------------------------------------------

#[test]
fn tpu_large_chip_count_no_overflow_panic() {
    // u32::MAX chips * 95 GiB would overflow u64 without saturating_mul.
    let p = AcceleratorProfile::tpu(0, u32::MAX, TpuVersion::V5p);
    // Should saturate to u64::MAX, not panic.
    assert!(p.memory_bytes > 0);
}
