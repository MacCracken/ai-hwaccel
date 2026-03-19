//! DetectBuilder and convenience constructor tests.

use crate::*;

#[test]
fn builder_detect_returns_at_least_cpu() {
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
fn builder_selective_detection() {
    // Even with all backends disabled, CPU is always present
    let reg = DetectBuilder::none().with_cuda().detect();
    // At minimum CPU is present; CUDA may or may not be depending on system
    assert!(
        reg.all_profiles()
            .iter()
            .any(|p| matches!(p.accelerator, AcceleratorType::Cpu))
    );
}

#[test]
fn builder_roundtrip_from_registry() {
    let reg = AcceleratorRegistry::builder().detect();
    assert!(!reg.all_profiles().is_empty());
}

#[test]
fn warnings_are_accessible() {
    let reg = AcceleratorRegistry::detect();
    // Warnings may or may not exist; just verify the accessor works
    let _warnings = reg.warnings();
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

#[test]
fn convenience_cuda() {
    let p = AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024);
    assert!(matches!(p.accelerator, AcceleratorType::CudaGpu { device_id: 0 }));
    assert_eq!(p.memory_bytes, 24 * 1024 * 1024 * 1024);
    assert!(p.available);
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
}

#[test]
fn convenience_gaudi() {
    let p = AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi3);
    assert_eq!(p.memory_bytes, GaudiGeneration::Gaudi3.hbm_bytes());
}

#[test]
fn convenience_cpu() {
    let p = AcceleratorProfile::cpu(32 * 1024 * 1024 * 1024);
    assert_eq!(p.accelerator, AcceleratorType::Cpu);
    assert_eq!(p.memory_bytes, 32 * 1024 * 1024 * 1024);
}

#[test]
fn convenience_rocm() {
    let p = AcceleratorProfile::rocm(0, 16 * 1024 * 1024 * 1024);
    assert!(matches!(p.accelerator, AcceleratorType::RocmGpu { device_id: 0 }));
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
}
