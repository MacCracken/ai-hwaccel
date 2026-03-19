//! Sharding plan tests.

use crate::*;

#[test]
fn plan_sharding_small_model_single_device() {
    let reg = AcceleratorRegistry::new();
    let plan = reg.plan_sharding(1_000_000_000, &QuantizationLevel::Int4);
    assert_eq!(plan.strategy, ShardingStrategy::None);
    assert_eq!(plan.shards.len(), 1);
}

#[test]
fn plan_sharding_tpu_tensor_parallel() {
    let mut reg = AcceleratorRegistry::new();
    // 4 separate TPU chips, each with 95 GB — model too large for any single chip
    for i in 0..4 {
        reg.add_profile(AcceleratorProfile {
            accelerator: AcceleratorType::Tpu {
                device_id: i,
                chip_count: 1,
                version: TpuVersion::V5p,
            },
            available: true,
            memory_bytes: 95 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        });
    }
    // 70B at BF16 = ~168 GB — doesn't fit on one 95GB chip, but fits on 4
    let plan = reg.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16);
    assert!(matches!(
        plan.strategy,
        ShardingStrategy::TensorParallel { num_devices: 4 }
    ));
}

#[test]
fn plan_sharding_multi_gpu_pipeline() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 8 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 1 },
        available: true,
        memory_bytes: 8 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    let plan = reg.plan_sharding(7_000_000_000, &QuantizationLevel::Float16);
    assert!(matches!(
        plan.strategy,
        ShardingStrategy::PipelineParallel { .. }
    ));
    assert_eq!(plan.shards.len(), 2);
}

#[test]
fn plan_sharding_cpu_fallback() {
    let reg = AcceleratorRegistry::new();
    let plan = reg.plan_sharding(70_000_000_000, &QuantizationLevel::None);
    assert_eq!(plan.shards[0].device, AcceleratorType::Cpu);
}

#[test]
fn model_shard_num_layers() {
    let shard = ModelShard {
        shard_id: 0,
        layer_range: (0, 31),
        device: AcceleratorType::Cpu,
        memory_bytes: 1024,
    };
    assert_eq!(shard.num_layers(), 32);
    assert!(shard.is_valid());
}

#[test]
fn model_shard_invalid_range() {
    let shard = ModelShard {
        shard_id: 0,
        layer_range: (10, 5),
        device: AcceleratorType::Cpu,
        memory_bytes: 0,
    };
    assert_eq!(shard.num_layers(), 0);
    assert!(!shard.is_valid());
}
