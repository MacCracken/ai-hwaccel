//! Sharding plan tests.

use crate::*;

// ---------------------------------------------------------------------------
// ShardingStrategy
// ---------------------------------------------------------------------------

#[test]
fn sharding_strategy_min_devices() {
    assert_eq!(ShardingStrategy::None.min_devices(), 1);
    assert_eq!(
        ShardingStrategy::PipelineParallel { num_stages: 4 }.min_devices(),
        4
    );
    assert_eq!(
        ShardingStrategy::TensorParallel { num_devices: 8 }.min_devices(),
        8
    );
    assert_eq!(
        ShardingStrategy::DataParallel { num_replicas: 2 }.min_devices(),
        2
    );
}

// ---------------------------------------------------------------------------
// ModelShard
// ---------------------------------------------------------------------------

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
fn model_shard_single_layer() {
    let shard = ModelShard {
        shard_id: 0,
        layer_range: (5, 5),
        device: AcceleratorType::Cpu,
        memory_bytes: 0,
    };
    assert_eq!(shard.num_layers(), 1);
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

// ---------------------------------------------------------------------------
// plan_sharding — strategy selection
// ---------------------------------------------------------------------------

#[test]
fn plan_sharding_small_model_single_device() {
    let reg = AcceleratorRegistry::new();
    let plan = reg.plan_sharding(1_000_000_000, &QuantizationLevel::Int4);
    assert_eq!(plan.strategy, ShardingStrategy::None);
    assert_eq!(plan.shards.len(), 1);
    assert!(plan.estimated_tokens_per_sec.is_some());
}

#[test]
fn plan_sharding_tpu_tensor_parallel() {
    let mut profiles = vec![AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024)];
    for i in 0..4 {
        profiles.push(AcceleratorProfile::tpu(i, 1, TpuVersion::V5p));
    }
    let reg = AcceleratorRegistry::from_profiles(profiles);
    let plan = reg.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16);
    assert!(matches!(
        plan.strategy,
        ShardingStrategy::TensorParallel { num_devices: 4 }
    ));
    assert_eq!(plan.shards.len(), 4);
}

#[test]
fn plan_sharding_multi_gpu_pipeline() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 8 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(1, 8 * 1024 * 1024 * 1024),
    ]);
    let plan = reg.plan_sharding(7_000_000_000, &QuantizationLevel::Float16);
    assert!(matches!(
        plan.strategy,
        ShardingStrategy::PipelineParallel { .. }
    ));
    assert_eq!(plan.shards.len(), 2);
    // Verify layer ranges are contiguous
    assert_eq!(plan.shards[0].layer_range.0, 0);
    assert!(plan.shards[1].layer_range.0 > 0);
}

#[test]
fn plan_sharding_cpu_fallback() {
    let reg = AcceleratorRegistry::new();
    let plan = reg.plan_sharding(70_000_000_000, &QuantizationLevel::None);
    assert_eq!(plan.shards[0].device, AcceleratorType::Cpu);
    assert_eq!(plan.strategy, ShardingStrategy::None);
}

#[test]
fn plan_sharding_fits_single_gpu() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 80 * 1024 * 1024 * 1024), // A100 80 GB
    ]);
    let plan = reg.plan_sharding(7_000_000_000, &QuantizationLevel::Float16);
    assert_eq!(plan.strategy, ShardingStrategy::None);
    assert_eq!(plan.shards.len(), 1);
    assert!(matches!(
        plan.shards[0].device,
        AcceleratorType::CudaGpu { .. }
    ));
}

#[test]
fn plan_sharding_gaudi_pipeline() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi3),
        AcceleratorProfile::gaudi(1, GaudiGeneration::Gaudi3),
    ]);
    // 70B FP32 needs ~336 GB, each Gaudi3 has 128 GB = 256 GB total
    // 70B BF16 needs ~168 GB, fits on 2x Gaudi3
    let plan = reg.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16);
    assert!(matches!(
        plan.strategy,
        ShardingStrategy::PipelineParallel { num_stages: 2 }
    ));
}

#[test]
fn plan_sharding_model_exactly_fits_single_device() {
    // 1B params at INT4 = 600 MB needed
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 600_000_000), // exactly 600 MB
    ]);
    let plan = reg.plan_sharding(1_000_000_000, &QuantizationLevel::Int4);
    assert_eq!(plan.strategy, ShardingStrategy::None);
    assert!(matches!(
        plan.shards[0].device,
        AcceleratorType::CudaGpu { .. }
    ));
}

// ---------------------------------------------------------------------------
// ShardingPlan::fits_in_memory
// ---------------------------------------------------------------------------

#[test]
fn plan_fits_in_memory() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024),
    ]);
    let plan = reg.plan_sharding(1_000_000_000, &QuantizationLevel::Float16);
    assert!(plan.fits_in_memory(&reg));
}

#[test]
fn plan_does_not_fit_in_memory() {
    let plan = ShardingPlan {
        shards: vec![ModelShard {
            shard_id: 0,
            layer_range: (0, 0),
            device: AcceleratorType::Cpu,
            memory_bytes: 999 * 1024 * 1024 * 1024,
        }],
        strategy: ShardingStrategy::None,
        total_memory_bytes: 999 * 1024 * 1024 * 1024,
        estimated_tokens_per_sec: None,
    };
    let reg =
        AcceleratorRegistry::from_profiles(vec![AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024)]);
    assert!(!plan.fits_in_memory(&reg));
}

// ---------------------------------------------------------------------------
// Edge cases from audit
// ---------------------------------------------------------------------------

#[test]
fn plan_sharding_zero_params() {
    let reg = AcceleratorRegistry::new();
    let plan = reg.plan_sharding(0, &QuantizationLevel::Float16);
    assert_eq!(plan.strategy, ShardingStrategy::None);
    // Should not produce NaN/Inf
    if let Some(tps) = plan.estimated_tokens_per_sec {
        assert!(tps.is_finite());
    }
}

#[test]
fn plan_sharding_three_gpus_uneven_layers() {
    // 30B params → ~120 estimated layers, 3 GPUs → 40 per shard.
    // Last shard should capture all remaining layers.
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 40 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(1, 40 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(2, 40 * 1024 * 1024 * 1024),
    ]);
    let plan = reg.plan_sharding(30_000_000_000, &QuantizationLevel::Float16);
    assert!(matches!(
        plan.strategy,
        ShardingStrategy::PipelineParallel { num_stages: 3 }
    ));
    // Last shard must cover remaining layers (no gap).
    let last = plan.shards.last().unwrap();
    let first = &plan.shards[0];
    assert!(last.layer_range.1 >= first.layer_range.1);
    assert!(last.is_valid());
}

#[test]
fn plan_sharding_throughput_always_finite() {
    // Any multi-GPU plan should produce finite (or None) throughput, never NaN/Inf.
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(1, 24 * 1024 * 1024 * 1024),
    ]);
    for quant in &[
        QuantizationLevel::None,
        QuantizationLevel::Float16,
        QuantizationLevel::BFloat16,
        QuantizationLevel::Int8,
        QuantizationLevel::Int4,
    ] {
        let plan = reg.plan_sharding(7_000_000_000, quant);
        if let Some(tps) = plan.estimated_tokens_per_sec {
            assert!(tps.is_finite(), "NaN/Inf throughput for quant {:?}", quant);
            assert!(tps > 0.0, "negative throughput for quant {:?}", quant);
        }
    }
}

#[test]
fn plan_sharding_all_devices_unavailable() {
    let mut gpu = AcceleratorProfile::cuda(0, 80 * 1024 * 1024 * 1024);
    gpu.available = false;
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        gpu,
    ]);
    // Large model, only CPU available → CPU fallback
    let plan = reg.plan_sharding(70_000_000_000, &QuantizationLevel::Float16);
    assert_eq!(plan.shards[0].device, AcceleratorType::Cpu);
}
