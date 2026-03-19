//! Property-based tests for planning and estimation functions.

use proptest::prelude::*;

use crate::*;

// ---------------------------------------------------------------------------
// estimate_memory: properties that must always hold
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn estimate_memory_monotonic_with_params(params in 1u64..1_000_000_000_000) {
        let small = AcceleratorRegistry::estimate_memory(params, &QuantizationLevel::Float16);
        let large = AcceleratorRegistry::estimate_memory(params + 1, &QuantizationLevel::Float16);
        prop_assert!(large >= small);
    }

    #[test]
    fn estimate_memory_decreases_with_quantization(params in 1_000_000u64..100_000_000_000) {
        let fp32 = AcceleratorRegistry::estimate_memory(params, &QuantizationLevel::None);
        let fp16 = AcceleratorRegistry::estimate_memory(params, &QuantizationLevel::Float16);
        let int8 = AcceleratorRegistry::estimate_memory(params, &QuantizationLevel::Int8);
        let int4 = AcceleratorRegistry::estimate_memory(params, &QuantizationLevel::Int4);
        prop_assert!(fp32 >= fp16);
        prop_assert!(fp16 >= int8);
        prop_assert!(int8 >= int4);
        prop_assert!(int4 > 0);
    }

    #[test]
    fn estimate_memory_never_zero(params in 1u64..1_000_000_000_000) {
        for q in [
            QuantizationLevel::None,
            QuantizationLevel::Float16,
            QuantizationLevel::BFloat16,
            QuantizationLevel::Int8,
            QuantizationLevel::Int4,
        ] {
            let est = AcceleratorRegistry::estimate_memory(params, &q);
            prop_assert!(est > 0, "estimate_memory({}, {:?}) = 0", params, q);
        }
    }
}

// ---------------------------------------------------------------------------
// suggest_quantization: always returns a valid level
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn suggest_quantization_always_valid(
        params in 1_000_000u64..1_000_000_000_000,
        gpu_mem_gb in 1u64..256,
    ) {
        let reg = AcceleratorRegistry::from_profiles(vec![
            AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
            AcceleratorProfile::cuda(0, gpu_mem_gb * 1024 * 1024 * 1024),
        ]);
        let q = reg.suggest_quantization(params);
        prop_assert!(q.bits_per_param() <= 32);
        prop_assert!(q.bits_per_param() >= 4);
    }
}

// ---------------------------------------------------------------------------
// plan_sharding: always produces a plan
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn plan_sharding_always_produces_shards(
        params in 1_000_000u64..500_000_000_000,
        num_gpus in 0u32..8,
        gpu_mem_gb in 4u64..256,
    ) {
        let mut profiles = vec![AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024)];
        for i in 0..num_gpus {
            profiles.push(AcceleratorProfile::cuda(i, gpu_mem_gb * 1024 * 1024 * 1024));
        }
        let reg = AcceleratorRegistry::from_profiles(profiles);
        let plan = reg.plan_sharding(params, &QuantizationLevel::Float16);
        prop_assert!(!plan.shards.is_empty(), "plan should always have at least one shard");
        prop_assert!(plan.total_memory_bytes > 0);
    }

    #[test]
    fn plan_sharding_single_device_when_fits(
        params in 1_000_000u64..1_000_000_000,
    ) {
        // 1B params at INT4 = 600 MB. Give GPU 100 GB — should always fit.
        let reg = AcceleratorRegistry::from_profiles(vec![
            AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
            AcceleratorProfile::cuda(0, 100 * 1024 * 1024 * 1024),
        ]);
        let plan = reg.plan_sharding(params, &QuantizationLevel::Int4);
        prop_assert_eq!(plan.strategy, ShardingStrategy::None);
        prop_assert_eq!(plan.shards.len(), 1);
    }
}

// ---------------------------------------------------------------------------
// estimate_training_memory: components sum to total
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn training_memory_components_sum(
        params_m in 100u64..100_000,
    ) {
        let methods = [
            TrainingMethod::FullFineTune,
            TrainingMethod::LoRA,
            TrainingMethod::QLoRA { bits: 4 },
            TrainingMethod::Prefix,
            TrainingMethod::DPO,
            TrainingMethod::RLHF,
            TrainingMethod::Distillation,
        ];
        let targets = [
            TrainingTarget::Gpu,
            TrainingTarget::Tpu,
            TrainingTarget::Gaudi,
            TrainingTarget::Cpu,
        ];
        for method in &methods {
            for target in &targets {
                let est = estimate_training_memory(params_m, *method, *target);
                let sum = est.model_gb + est.optimizer_gb + est.activation_gb;
                prop_assert!(
                    (sum - est.total_gb).abs() < 0.01,
                    "{:?}/{:?}: sum={} total={}",
                    method, target, sum, est.total_gb
                );
                prop_assert!(est.total_gb > 0.0);
            }
        }
    }
}
