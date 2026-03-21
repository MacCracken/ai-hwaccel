//! Tests for cost-aware planning.

use crate::QuantizationLevel;
use crate::cost::*;

#[test]
fn all_instances_not_empty() {
    let instances = all_instances();
    assert!(!instances.is_empty(), "pricing table should have instances");
}

#[test]
fn all_instances_have_valid_prices() {
    for inst in all_instances() {
        assert!(inst.price_per_hour > 0.0, "{} has zero price", inst.name);
        assert!(inst.gpu_count > 0, "{} has zero GPUs", inst.name);
        assert!(
            inst.total_gpu_memory_gb > 0,
            "{} has zero GPU memory",
            inst.name
        );
    }
}

#[test]
fn all_instances_cached_across_calls() {
    let a = all_instances();
    let b = all_instances();
    // Same pointer (OnceLock returns same &'static slice).
    assert!(std::ptr::eq(a, b));
}

#[test]
fn recommend_7b_bf16_finds_instances() {
    let recs = recommend_instance(7_000_000_000, &QuantizationLevel::BFloat16, None);
    assert!(
        !recs.is_empty(),
        "7B at BF16 should fit on at least one instance"
    );
    // Should be sorted by price.
    for pair in recs.windows(2) {
        assert!(pair[0].instance.price_per_hour <= pair[1].instance.price_per_hour);
    }
}

#[test]
fn recommend_70b_bf16_needs_multi_gpu() {
    let recs = recommend_instance(70_000_000_000, &QuantizationLevel::BFloat16, None);
    assert!(!recs.is_empty());
    // 70B at BF16 needs ~156 GB, so single-GPU instances should be filtered out.
    for rec in &recs {
        assert!(rec.instance.total_gpu_memory_gb as f64 >= 150.0);
    }
}

#[test]
fn recommend_huge_model_may_return_empty() {
    // 10 trillion params at FP32 — nothing can fit this.
    let recs = recommend_instance(10_000_000_000_000, &QuantizationLevel::None, None);
    // May be empty if no instance has enough memory.
    for rec in &recs {
        assert!(rec.memory_headroom_pct >= 0.0);
    }
}

#[test]
fn recommend_filter_by_provider() {
    let aws = recommend_instance(
        7_000_000_000,
        &QuantizationLevel::Int8,
        Some(CloudProvider::Aws),
    );
    for rec in &aws {
        assert_eq!(rec.instance.provider, "aws");
    }

    let gcp = recommend_instance(
        7_000_000_000,
        &QuantizationLevel::Int8,
        Some(CloudProvider::Gcp),
    );
    for rec in &gcp {
        assert_eq!(rec.instance.provider, "gcp");
    }
}

#[test]
fn cheapest_instance_returns_first() {
    let cheapest = cheapest_instance(7_000_000_000, &QuantizationLevel::Int8, None);
    let all = recommend_instance(7_000_000_000, &QuantizationLevel::Int8, None);
    if let Some(c) = &cheapest {
        assert_eq!(c.instance.name, all[0].instance.name);
    }
}

#[test]
fn headroom_percentage_correct() {
    let recs = recommend_instance(7_000_000_000, &QuantizationLevel::Int8, None);
    for rec in &recs {
        assert!(rec.memory_headroom_pct >= 0.0);
        assert!(rec.memory_headroom_pct <= 100.0);
    }
}
