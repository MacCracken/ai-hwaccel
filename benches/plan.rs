//! Benchmarks for sharding planning and memory estimation.

use criterion::{Criterion, criterion_group, criterion_main};

use ai_hwaccel::*;

fn build_multi_gpu_registry() -> AcceleratorRegistry {
    AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 80 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(1, 80 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(2, 80 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(3, 80 * 1024 * 1024 * 1024),
    ])
}

fn build_large_registry() -> AcceleratorRegistry {
    let mut profiles = vec![AcceleratorProfile::cpu(512 * 1024 * 1024 * 1024)];
    for i in 0..8 {
        profiles.push(AcceleratorProfile::cuda(i, 80 * 1024 * 1024 * 1024));
    }
    for i in 0..4 {
        profiles.push(AcceleratorProfile::tpu(i, 4, TpuVersion::V5p));
    }
    profiles.push(AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi3));
    AcceleratorRegistry::from_profiles(profiles)
}

fn bench_plan_sharding(c: &mut Criterion) {
    let registry = build_multi_gpu_registry();
    c.bench_function("plan_sharding 70B BF16 (4 GPU)", |b| {
        b.iter(|| registry.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16));
    });
}

fn bench_plan_sharding_large(c: &mut Criterion) {
    let registry = build_large_registry();
    c.bench_function("plan_sharding 70B BF16 (13 devices)", |b| {
        b.iter(|| registry.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16));
    });
}

fn bench_suggest_quantization(c: &mut Criterion) {
    let registry = build_multi_gpu_registry();
    c.bench_function("suggest_quantization 70B (4 GPU)", |b| {
        b.iter(|| registry.suggest_quantization(70_000_000_000));
    });
}

fn bench_suggest_quantization_large(c: &mut Criterion) {
    let registry = build_large_registry();
    c.bench_function("suggest_quantization 70B (13 devices)", |b| {
        b.iter(|| registry.suggest_quantization(70_000_000_000));
    });
}

fn bench_estimate_memory(c: &mut Criterion) {
    c.bench_function("estimate_memory 70B FP16", |b| {
        b.iter(|| {
            AcceleratorRegistry::estimate_memory(70_000_000_000, &QuantizationLevel::Float16)
        });
    });
}

fn bench_estimate_training_memory(c: &mut Criterion) {
    c.bench_function("estimate_training_memory 7B LoRA GPU", |b| {
        b.iter(|| estimate_training_memory(7000, TrainingMethod::LoRA, TrainingTarget::Gpu));
    });
}

fn bench_best_available(c: &mut Criterion) {
    let registry = build_large_registry();
    c.bench_function("best_available (13 devices)", |b| {
        b.iter(|| registry.best_available());
    });
}

fn bench_total_memory(c: &mut Criterion) {
    let registry = build_large_registry();
    c.bench_function("total_memory (13 devices)", |b| {
        b.iter(|| registry.total_memory());
    });
}

criterion_group!(
    benches,
    bench_plan_sharding,
    bench_plan_sharding_large,
    bench_suggest_quantization,
    bench_suggest_quantization_large,
    bench_estimate_memory,
    bench_estimate_training_memory,
    bench_best_available,
    bench_total_memory,
);
criterion_main!(benches);
