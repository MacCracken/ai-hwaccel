//! Benchmarks for sharding planning and memory estimation.

use criterion::{criterion_group, criterion_main, Criterion};

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

fn bench_plan_sharding(c: &mut Criterion) {
    let registry = build_multi_gpu_registry();
    c.bench_function("plan_sharding 70B BF16", |b| {
        b.iter(|| registry.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16));
    });
}

fn bench_suggest_quantization(c: &mut Criterion) {
    let registry = build_multi_gpu_registry();
    c.bench_function("suggest_quantization 70B", |b| {
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

criterion_group!(
    benches,
    bench_plan_sharding,
    bench_suggest_quantization,
    bench_estimate_memory,
    bench_estimate_training_memory
);
criterion_main!(benches);
