//! Benchmarks for quantization operations and memory estimation.

use criterion::{Criterion, criterion_group, criterion_main};

use ai_hwaccel::*;

fn bench_bits_per_param(c: &mut Criterion) {
    let levels = [
        QuantizationLevel::None,
        QuantizationLevel::Float16,
        QuantizationLevel::BFloat16,
        QuantizationLevel::Int8,
        QuantizationLevel::Int4,
    ];
    c.bench_function("bits_per_param_all_levels", |b| {
        b.iter(|| {
            for q in &levels {
                let _ = q.bits_per_param();
            }
        });
    });
}

fn bench_memory_reduction_factor(c: &mut Criterion) {
    let levels = [
        QuantizationLevel::None,
        QuantizationLevel::Float16,
        QuantizationLevel::BFloat16,
        QuantizationLevel::Int8,
        QuantizationLevel::Int4,
    ];
    c.bench_function("memory_reduction_factor_all_levels", |b| {
        b.iter(|| {
            for q in &levels {
                let _ = q.memory_reduction_factor();
            }
        });
    });
}

fn bench_estimate_memory_all_quants(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_memory");

    let levels = [
        (QuantizationLevel::None, "fp32"),
        (QuantizationLevel::Float16, "fp16"),
        (QuantizationLevel::BFloat16, "bf16"),
        (QuantizationLevel::Int8, "int8"),
        (QuantizationLevel::Int4, "int4"),
    ];

    for (quant, name) in &levels {
        group.bench_function(format!("70B_{}", name), |b| {
            b.iter(|| AcceleratorRegistry::estimate_memory(70_000_000_000, quant));
        });
    }

    group.finish();
}

fn bench_suggest_quantization_model_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("suggest_quantization");

    let registry = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024),
    ]);

    let sizes: &[(u64, &str)] = &[
        (1_000_000_000, "1B"),
        (7_000_000_000, "7B"),
        (13_000_000_000, "13B"),
        (70_000_000_000, "70B"),
        (405_000_000_000, "405B"),
    ];

    for (params, name) in sizes {
        group.bench_function(format!("{}_1gpu", name), |b| {
            b.iter(|| registry.suggest_quantization(*params));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bits_per_param,
    bench_memory_reduction_factor,
    bench_estimate_memory_all_quants,
    bench_suggest_quantization_model_sizes,
);
criterion_main!(benches);
