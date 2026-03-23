//! Benchmarks for cost-aware instance recommendation.

use criterion::{Criterion, criterion_group, criterion_main};

use ai_hwaccel::QuantizationLevel;
use ai_hwaccel::cost::{CloudProvider, all_instances, cheapest_instance, recommend_instance};

fn bench_load_instances(c: &mut Criterion) {
    c.bench_function("load_pricing_table", |b| {
        b.iter(all_instances);
    });
}

fn bench_recommend_7b(c: &mut Criterion) {
    let mut group = c.benchmark_group("recommend_instance");

    group.bench_function("7B_bf16_all", |b| {
        b.iter(|| recommend_instance(7_000_000_000, &QuantizationLevel::BFloat16, None));
    });

    group.bench_function("7B_int8_aws", |b| {
        b.iter(|| {
            recommend_instance(
                7_000_000_000,
                &QuantizationLevel::Int8,
                Some(CloudProvider::Aws),
            )
        });
    });

    group.bench_function("70B_bf16_all", |b| {
        b.iter(|| recommend_instance(70_000_000_000, &QuantizationLevel::BFloat16, None));
    });

    group.bench_function("70B_int4_gcp", |b| {
        b.iter(|| {
            recommend_instance(
                70_000_000_000,
                &QuantizationLevel::Int4,
                Some(CloudProvider::Gcp),
            )
        });
    });

    group.finish();
}

fn bench_cheapest_instance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cheapest_instance");

    group.bench_function("7B_bf16", |b| {
        b.iter(|| cheapest_instance(7_000_000_000, &QuantizationLevel::BFloat16, None));
    });

    group.bench_function("70B_bf16", |b| {
        b.iter(|| cheapest_instance(70_000_000_000, &QuantizationLevel::BFloat16, None));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_load_instances,
    bench_recommend_7b,
    bench_cheapest_instance,
);
criterion_main!(benches);
