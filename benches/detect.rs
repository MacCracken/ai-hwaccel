//! Benchmarks for hardware detection.

use criterion::{criterion_group, criterion_main, Criterion};

use ai_hwaccel::{AcceleratorRegistry, DetectBuilder};

fn bench_detect_all(c: &mut Criterion) {
    c.bench_function("detect_all", |b| {
        b.iter(AcceleratorRegistry::detect);
    });
}

fn bench_detect_none(c: &mut Criterion) {
    c.bench_function("detect_none (CPU only)", |b| {
        b.iter(|| DetectBuilder::none().detect());
    });
}

criterion_group!(benches, bench_detect_all, bench_detect_none);
criterion_main!(benches);
