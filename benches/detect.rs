//! Benchmarks for hardware detection and system I/O.

use criterion::{Criterion, criterion_group, criterion_main};

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

fn bench_detect_single_backend(c: &mut Criterion) {
    let mut group = c.benchmark_group("detect_single");

    group.bench_function("cuda", |b| {
        b.iter(|| DetectBuilder::none().with_cuda().detect());
    });
    group.bench_function("rocm", |b| {
        b.iter(|| DetectBuilder::none().with_rocm().detect());
    });
    group.bench_function("vulkan", |b| {
        b.iter(|| DetectBuilder::none().with_vulkan().detect());
    });
    group.bench_function("apple", |b| {
        b.iter(|| DetectBuilder::none().with_apple().detect());
    });
    group.bench_function("tpu", |b| {
        b.iter(|| DetectBuilder::none().with_tpu().detect());
    });

    group.finish();
}

fn bench_system_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("system_io");

    // Benchmark just the system I/O detection (interconnects + storage).
    // We do this by detecting everything and measuring the full pipeline,
    // then comparing against CPU-only to isolate system I/O overhead.
    group.bench_function("full_with_sysio", |b| {
        b.iter(AcceleratorRegistry::detect);
    });

    // Benchmark registry queries after detection.
    group.bench_function("query_system_io", |b| {
        let registry = AcceleratorRegistry::detect();
        b.iter(|| {
            let sio = registry.system_io();
            let _ = sio.total_interconnect_bandwidth_gbps();
            let _ = sio.estimate_ingestion_secs(100 * 1024 * 1024 * 1024);
            let _ = sio.has_interconnect();
        });
    });

    // Benchmark ingestion estimation at various scales.
    group.bench_function("ingestion_1gb", |b| {
        let registry = AcceleratorRegistry::detect();
        b.iter(|| registry.system_io().estimate_ingestion_secs(1_000_000_000));
    });
    group.bench_function("ingestion_100gb", |b| {
        let registry = AcceleratorRegistry::detect();
        b.iter(|| registry.system_io().estimate_ingestion_secs(100_000_000_000));
    });
    group.bench_function("ingestion_1tb", |b| {
        let registry = AcceleratorRegistry::detect();
        b.iter(|| registry.system_io().estimate_ingestion_secs(1_000_000_000_000));
    });

    // Benchmark serialization (includes system I/O fields).
    group.bench_function("serialize_registry", |b| {
        let registry = AcceleratorRegistry::detect();
        b.iter(|| serde_json::to_string(&registry).unwrap());
    });

    group.bench_function("deserialize_registry", |b| {
        let registry = AcceleratorRegistry::detect();
        let json = serde_json::to_string(&registry).unwrap();
        b.iter(|| serde_json::from_str::<AcceleratorRegistry>(&json).unwrap());
    });

    group.finish();
}

fn bench_concurrent_detect(c: &mut Criterion) {
    c.bench_function("concurrent_detect_4_threads", |b| {
        b.iter(|| {
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..4)
                    .map(|_| s.spawn(AcceleratorRegistry::detect))
                    .collect();
                for h in handles {
                    let _ = h.join();
                }
            });
        });
    });
}

criterion_group!(
    benches,
    bench_detect_all,
    bench_detect_none,
    bench_detect_single_backend,
    bench_system_io,
    bench_concurrent_detect,
);
criterion_main!(benches);
