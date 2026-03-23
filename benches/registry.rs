//! Benchmarks for registry queries, cache, lazy detection, and large registries.

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use ai_hwaccel::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_large_registry(n_gpus: u32) -> AcceleratorRegistry {
    let mut profiles = vec![AcceleratorProfile::cpu(512 * 1024 * 1024 * 1024)];
    for i in 0..n_gpus {
        profiles.push(AcceleratorProfile::cuda(i, 80 * 1024 * 1024 * 1024));
    }
    AcceleratorRegistry::from_profiles(profiles)
}

fn build_mixed_large_registry() -> AcceleratorRegistry {
    let mut profiles = vec![AcceleratorProfile::cpu(512 * 1024 * 1024 * 1024)];
    for i in 0..32 {
        profiles.push(AcceleratorProfile::cuda(i, 80 * 1024 * 1024 * 1024));
    }
    for i in 0..16 {
        profiles.push(AcceleratorProfile::rocm(i, 192 * 1024 * 1024 * 1024));
    }
    for i in 0..8 {
        profiles.push(AcceleratorProfile::tpu(i, 4, TpuVersion::V5p));
    }
    for i in 0..4 {
        profiles.push(AcceleratorProfile::gaudi(i, GaudiGeneration::Gaudi3));
    }
    AcceleratorRegistry::from_profiles(profiles)
}

// ---------------------------------------------------------------------------
// Registry query benchmarks
// ---------------------------------------------------------------------------

fn bench_registry_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("registry_queries");

    let small = build_large_registry(4);
    let large = build_large_registry(128);
    let mixed = build_mixed_large_registry();

    group.bench_function("available_4dev", |b| {
        b.iter(|| small.available());
    });
    group.bench_function("available_129dev", |b| {
        b.iter(|| large.available());
    });
    group.bench_function("available_61dev_mixed", |b| {
        b.iter(|| mixed.available());
    });

    group.bench_function("best_available_129dev", |b| {
        b.iter(|| large.best_available());
    });

    group.bench_function("total_memory_129dev", |b| {
        b.iter(|| large.total_memory());
    });

    group.bench_function("total_accelerator_memory_129dev", |b| {
        b.iter(|| large.total_accelerator_memory());
    });

    group.bench_function("has_accelerator_129dev", |b| {
        b.iter(|| large.has_accelerator());
    });

    group.bench_function("by_family_gpu_61dev", |b| {
        b.iter(|| mixed.by_family(AcceleratorFamily::Gpu));
    });
    group.bench_function("by_family_tpu_61dev", |b| {
        b.iter(|| mixed.by_family(AcceleratorFamily::Tpu));
    });

    group.bench_function("satisfying_gpu_61dev", |b| {
        b.iter(|| mixed.satisfying(&AcceleratorRequirement::Gpu));
    });
    group.bench_function("satisfying_any_accel_61dev", |b| {
        b.iter(|| mixed.satisfying(&AcceleratorRequirement::AnyAccelerator));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Cache benchmarks
// ---------------------------------------------------------------------------

fn bench_cached_registry(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_registry");

    group.bench_function("get_cached_hit", |b| {
        let cache = CachedRegistry::new(Duration::from_secs(300));
        let _ = cache.get(); // prime the cache
        b.iter(|| cache.get());
    });

    group.bench_function("invalidate", |b| {
        let cache = CachedRegistry::new(Duration::from_secs(300));
        let _ = cache.get();
        b.iter(|| cache.invalidate());
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Lazy registry benchmarks
// ---------------------------------------------------------------------------

fn bench_lazy_registry(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_registry");

    group.bench_function("new", |b| {
        b.iter(LazyRegistry::new);
    });

    group.bench_function("by_family_gpu_cold", |b| {
        b.iter(|| {
            let lazy = LazyRegistry::new();
            lazy.by_family(AcceleratorFamily::Gpu)
        });
    });

    group.bench_function("by_family_gpu_warm", |b| {
        let lazy = LazyRegistry::new();
        let _ = lazy.by_family(AcceleratorFamily::Gpu);
        b.iter(|| lazy.by_family(AcceleratorFamily::Gpu));
    });

    group.bench_function("into_registry", |b| {
        b.iter(|| {
            let lazy = LazyRegistry::new();
            lazy.into_registry()
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Large-registry sharding benchmarks
// ---------------------------------------------------------------------------

fn bench_large_registry_sharding(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_registry_sharding");

    let reg_128 = build_large_registry(128);
    let mixed = build_mixed_large_registry();

    group.bench_function("plan_sharding_70B_128gpu", |b| {
        b.iter(|| reg_128.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16));
    });

    group.bench_function("plan_sharding_405B_128gpu", |b| {
        b.iter(|| reg_128.plan_sharding(405_000_000_000, &QuantizationLevel::Int8));
    });

    group.bench_function("suggest_quantization_70B_128gpu", |b| {
        b.iter(|| reg_128.suggest_quantization(70_000_000_000));
    });

    group.bench_function("plan_sharding_70B_mixed_61dev", |b| {
        b.iter(|| mixed.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// JSON serialization at scale
// ---------------------------------------------------------------------------

fn bench_large_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_json");

    let reg = build_large_registry(128);
    let json = serde_json::to_string(&reg).unwrap();

    group.bench_function("serialize_129dev", |b| {
        b.iter(|| serde_json::to_string(&reg).unwrap());
    });

    group.bench_function("deserialize_129dev", |b| {
        b.iter(|| AcceleratorRegistry::from_json(&json).unwrap());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_registry_queries,
    bench_cached_registry,
    bench_lazy_registry,
    bench_large_registry_sharding,
    bench_large_json,
);
criterion_main!(benches);
