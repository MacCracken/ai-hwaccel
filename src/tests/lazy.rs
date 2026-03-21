//! Tests for LazyRegistry.

use crate::*;

#[test]
fn lazy_new_has_cpu_only() {
    let lazy = LazyRegistry::new();
    let profiles = lazy.probed_profiles();
    assert_eq!(profiles.len(), 1);
    assert!(matches!(profiles[0].accelerator, AcceleratorType::Cpu));
}

#[test]
fn lazy_by_family_cpu_returns_cpu() {
    let lazy = LazyRegistry::new();
    let cpus = lazy.by_family(AcceleratorFamily::Cpu);
    assert_eq!(cpus.len(), 1);
    assert!(matches!(cpus[0].accelerator, AcceleratorType::Cpu));
}

#[test]
fn lazy_by_family_gpu_triggers_detection() {
    let lazy = LazyRegistry::new();
    // GPU detection runs (may find devices or not depending on hardware).
    let _gpus = lazy.by_family(AcceleratorFamily::Gpu);
    // After probing GPU, we should have at least 1 profile (CPU is always present).
    assert!(!lazy.probed_profiles().is_empty());
}

#[test]
fn lazy_into_registry_probes_all() {
    let lazy = LazyRegistry::new();
    let registry = lazy.into_registry();
    // Should have at least the CPU profile.
    assert!(!registry.all_profiles().is_empty());
    assert!(registry
        .all_profiles()
        .iter()
        .any(|p| matches!(p.accelerator, AcceleratorType::Cpu)));
}

#[test]
fn lazy_default_same_as_new() {
    let lazy = LazyRegistry::default();
    assert_eq!(lazy.probed_profiles().len(), 1);
}

#[test]
fn lazy_debug_impl() {
    let lazy = LazyRegistry::new();
    let debug = format!("{:?}", lazy);
    assert!(debug.contains("LazyRegistry"));
}

#[test]
fn lazy_repeated_family_query_idempotent() {
    let lazy = LazyRegistry::new();
    let first = lazy.by_family(AcceleratorFamily::Cpu);
    let second = lazy.by_family(AcceleratorFamily::Cpu);
    assert_eq!(first.len(), second.len());
}
