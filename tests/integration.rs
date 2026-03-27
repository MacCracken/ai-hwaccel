//! Crate-level integration tests exercising the public API end-to-end.

use ai_hwaccel::*;

// ---------------------------------------------------------------------------
// Full detect → query → plan pipeline
// ---------------------------------------------------------------------------

#[test]
fn detect_query_plan_pipeline() {
    let registry = AcceleratorRegistry::detect();

    // Always has at least CPU.
    assert!(!registry.all_profiles().is_empty());
    assert!(registry.best_available().is_some());
    assert!(registry.total_memory() > 0);

    // Suggest quantization for a 7B model.
    let quant = registry.suggest_quantization(7_000_000_000);
    assert!(quant.bits_per_param() <= 32);

    // Plan sharding — should always produce a plan (even CPU-only).
    let plan = registry.plan_sharding(7_000_000_000, &quant);
    assert!(!plan.shards().is_empty());
    assert!(plan.total_memory_bytes > 0);
}

// ---------------------------------------------------------------------------
// Builder → selective detection
// ---------------------------------------------------------------------------

#[test]
fn builder_none_is_cpu_only() {
    let registry = DetectBuilder::none().detect();
    assert_eq!(registry.all_profiles().len(), 1);
    assert_eq!(registry.all_profiles()[0].accelerator, AcceleratorType::Cpu);
    assert!(!registry.has_accelerator());
}

#[test]
fn builder_all_equals_detect() {
    let full = AcceleratorRegistry::detect();
    let built = AcceleratorRegistry::builder().detect();
    // Same number of profiles (parallel execution order may differ, but
    // the set of detected hardware should be identical).
    assert_eq!(full.all_profiles().len(), built.all_profiles().len());
}

// ---------------------------------------------------------------------------
// Serde round-trip through JSON
// ---------------------------------------------------------------------------

#[test]
fn registry_json_roundtrip() {
    let registry = AcceleratorRegistry::detect();
    let json = serde_json::to_string(&registry).unwrap();
    let back: AcceleratorRegistry = serde_json::from_str(&json).unwrap();
    assert_eq!(registry.all_profiles().len(), back.all_profiles().len());
}

#[test]
fn registry_pretty_json_roundtrip() {
    let registry = AcceleratorRegistry::detect();
    let json = serde_json::to_string_pretty(&registry).unwrap();
    let back: AcceleratorRegistry = serde_json::from_str(&json).unwrap();
    assert_eq!(registry.all_profiles().len(), back.all_profiles().len());
}

// ---------------------------------------------------------------------------
// Convenience constructors → registry
// ---------------------------------------------------------------------------

#[test]
fn manual_registry_plan() {
    let registry = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(64 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(0, 80 * 1024 * 1024 * 1024),
        AcceleratorProfile::cuda(1, 80 * 1024 * 1024 * 1024),
    ]);

    assert!(registry.has_accelerator());
    assert_eq!(registry.by_family(AcceleratorFamily::Gpu).len(), 2);

    let plan = registry.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16);
    // 70B BF16 ~168 GB, each GPU 80 GB — should pipeline across 2
    assert!(plan.shards().len() <= 2);
    assert!(plan.estimated_tokens_per_sec.is_some());
}

// ---------------------------------------------------------------------------
// Training memory estimation
// ---------------------------------------------------------------------------

#[test]
fn training_estimate_consistency() {
    let est = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Gpu);
    assert!(est.total_gb > 0.0);
    assert!((est.model_gb + est.optimizer_gb + est.activation_gb - est.total_gb).abs() < 0.001);
}

// ---------------------------------------------------------------------------
// Display impls don't panic
// ---------------------------------------------------------------------------

#[test]
fn display_impls_dont_panic() {
    let registry = AcceleratorRegistry::detect();
    for p in registry.all_profiles() {
        let _ = p.to_string();
        let _ = p.accelerator.to_string();
        let _ = p.accelerator.family().to_string();
    }
    let plan = registry.plan_sharding(7_000_000_000, &QuantizationLevel::Float16);
    let _ = plan.to_string();
    let _ = plan.strategy.to_string();
}

// ---------------------------------------------------------------------------
// Warnings are non-fatal
// ---------------------------------------------------------------------------

#[test]
fn warnings_dont_prevent_detection() {
    let registry = AcceleratorRegistry::detect();
    // Regardless of warnings, CPU should always be present.
    assert!(
        registry
            .all_profiles()
            .iter()
            .any(|p| matches!(p.accelerator, AcceleratorType::Cpu))
    );
    // Warnings should be displayable.
    for w in registry.warnings() {
        let _ = w.to_string();
    }
}
