//! Tests for per-backend timing.

use crate::*;

#[test]
fn timed_detection_returns_timings() {
    let result = AcceleratorRegistry::detect_with_timing();
    // Should have at least one backend timing entry.
    assert!(!result.timings.is_empty());
    // Total wall time should be positive.
    assert!(result.total.as_nanos() > 0);
    // Registry should have at least the CPU profile.
    assert!(!result.registry.all_profiles().is_empty());
}

#[test]
fn timed_detection_has_enrich_timing() {
    let result = AcceleratorRegistry::detect_with_timing();
    // _enrich and _system_io are always present.
    assert!(result.timings.contains_key("_enrich"));
    assert!(result.timings.contains_key("_system_io"));
}

#[test]
fn timed_detection_total_ge_max_backend() {
    let result = AcceleratorRegistry::detect_with_timing();
    let max_backend = result
        .timings
        .values()
        .max()
        .copied()
        .unwrap_or_default();
    // Total wall time should be >= the slowest individual backend
    // (though total includes sequential post-passes).
    assert!(result.total >= max_backend);
}

#[test]
fn builder_timed_detection() {
    let result = AcceleratorRegistry::builder()
        .without_vulkan()
        .detect_with_timing();
    // Vulkan should not appear in timings.
    assert!(!result.timings.contains_key("vulkan"));
    assert!(!result.registry.all_profiles().is_empty());
}
