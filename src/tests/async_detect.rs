//! Tests for async detection.
//!
//! Requires the `async-detect` feature (enabled in dev-dependencies via tokio).

#[cfg(feature = "async-detect")]
mod async_tests {
    use crate::*;

    #[tokio::test]
    async fn detect_async_returns_cpu() {
        let registry = AcceleratorRegistry::detect_async().await.unwrap();
        assert!(!registry.all_profiles().is_empty());
        assert!(registry
            .all_profiles()
            .iter()
            .any(|p| matches!(p.accelerator, AcceleratorType::Cpu)));
    }

    #[tokio::test]
    async fn detect_async_matches_sync() {
        let async_reg = AcceleratorRegistry::detect_async().await.unwrap();
        let sync_reg = AcceleratorRegistry::detect();

        // Same number of device families detected (exact profile count may differ
        // due to timing, but families should be the same).
        let async_families: std::collections::HashSet<_> = async_reg
            .all_profiles()
            .iter()
            .map(|p| p.accelerator.family())
            .collect();
        let sync_families: std::collections::HashSet<_> = sync_reg
            .all_profiles()
            .iter()
            .map(|p| p.accelerator.family())
            .collect();
        assert_eq!(async_families, sync_families);
    }

    #[tokio::test]
    async fn detect_async_builder() {
        let registry = DetectBuilder::none()
            .detect_async()
            .await
            .unwrap();
        // CPU-only (no backends enabled).
        assert!(registry.all_profiles().len() >= 1);
        assert!(registry
            .all_profiles()
            .iter()
            .all(|p| matches!(p.accelerator, AcceleratorType::Cpu)));
    }

    #[tokio::test]
    async fn detect_async_has_warnings_not_errors() {
        let registry = AcceleratorRegistry::detect_async().await.unwrap();
        // Warnings are non-fatal — the detection should never return Err
        // unless a task panicked.
        for w in registry.warnings() {
            // Should be tool-not-found or parse errors, not panics.
            let msg = format!("{}", w);
            assert!(!msg.contains("panic"), "unexpected panic warning: {}", msg);
        }
    }

    #[tokio::test]
    async fn detect_async_serializable() {
        let registry = AcceleratorRegistry::detect_async().await.unwrap();
        let json = serde_json::to_string(&registry).unwrap();
        let deserialized = AcceleratorRegistry::from_json(&json).unwrap();
        assert_eq!(
            registry.all_profiles().len(),
            deserialized.all_profiles().len()
        );
    }
}
