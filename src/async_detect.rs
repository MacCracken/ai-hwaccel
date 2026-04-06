//! True async hardware detection via `tokio::process`.
//!
//! Requires the `async-detect` cargo feature:
//!
//! ```toml
//! [dependencies]
//! ai-hwaccel = { version = "1.2", features = ["async-detect"] }
//! ```
//!
//! Unlike the sync [`AcceleratorRegistry::detect`], this uses
//! `tokio::process::Command` for non-blocking subprocess I/O. Each CLI
//! backend runs as a concurrent tokio task, and sysfs-only backends run in
//! a single `spawn_blocking` task.
//!
//! # Examples
//!
//! ```rust,ignore
//! use ai_hwaccel::AcceleratorRegistry;
//!
//! #[tokio::main]
//! async fn main() {
//!     let registry = AcceleratorRegistry::detect_async().await.unwrap();
//!     println!("Best: {}", registry.best_available().unwrap());
//! }
//! ```

use crate::registry::AcceleratorRegistry;

/// Error returned when async detection fails due to a task panic.
#[cfg(feature = "async-detect")]
#[derive(Debug, Clone)]
pub struct AsyncDetectError;

#[cfg(feature = "async-detect")]
impl std::fmt::Display for AsyncDetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "detection task panicked")
    }
}

#[cfg(feature = "async-detect")]
impl std::error::Error for AsyncDetectError {}

impl AcceleratorRegistry {
    /// True async variant of [`detect`](Self::detect).
    ///
    /// Uses `tokio::process::Command` for non-blocking subprocess I/O
    /// instead of `spawn_blocking`. Each CLI backend (nvidia-smi, vulkaninfo,
    /// hl-smi, etc.) runs as a concurrent tokio task. Sysfs-only backends
    /// (ROCm, TPU, Intel NPU, etc.) run in a single `spawn_blocking` task
    /// since they are fast filesystem reads.
    ///
    /// Requires the `async-detect` feature.
    #[cfg(feature = "async-detect")]
    pub async fn detect_async() -> Result<Self, AsyncDetectError> {
        Ok(crate::detect::detect_with_builder_async(crate::registry::DetectBuilder::new()).await)
    }
}

impl crate::registry::DetectBuilder {
    /// True async variant of [`detect`](Self::detect).
    ///
    /// Requires the `async-detect` feature.
    #[cfg(feature = "async-detect")]
    pub async fn detect_async(self) -> Result<AcceleratorRegistry, AsyncDetectError> {
        Ok(crate::detect::detect_with_builder_async(self).await)
    }
}
