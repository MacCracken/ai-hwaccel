//! Async hardware detection via `tokio::process`.
//!
//! Requires the `async-detect` cargo feature:
//!
//! ```toml
//! [dependencies]
//! ai-hwaccel = { version = "0.19", features = ["async-detect"] }
//! ```
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

/// Error returned when async detection fails due to a thread panic.
#[cfg(feature = "async-detect")]
#[derive(Debug, Clone)]
pub struct AsyncDetectError;

#[cfg(feature = "async-detect")]
impl std::fmt::Display for AsyncDetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "detection thread panicked")
    }
}

#[cfg(feature = "async-detect")]
impl std::error::Error for AsyncDetectError {}

impl AcceleratorRegistry {
    /// Async variant of [`detect`](Self::detect).
    ///
    /// Spawns detection on a blocking thread via [`tokio::task::spawn_blocking`]
    /// so it doesn't block the async runtime. The underlying detection already
    /// runs backends in parallel via `std::thread::scope`.
    ///
    /// Requires the `async-detect` feature.
    #[cfg(feature = "async-detect")]
    pub async fn detect_async() -> Result<Self, AsyncDetectError> {
        tokio::task::spawn_blocking(Self::detect)
            .await
            .map_err(|_| AsyncDetectError)
    }
}

impl crate::registry::DetectBuilder {
    /// Async variant of [`detect`](Self::detect).
    ///
    /// Requires the `async-detect` feature.
    #[cfg(feature = "async-detect")]
    pub async fn detect_async(self) -> Result<AcceleratorRegistry, AsyncDetectError> {
        tokio::task::spawn_blocking(move || self.detect())
            .await
            .map_err(|_| AsyncDetectError)
    }
}
