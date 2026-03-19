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
//!     let registry = AcceleratorRegistry::detect_async().await;
//!     println!("Best: {}", registry.best_available().unwrap());
//! }
//! ```

use crate::registry::AcceleratorRegistry;

impl AcceleratorRegistry {
    /// Async variant of [`detect`](Self::detect).
    ///
    /// Spawns detection on a blocking thread via [`tokio::task::spawn_blocking`]
    /// so it doesn't block the async runtime. The underlying detection already
    /// runs backends in parallel via `std::thread::scope`.
    ///
    /// Requires the `async-detect` feature.
    #[cfg(feature = "async-detect")]
    pub async fn detect_async() -> Self {
        tokio::task::spawn_blocking(Self::detect)
            .await
            .expect("detection thread panicked")
    }
}

impl crate::registry::DetectBuilder {
    /// Async variant of [`detect`](Self::detect).
    ///
    /// Requires the `async-detect` feature.
    #[cfg(feature = "async-detect")]
    pub async fn detect_async(self) -> AcceleratorRegistry {
        tokio::task::spawn_blocking(move || self.detect())
            .await
            .expect("detection thread panicked")
    }
}
