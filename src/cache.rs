//! Detection result caching with configurable TTL.
//!
//! Repeated calls to [`AcceleratorRegistry::detect`] re-run all CLI tools
//! and sysfs probes. For applications that call `detect()` frequently (e.g.
//! a scheduler polling on a timer), [`CachedRegistry`] avoids redundant work.
//!
//! # Examples
//!
//! ```rust
//! use std::time::Duration;
//! use ai_hwaccel::CachedRegistry;
//!
//! let cache = CachedRegistry::new(Duration::from_secs(60));
//! let registry = cache.get(); // first call: runs detection
//! let registry = cache.get(); // second call within 60s: returns cached (Arc, no clone)
//! ```

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::registry::AcceleratorRegistry;

/// A thread-safe cache for [`AcceleratorRegistry`] detection results.
///
/// Uses `Arc` internally so [`get`](Self::get) returns a cheap reference-counted
/// pointer instead of cloning the entire registry on every call.
pub struct CachedRegistry {
    ttl: Duration,
    inner: Mutex<CacheState>,
}

impl std::fmt::Debug for CachedRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedRegistry")
            .field("ttl", &self.ttl)
            .field(
                "cached",
                &self.inner.lock().is_ok_and(|s| s.registry.is_some()),
            )
            .finish()
    }
}

struct CacheState {
    registry: Option<Arc<AcceleratorRegistry>>,
    last_detect: Option<Instant>,
}

impl CachedRegistry {
    /// Create a new cache with the given time-to-live.
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            inner: Mutex::new(CacheState {
                registry: None,
                last_detect: None,
            }),
        }
    }

    /// Get the cached registry, re-detecting if the TTL has expired.
    ///
    /// Returns an `Arc` — cloning this is a cheap pointer increment, not a
    /// deep copy of all profiles.
    pub fn get(&self) -> Arc<AcceleratorRegistry> {
        let mut state = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::warn!("CachedRegistry lock was poisoned, invalidating cache");
                let mut guard = poisoned.into_inner();
                guard.registry = None;
                guard.last_detect = None;
                guard
            }
        };
        let now = Instant::now();

        if let Some(ref reg) = state.registry
            && let Some(last) = state.last_detect
            && now.duration_since(last) < self.ttl
        {
            return Arc::clone(reg);
        }

        let reg = Arc::new(AcceleratorRegistry::detect());
        state.registry = Some(Arc::clone(&reg));
        state.last_detect = Some(now);
        reg
    }

    /// Force the next call to [`get`](Self::get) to re-detect.
    pub fn invalidate(&self) {
        let mut state = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::warn!("CachedRegistry lock was poisoned, invalidating cache");
                let mut guard = poisoned.into_inner();
                guard.registry = None;
                guard.last_detect = None;
                guard
            }
        };
        state.registry = None;
        state.last_detect = None;
    }

    /// The configured time-to-live.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }
}
