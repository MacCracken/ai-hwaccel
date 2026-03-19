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
//! let registry = cache.get(); // second call within 60s: returns cached
//! ```

use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::registry::AcceleratorRegistry;

/// A thread-safe cache for [`AcceleratorRegistry`] detection results.
///
/// The cache is populated on the first call to [`get`](Self::get) and
/// refreshed after the TTL expires. The cache can be manually invalidated
/// with [`invalidate`](Self::invalidate).
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
    registry: Option<AcceleratorRegistry>,
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
    pub fn get(&self) -> AcceleratorRegistry {
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();

        if let Some(ref reg) = state.registry
            && let Some(last) = state.last_detect
            && now.duration_since(last) < self.ttl
        {
            return reg.clone();
        }

        let reg = AcceleratorRegistry::detect();
        state.registry = Some(reg.clone());
        state.last_detect = Some(now);
        reg
    }

    /// Force the next call to [`get`](Self::get) to re-detect.
    pub fn invalidate(&self) {
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        state.registry = None;
        state.last_detect = None;
    }

    /// The configured time-to-live.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }
}
