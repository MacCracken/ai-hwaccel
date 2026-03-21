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
    ///
    /// The lock is released before running detection, so concurrent readers
    /// are not blocked during the (potentially slow) detection phase.
    pub fn get(&self) -> Arc<AcceleratorRegistry> {
        // Fast path: check cache under lock, release immediately.
        {
            let state = match self.inner.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    tracing::warn!("CachedRegistry lock was poisoned, invalidating cache");
                    let mut guard = poisoned.into_inner();
                    guard.registry = None;
                    guard.last_detect = None;
                    guard
                }
            };

            if let Some(ref reg) = state.registry
                && let Some(last) = state.last_detect
                && Instant::now().duration_since(last) < self.ttl
            {
                return Arc::clone(reg);
            }
        }
        // Lock released here — detection runs without blocking readers.

        let reg = Arc::new(AcceleratorRegistry::detect());

        // Re-acquire lock to store result.
        let mut state = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.registry = Some(Arc::clone(&reg));
        state.last_detect = Some(Instant::now());
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

// ---------------------------------------------------------------------------
// Disk-backed cache
// ---------------------------------------------------------------------------

/// A [`CachedRegistry`] variant that persists detection results to disk.
///
/// On `get()`, reads from the cache file if it exists and is within TTL.
/// On cache miss, runs detection and writes results to disk. Useful for
/// gateway servers or orchestrators that call `detect()` at startup and
/// don't need real-time hardware change tracking.
///
/// # Examples
///
/// ```rust,no_run
/// use std::time::Duration;
/// use ai_hwaccel::DiskCachedRegistry;
///
/// let cache = DiskCachedRegistry::new(Duration::from_secs(300));
/// let registry = cache.get(); // first call: detects and writes to disk
/// let registry = cache.get(); // within TTL: reads from disk (instant)
/// ```
pub struct DiskCachedRegistry {
    ttl: Duration,
    cache_path: std::path::PathBuf,
    memory: Mutex<CacheState>,
}

impl std::fmt::Debug for DiskCachedRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskCachedRegistry")
            .field("ttl", &self.ttl)
            .field("cache_path", &self.cache_path)
            .finish()
    }
}

impl DiskCachedRegistry {
    /// Create a new disk-backed cache with the given TTL.
    ///
    /// Cache file is stored at `$XDG_CACHE_HOME/ai-hwaccel/registry.json`
    /// (or `~/.cache/ai-hwaccel/registry.json` if `XDG_CACHE_HOME` is unset).
    pub fn new(ttl: Duration) -> Self {
        let cache_path = Self::default_cache_path();
        Self {
            ttl,
            cache_path,
            memory: Mutex::new(CacheState {
                registry: None,
                last_detect: None,
            }),
        }
    }

    /// Create a disk-backed cache with a custom file path.
    pub fn with_path(ttl: Duration, path: std::path::PathBuf) -> Self {
        Self {
            ttl,
            cache_path: path,
            memory: Mutex::new(CacheState {
                registry: None,
                last_detect: None,
            }),
        }
    }

    /// Get the cached registry, re-detecting if the TTL has expired.
    pub fn get(&self) -> Arc<AcceleratorRegistry> {
        let mut state = match self.memory.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::warn!("DiskCachedRegistry lock was poisoned, invalidating cache");
                let mut guard = poisoned.into_inner();
                guard.registry = None;
                guard.last_detect = None;
                guard
            }
        };

        // Check in-memory cache first.
        if let Some(ref reg) = state.registry
            && let Some(last) = state.last_detect
            && Instant::now().duration_since(last) < self.ttl
        {
            return Arc::clone(reg);
        }

        // Check disk cache.
        if let Some(reg) = self.read_disk_cache() {
            let arc = Arc::new(reg);
            state.registry = Some(Arc::clone(&arc));
            state.last_detect = Some(Instant::now());
            return arc;
        }

        // Cache miss — run detection.
        let reg = Arc::new(AcceleratorRegistry::detect());
        state.registry = Some(Arc::clone(&reg));
        state.last_detect = Some(Instant::now());

        // Write to disk (best-effort).
        self.write_disk_cache(&reg);

        reg
    }

    /// Force the next call to re-detect and update the disk cache.
    pub fn invalidate(&self) {
        let mut state = match self.memory.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.registry = None;
        state.last_detect = None;
        let _ = std::fs::remove_file(&self.cache_path);
    }

    /// The cache file path.
    pub fn cache_path(&self) -> &std::path::Path {
        &self.cache_path
    }

    fn default_cache_path() -> std::path::PathBuf {
        let cache_dir = std::env::var("XDG_CACHE_HOME")
            .ok()
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| std::path::PathBuf::from(h).join(".cache"))
            })
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
        cache_dir.join("ai-hwaccel").join("registry.json")
    }

    fn read_disk_cache(&self) -> Option<AcceleratorRegistry> {
        let metadata = std::fs::metadata(&self.cache_path).ok()?;
        let age = metadata
            .modified()
            .ok()?
            .elapsed()
            .unwrap_or(Duration::MAX);
        if age > self.ttl {
            return None;
        }
        let data = std::fs::read_to_string(&self.cache_path).ok()?;
        let reg = AcceleratorRegistry::from_json(&data).ok()?;
        tracing::debug!(
            age_secs = age.as_secs_f64(),
            path = %self.cache_path.display(),
            "loaded registry from disk cache"
        );
        Some(reg)
    }

    fn write_disk_cache(&self, registry: &AcceleratorRegistry) {
        if let Some(parent) = self.cache_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string(registry) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&self.cache_path, json) {
                    tracing::debug!(
                        error = %e,
                        path = %self.cache_path.display(),
                        "failed to write disk cache"
                    );
                }
            }
            Err(e) => {
                tracing::debug!(error = %e, "failed to serialize registry for disk cache");
            }
        }
    }
}
