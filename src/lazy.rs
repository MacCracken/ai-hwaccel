//! Lazy detection: probe backends only when their family is first queried.
//!
//! Unlike [`AcceleratorRegistry::detect`], which probes all enabled backends
//! upfront, [`LazyRegistry`] defers detection until the caller actually queries
//! a specific accelerator family. This avoids spawning `nvidia-smi` when the
//! caller only needs TPU info, for example.
//!
//! # Examples
//!
//! ```rust,no_run
//! use ai_hwaccel::{LazyRegistry, AcceleratorFamily};
//!
//! let lazy = LazyRegistry::new();
//!
//! // Only probes GPU backends (cuda, rocm, vulkan) — TPU/NPU/etc. untouched.
//! let gpus = lazy.by_family(AcceleratorFamily::Gpu);
//! ```

use std::sync::Mutex;

use crate::hardware::AcceleratorFamily;
use crate::profile::AcceleratorProfile;
use crate::registry::{AcceleratorRegistry, DetectBuilder};

/// A lazy registry that detects backends on first access per family.
///
/// Thread-safe — can be shared across threads via `Arc<LazyRegistry>`.
pub struct LazyRegistry {
    state: Mutex<LazyState>,
}

struct LazyState {
    /// Profiles detected so far (always includes CPU).
    profiles: Vec<AcceleratorProfile>,
    /// Which families have been probed.
    probed: [bool; 5], // Cpu, Gpu, Npu, Tpu, AiAsic
}

impl LazyRegistry {
    /// Create a new lazy registry. No detection runs until a query method is called.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(LazyState {
                profiles: vec![crate::detect::cpu_profile()],
                probed: [true, false, false, false, false], // CPU is always probed
            }),
        }
    }

    /// Ensure a family is probed, then return all profiles for that family.
    pub fn by_family(&self, family: AcceleratorFamily) -> Vec<AcceleratorProfile> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        Self::ensure_probed(&mut state, family);
        state
            .profiles
            .iter()
            .filter(|p| p.available && p.accelerator.family() == family)
            .cloned()
            .collect()
    }

    /// All profiles detected so far. Does NOT trigger additional detection.
    pub fn probed_profiles(&self) -> Vec<AcceleratorProfile> {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.profiles.clone()
    }

    /// Probe all families and return a full [`AcceleratorRegistry`].
    ///
    /// After this call, the lazy registry is fully populated.
    pub fn into_registry(self) -> AcceleratorRegistry {
        let mut state = self.state.into_inner().unwrap_or_else(|p| p.into_inner());
        for family in [
            AcceleratorFamily::Gpu,
            AcceleratorFamily::Npu,
            AcceleratorFamily::Tpu,
            AcceleratorFamily::AiAsic,
        ] {
            Self::ensure_probed(&mut state, family);
        }
        AcceleratorRegistry::from_profiles(state.profiles)
    }

    fn ensure_probed(state: &mut LazyState, family: AcceleratorFamily) {
        let idx = family_index(family);
        if state.probed[idx] {
            return;
        }
        state.probed[idx] = true;

        // Build a DetectBuilder that only enables backends for this family.
        let builder = match family {
            AcceleratorFamily::Cpu => return, // Always probed.
            AcceleratorFamily::Gpu => DetectBuilder::none()
                .with_cuda()
                .with_rocm()
                .with_vulkan(),
            AcceleratorFamily::Npu => DetectBuilder::none()
                .with_intel_npu()
                .with_amd_xdna()
                .with_samsung_npu()
                .with_mediatek_apu(),
            AcceleratorFamily::Tpu => DetectBuilder::none().with_tpu(),
            AcceleratorFamily::AiAsic => DetectBuilder::none()
                .with_gaudi()
                .with_aws_neuron()
                .with_intel_oneapi()
                .with_qualcomm()
                .with_cerebras()
                .with_graphcore()
                .with_groq(),
        };

        let partial = builder.detect();
        // Merge non-CPU profiles from partial detection.
        for p in partial.all_profiles() {
            if !matches!(p.accelerator, crate::hardware::AcceleratorType::Cpu) {
                state.profiles.push(p.clone());
            }
        }
    }
}

impl Default for LazyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LazyRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        f.debug_struct("LazyRegistry")
            .field("profiles", &state.profiles.len())
            .field("probed_families", &state.probed)
            .finish()
    }
}

fn family_index(family: AcceleratorFamily) -> usize {
    match family {
        AcceleratorFamily::Cpu => 0,
        AcceleratorFamily::Gpu => 1,
        AcceleratorFamily::Npu => 2,
        AcceleratorFamily::Tpu => 3,
        AcceleratorFamily::AiAsic => 4,
    }
}
