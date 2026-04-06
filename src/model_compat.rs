//! Model compatibility database.
//!
//! Answers questions like "can I run Llama 70B on 2x RTX 4090?" by matching
//! model memory requirements against available device memory. The model
//! catalogue is embedded at compile time from `data/models.json`.
//!
//! # Examples
//!
//! ```rust
//! use ai_hwaccel::model_compat::{all_models, compatible_models, can_run};
//! use ai_hwaccel::QuantizationLevel;
//!
//! // List all known models
//! let models = all_models();
//! assert!(!models.is_empty());
//!
//! // Check if a specific model fits in 48 GB
//! let llama_70b = all_models().iter().find(|m| m.name == "Llama 3.1 70B").unwrap();
//! assert!(can_run(llama_70b, &QuantizationLevel::Int4, 48 * 1024 * 1024 * 1024));
//!
//! // Find all models that fit in 24 GB at FP16
//! let fits = compatible_models(&QuantizationLevel::Float16, 24 * 1024 * 1024 * 1024);
//! assert!(fits.iter().any(|m| m.model.name == "Llama 3.1 8B"));
//! ```

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::quantization::QuantizationLevel;
use crate::registry::AcceleratorRegistry;

/// Embedded model catalogue (compiled in from data/models.json).
const MODELS_JSON: &str = include_str!("../data/models.json");

/// Parsed models, initialized once on first access.
static PARSED_MODELS: OnceLock<Vec<ModelProfile>> = OnceLock::new();

/// A known model from the compatibility database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    /// Human-readable name (e.g. "Llama 3.1 70B").
    pub name: String,
    /// Model family (e.g. "llama", "mistral", "phi").
    pub family: String,
    /// Total parameter count in billions.
    pub params_billions: f64,
    /// Default weight dtype (e.g. "bf16", "fp16").
    pub default_dtype: String,
    /// Supported serialisation formats (e.g. "safetensors", "gguf", "onnx").
    #[serde(default)]
    pub formats: Vec<String>,
    /// Supported context lengths (empty for non-LLM models).
    #[serde(default)]
    pub context_lengths: Vec<u64>,
}

impl ModelProfile {
    /// Total parameter count (not billions).
    #[must_use]
    #[inline]
    pub fn param_count(&self) -> u64 {
        (self.params_billions * 1_000_000_000.0) as u64
    }

    /// Estimate memory required at a given quantisation level (bytes).
    #[must_use]
    #[inline]
    pub fn memory_bytes(&self, quant: &QuantizationLevel) -> u64 {
        AcceleratorRegistry::estimate_memory(self.param_count(), quant)
    }

    /// Estimate memory required at a given quantisation level (GB).
    #[must_use]
    #[inline]
    pub fn memory_gb(&self, quant: &QuantizationLevel) -> f64 {
        self.memory_bytes(quant) as f64 / crate::units::BYTES_PER_GIB
    }
}

/// Result of a compatibility check.
#[derive(Debug, Clone)]
pub struct CompatResult<'a> {
    /// The model profile.
    pub model: &'a ModelProfile,
    /// Memory required at the given quantisation (bytes).
    pub memory_required_bytes: u64,
    /// Available memory (bytes).
    pub memory_available_bytes: u64,
    /// Headroom percentage.
    pub headroom_pct: f64,
}

/// Load all models from the embedded catalogue.
///
/// Results are parsed once and cached for the lifetime of the process.
#[must_use]
pub fn all_models() -> &'static [ModelProfile] {
    #[derive(Deserialize)]
    struct ModelData {
        #[serde(default)]
        models: Vec<ModelProfile>,
    }

    PARSED_MODELS.get_or_init(|| {
        serde_json::from_str::<ModelData>(MODELS_JSON)
            .map(|d| d.models)
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, "failed to parse embedded model catalogue");
                Vec::new()
            })
    })
}

/// Check whether a model can run with the given quantisation and memory budget.
#[must_use]
#[inline]
pub fn can_run(
    model: &ModelProfile,
    quant: &QuantizationLevel,
    available_memory_bytes: u64,
) -> bool {
    model.memory_bytes(quant) <= available_memory_bytes
}

/// Find all models from the catalogue that fit within the given memory budget.
///
/// Returns models sorted by parameter count (largest first), with headroom info.
#[must_use]
pub fn compatible_models(
    quant: &QuantizationLevel,
    available_memory_bytes: u64,
) -> Vec<CompatResult<'static>> {
    let mut results: Vec<CompatResult<'static>> = all_models()
        .iter()
        .filter_map(|model| {
            let needed = model.memory_bytes(quant);
            if needed <= available_memory_bytes {
                let headroom = if available_memory_bytes == 0 {
                    0.0
                } else {
                    (available_memory_bytes - needed) as f64 / available_memory_bytes as f64 * 100.0
                };
                Some(CompatResult {
                    model,
                    memory_required_bytes: needed,
                    memory_available_bytes: available_memory_bytes,
                    headroom_pct: headroom,
                })
            } else {
                None
            }
        })
        .collect();

    // Largest models first — more interesting to know "biggest model I can run".
    results.sort_by(|a, b| {
        b.model
            .params_billions
            .partial_cmp(&a.model.params_billions)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Find a model by exact name.
#[must_use]
pub fn find_model(name: &str) -> Option<&'static ModelProfile> {
    all_models().iter().find(|m| m.name == name)
}

/// Find models by family (e.g. "llama", "mistral").
#[must_use]
pub fn models_by_family(family: &str) -> Vec<&'static ModelProfile> {
    all_models()
        .iter()
        .filter(|m| m.family.eq_ignore_ascii_case(family))
        .collect()
}

/// Check model compatibility against an [`AcceleratorRegistry`].
///
/// Returns a list of models that can run on the best available device,
/// considering quantisation.
#[must_use]
pub fn compatible_with_registry(
    registry: &AcceleratorRegistry,
    quant: &QuantizationLevel,
) -> Vec<CompatResult<'static>> {
    let total_memory = registry.total_accelerator_memory();
    compatible_models(quant, total_memory)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_models_not_empty() {
        assert!(!all_models().is_empty());
    }

    #[test]
    fn all_models_have_valid_params() {
        for model in all_models() {
            assert!(
                model.params_billions > 0.0,
                "model {} has zero params",
                model.name
            );
            assert!(
                model.param_count() > 0,
                "model {} param_count is zero",
                model.name
            );
        }
    }

    #[test]
    fn find_model_exists() {
        assert!(find_model("Llama 3.1 70B").is_some());
        assert!(find_model("Nonexistent Model").is_none());
    }

    #[test]
    fn models_by_family_llama() {
        let llamas = models_by_family("llama");
        assert!(llamas.len() >= 3);
        assert!(llamas.iter().all(|m| m.family == "llama"));
    }

    #[test]
    fn can_run_small_model_large_memory() {
        let model = find_model("Llama 3.2 1B").unwrap();
        // 1B at INT4 needs ~0.6 GB — 24 GB should be plenty.
        assert!(can_run(
            model,
            &QuantizationLevel::Int4,
            24 * 1024 * 1024 * 1024
        ));
    }

    #[test]
    fn cannot_run_huge_model_small_memory() {
        let model = find_model("Llama 3.1 405B").unwrap();
        // 405B at FP32 needs ~972 GB — 24 GB is not enough.
        assert!(!can_run(
            model,
            &QuantizationLevel::None,
            24 * 1024 * 1024 * 1024
        ));
    }

    #[test]
    fn compatible_models_24gb_fp16() {
        let results = compatible_models(&QuantizationLevel::Float16, 24 * 1024 * 1024 * 1024);
        // Should include small models like 1B, 3B, 7B, 8B
        assert!(!results.is_empty());
        // Should NOT include 70B+ models at FP16
        assert!(!results.iter().any(|r| r.model.params_billions >= 70.0));
        // Should be sorted largest first
        for w in results.windows(2) {
            assert!(w[0].model.params_billions >= w[1].model.params_billions);
        }
    }

    #[test]
    fn compatible_models_headroom_is_valid() {
        let results = compatible_models(&QuantizationLevel::Int4, 80 * 1024 * 1024 * 1024);
        for r in &results {
            assert!(r.headroom_pct >= 0.0);
            assert!(r.headroom_pct <= 100.0);
        }
    }

    #[test]
    fn memory_gb_reasonable() {
        let model = find_model("Llama 3.1 8B").unwrap();
        let gb = model.memory_gb(&QuantizationLevel::Float16);
        // 8B at FP16: ~16B bytes * 1.2 overhead = ~19.2 GB
        assert!(gb > 15.0 && gb < 25.0, "8B FP16 memory: {gb} GB");
    }

    #[test]
    fn model_profile_serde_roundtrip() {
        let model = find_model("Mistral 7B").unwrap();
        let json = serde_json::to_string(model).unwrap();
        let back: ModelProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(model.name, back.name);
        assert_eq!(model.param_count(), back.param_count());
    }
}
