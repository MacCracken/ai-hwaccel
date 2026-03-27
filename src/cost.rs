//! Cost-aware cloud instance planning.
//!
//! Given a model size and quantisation level, recommends the cheapest viable
//! cloud GPU instance from a static pricing table. The pricing data lives in
//! `data/cloud_pricing.json` and is embedded at compile time — update the JSON
//! file and recompile to refresh prices.
//!
//! # Examples
//!
//! ```rust
//! use ai_hwaccel::cost::{recommend_instance, CloudProvider};
//! use ai_hwaccel::QuantizationLevel;
//!
//! let recs = recommend_instance(70_000_000_000, &QuantizationLevel::BFloat16, None);
//! for rec in &recs {
//!     println!("{} ({}) — ${:.2}/hr, {} GB GPU memory",
//!         rec.instance.name, rec.instance.provider,
//!         rec.instance.price_per_hour, rec.instance.total_gpu_memory_gb);
//! }
//! ```

use std::sync::OnceLock;

use crate::quantization::QuantizationLevel;
use crate::registry::AcceleratorRegistry;

/// Embedded pricing data (compiled in from data/cloud_pricing.json).
const PRICING_JSON: &str = include_str!("../data/cloud_pricing.json");

/// Parsed instances, initialized once on first access.
static PARSED_INSTANCES: OnceLock<Vec<CloudInstance>> = OnceLock::new();

/// A cloud GPU instance from the pricing table.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CloudInstance {
    /// Instance name (e.g. "p5.48xlarge").
    pub name: String,
    /// Cloud provider (e.g. "aws", "gcp", "azure").
    pub provider: String,
    /// GPU model name.
    pub gpu: String,
    /// Number of GPUs.
    pub gpu_count: u32,
    /// Memory per GPU in GB.
    pub gpu_memory_gb: u32,
    /// Total GPU memory in GB.
    pub total_gpu_memory_gb: u32,
    /// vCPU count.
    pub vcpus: u32,
    /// System RAM in GB.
    pub ram_gb: u32,
    /// Interconnect type.
    pub interconnect: String,
    /// On-demand price in USD/hour.
    pub price_per_hour: f64,
}

/// Cloud provider filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CloudProvider {
    Aws,
    Gcp,
    Azure,
}

impl CloudProvider {
    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            Self::Aws => "aws",
            Self::Gcp => "gcp",
            Self::Azure => "azure",
        }
    }
}

/// A recommended instance with cost analysis.
#[derive(Debug, Clone)]
pub struct InstanceRecommendation {
    /// The recommended cloud instance.
    pub instance: CloudInstance,
    /// Estimated memory required for the model (bytes).
    pub memory_required_bytes: u64,
    /// Memory headroom percentage (how much extra memory is available).
    pub memory_headroom_pct: f64,
}

/// Load all cloud instances from the embedded pricing table.
///
/// Results are parsed once and cached for the lifetime of the process.
#[must_use]
pub fn all_instances() -> &'static [CloudInstance] {
    #[derive(serde::Deserialize)]
    struct PricingData {
        #[serde(default)]
        instances: Vec<CloudInstance>,
    }

    PARSED_INSTANCES.get_or_init(|| {
        serde_json::from_str::<PricingData>(PRICING_JSON)
            .map(|d| d.instances)
            .unwrap_or_default()
    })
}

/// Recommend the cheapest viable cloud instance(s) for a model.
///
/// Returns instances sorted by price (cheapest first) that have enough
/// total GPU memory to fit the model at the given quantisation level.
///
/// # Arguments
///
/// * `model_params` — number of model parameters
/// * `quant` — quantisation level
/// * `provider` — optional filter to a specific cloud provider
#[must_use]
pub fn recommend_instance(
    model_params: u64,
    quant: &QuantizationLevel,
    provider: Option<CloudProvider>,
) -> Vec<InstanceRecommendation> {
    let needed = AcceleratorRegistry::estimate_memory(model_params, quant);
    let needed_gb = (needed as f64) / crate::units::BYTES_PER_GIB;

    let mut candidates: Vec<InstanceRecommendation> = all_instances()
        .iter()
        .filter(|inst| {
            if let Some(p) = provider
                && inst.provider != p.as_str()
            {
                return false;
            }
            inst.total_gpu_memory_gb as f64 >= needed_gb
        })
        .map(|inst| {
            let headroom = (inst.total_gpu_memory_gb as f64 - needed_gb)
                / inst.total_gpu_memory_gb as f64
                * 100.0;
            InstanceRecommendation {
                instance: inst.clone(),
                memory_required_bytes: needed,
                memory_headroom_pct: headroom,
            }
        })
        .collect();

    candidates.sort_by(|a, b| {
        a.instance
            .price_per_hour
            .partial_cmp(&b.instance.price_per_hour)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Recommend the single cheapest instance, if any.
#[must_use]
pub fn cheapest_instance(
    model_params: u64,
    quant: &QuantizationLevel,
    provider: Option<CloudProvider>,
) -> Option<InstanceRecommendation> {
    recommend_instance(model_params, quant, provider)
        .into_iter()
        .next()
}
