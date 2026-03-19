//! Model weight quantisation levels.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Model weight quantisation levels.
///
/// # Examples
///
/// ```rust
/// use ai_hwaccel::QuantizationLevel;
///
/// let q = QuantizationLevel::Int8;
/// assert_eq!(q.bits_per_param(), 8);
/// assert!((q.memory_reduction_factor() - 4.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum QuantizationLevel {
    /// Full precision — FP32, 32 bits per parameter.
    None,
    /// Half precision — FP16, 16 bits per parameter.
    Float16,
    /// Brain floating point — BF16, 16 bits per parameter.
    BFloat16,
    /// 8-bit integer quantisation.
    Int8,
    /// 4-bit integer quantisation (GPTQ / AWQ style).
    Int4,
}

impl QuantizationLevel {
    /// Number of bits used per model parameter.
    pub fn bits_per_param(&self) -> u32 {
        match self {
            Self::None => 32,
            Self::Float16 | Self::BFloat16 => 16,
            Self::Int8 => 8,
            Self::Int4 => 4,
        }
    }

    /// Memory reduction factor relative to FP32.
    pub fn memory_reduction_factor(&self) -> f64 {
        32.0 / self.bits_per_param() as f64
    }
}

impl fmt::Display for QuantizationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "FP32"),
            Self::Float16 => write!(f, "FP16"),
            Self::BFloat16 => write!(f, "BF16"),
            Self::Int8 => write!(f, "INT8"),
            Self::Int4 => write!(f, "INT4"),
        }
    }
}
