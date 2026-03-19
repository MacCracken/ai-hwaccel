//! Intel Gaudi (Habana Labs HPU) generation types.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Intel Gaudi (Habana Labs HPU) generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GaudiGeneration {
    Gaudi2,
    Gaudi3,
}

impl GaudiGeneration {
    /// HBM per device in bytes.
    pub fn hbm_bytes(&self) -> u64 {
        match self {
            Self::Gaudi2 => 96 * 1024 * 1024 * 1024,
            Self::Gaudi3 => 128 * 1024 * 1024 * 1024,
        }
    }
}

impl fmt::Display for GaudiGeneration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gaudi2 => write!(f, "Gaudi2"),
            Self::Gaudi3 => write!(f, "Gaudi3"),
        }
    }
}
