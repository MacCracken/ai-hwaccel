//! Google TPU generation types.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Google TPU generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TpuVersion {
    V4,
    V5e,
    V5p,
}

impl TpuVersion {
    /// HBM (High Bandwidth Memory) per chip in bytes.
    ///
    /// Sources: Google Cloud TPU documentation (2024).
    /// - V4: 32 GiB HBM2e per chip
    /// - V5e: 16 GiB HBM2e per chip
    /// - V5p: 95 GiB HBM2e per chip
    pub fn hbm_per_chip_bytes(&self) -> u64 {
        match self {
            Self::V4 => 32 * 1024 * 1024 * 1024,
            Self::V5e => 16 * 1024 * 1024 * 1024,
            Self::V5p => 95 * 1024 * 1024 * 1024,
        }
    }
}

impl fmt::Display for TpuVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V4 => write!(f, "v4"),
            Self::V5e => write!(f, "v5e"),
            Self::V5p => write!(f, "v5p"),
        }
    }
}
