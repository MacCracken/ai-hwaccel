//! AWS Neuron chip types (Inferentia / Trainium).

use std::fmt;

use serde::{Deserialize, Serialize};

/// AWS Neuron chip type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NeuronChipType {
    /// Inference-optimised (inf1, inf2).
    Inferentia,
    /// Training-optimised (trn1).
    Trainium,
}

impl NeuronChipType {
    /// HBM per NeuronCore in bytes.
    pub fn hbm_per_core_bytes(&self) -> u64 {
        match self {
            // inf2 NeuronCore-v2: 32 GB HBM per accelerator (2 cores share it)
            Self::Inferentia => 16 * 1024 * 1024 * 1024,
            // trn1 NeuronCore-v2: 32 GB HBM per accelerator
            Self::Trainium => 32 * 1024 * 1024 * 1024,
        }
    }
}

impl fmt::Display for NeuronChipType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inferentia => write!(f, "Inferentia"),
            Self::Trainium => write!(f, "Trainium"),
        }
    }
}
