//! Model sharding strategies and plans.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::hardware::AcceleratorType;

/// Strategy for distributing a model across devices.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// No sharding — run on a single device.
    None,
    /// Split layers across devices in a pipeline.
    PipelineParallel { num_stages: u32 },
    /// Split individual tensors across devices.
    TensorParallel { num_devices: u32 },
    /// Replicate the full model for higher throughput.
    DataParallel { num_replicas: u32 },
}

impl ShardingStrategy {
    /// Minimum number of devices required.
    pub fn min_devices(&self) -> u32 {
        match self {
            Self::None => 1,
            Self::PipelineParallel { num_stages } => *num_stages,
            Self::TensorParallel { num_devices } => *num_devices,
            Self::DataParallel { num_replicas } => *num_replicas,
        }
    }
}

impl fmt::Display for ShardingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::PipelineParallel { num_stages } => {
                write!(f, "Pipeline Parallel ({} stages)", num_stages)
            }
            Self::TensorParallel { num_devices } => {
                write!(f, "Tensor Parallel ({} devices)", num_devices)
            }
            Self::DataParallel { num_replicas } => {
                write!(f, "Data Parallel ({} replicas)", num_replicas)
            }
        }
    }
}

/// A slice of model layers assigned to a specific device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelShard {
    pub shard_id: u32,
    /// Inclusive layer range `(start, end)`.
    pub layer_range: (u32, u32),
    pub device: AcceleratorType,
    /// Estimated memory consumption in bytes.
    pub memory_bytes: u64,
}

impl ModelShard {
    /// Number of layers in this shard.
    pub fn num_layers(&self) -> u32 {
        if self.layer_range.1 >= self.layer_range.0 {
            self.layer_range.1 - self.layer_range.0 + 1
        } else {
            0
        }
    }

    /// Whether the layer range is valid.
    pub fn is_valid(&self) -> bool {
        self.layer_range.0 <= self.layer_range.1
    }
}

/// A concrete plan for distributing model shards across devices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardingPlan {
    pub shards: Vec<ModelShard>,
    pub strategy: ShardingStrategy,
    pub total_memory_bytes: u64,
    pub estimated_tokens_per_sec: Option<f64>,
}

impl fmt::Display for ShardingPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Strategy: {}", self.strategy)?;
        writeln!(
            f,
            "Total memory: {:.1} GB",
            self.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        )?;
        if let Some(tps) = self.estimated_tokens_per_sec {
            writeln!(f, "Est. throughput: {:.0} tok/s", tps)?;
        }
        if self.shards.len() > 1 {
            writeln!(f, "Shards:")?;
            for shard in &self.shards {
                writeln!(
                    f,
                    "  [{}] {} — layers {}-{} ({:.1} GB)",
                    shard.shard_id,
                    shard.device,
                    shard.layer_range.0,
                    shard.layer_range.1,
                    shard.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                )?;
            }
        } else if let Some(shard) = self.shards.first() {
            writeln!(f, "Device: {}", shard.device)?;
        }
        Ok(())
    }
}
