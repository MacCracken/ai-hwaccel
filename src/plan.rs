//! Sharding and workload planning logic.

use crate::detect::AcceleratorRegistry;
use crate::types::*;

impl AcceleratorRegistry {
    /// Generate a sharding plan for a model given its parameter count and quantisation.
    ///
    /// Strategy selection:
    /// 1. If model fits on the best single device → `None` (no sharding).
    /// 2. If TPU pod slice available → `TensorParallel` (ICI mesh favours this).
    /// 3. If multiple GPUs/ASICs available → `PipelineParallel`.
    /// 4. Fallback → CPU with `None`.
    pub fn plan_sharding(&self, model_params: u64, quant: &QuantizationLevel) -> ShardingPlan {
        let needed = Self::estimate_memory(model_params, quant);

        let best = match self.best_available() {
            Some(b) => b,
            None => {
                return ShardingPlan {
                    shards: vec![],
                    strategy: ShardingStrategy::DataParallel { num_replicas: 0 },
                    total_memory_bytes: 0,
                    estimated_tokens_per_sec: None,
                };
            }
        };

        // Case 1: fits on a single best device.
        if needed <= best.memory_bytes {
            let tps = estimate_tokens_per_sec(&best.accelerator, model_params, quant);
            return ShardingPlan {
                shards: vec![ModelShard {
                    shard_id: 0,
                    layer_range: (0, 0),
                    device: best.accelerator.clone(),
                    memory_bytes: needed,
                }],
                strategy: ShardingStrategy::None,
                total_memory_bytes: needed,
                estimated_tokens_per_sec: Some(tps),
            };
        }

        // Case 2: TPU tensor parallel (ICI mesh has high inter-chip bandwidth).
        let tpu_devices: Vec<&AcceleratorProfile> = self
            .all_profiles()
            .iter()
            .filter(|p| p.available && p.accelerator.is_tpu())
            .collect();
        let tpu_memory: u64 = tpu_devices.iter().map(|p| p.memory_bytes).sum();

        if !tpu_devices.is_empty() && tpu_memory >= needed {
            let total_chips: u32 = tpu_devices
                .iter()
                .filter_map(|p| match &p.accelerator {
                    AcceleratorType::Tpu { chip_count, .. } => Some(*chip_count),
                    _ => None,
                })
                .sum();
            let per_chip = needed / total_chips.max(1) as u64;

            let shards: Vec<ModelShard> = (0..total_chips)
                .map(|i| ModelShard {
                    shard_id: i,
                    layer_range: (0, 0), // tensor parallel: all chips see all layers
                    device: tpu_devices[0].accelerator.clone(),
                    memory_bytes: per_chip,
                })
                .collect();

            let tpu_multiplier = tpu_devices
                .iter()
                .map(|d| d.accelerator.throughput_multiplier())
                .fold(f64::INFINITY, f64::min);
            let tps =
                tpu_multiplier * total_chips as f64 * 8.0 / (quant.bits_per_param() as f64 / 4.0);

            return ShardingPlan {
                shards,
                strategy: ShardingStrategy::TensorParallel {
                    num_devices: total_chips,
                },
                total_memory_bytes: needed,
                estimated_tokens_per_sec: Some(tps),
            };
        }

        // Case 3: GPU / AI ASIC pipeline parallel.
        let gpu_devices: Vec<&AcceleratorProfile> = self
            .all_profiles()
            .iter()
            .filter(|p| {
                p.available
                    && (p.accelerator.is_gpu()
                        || p.accelerator.is_ai_asic()
                        || p.accelerator.is_tpu())
            })
            .collect();
        let gpu_memory: u64 = gpu_devices.iter().map(|p| p.memory_bytes).sum();

        if !gpu_devices.is_empty() && gpu_memory >= needed {
            let num_stages = gpu_devices.len() as u32;
            let per_shard = needed / num_stages as u64;
            let estimated_layers = (model_params / 250_000_000).max(1) as u32;
            let layers_per_shard = (estimated_layers / num_stages).max(1);

            let shards: Vec<ModelShard> = gpu_devices
                .iter()
                .enumerate()
                .map(|(i, dev)| {
                    let start = i as u32 * layers_per_shard;
                    let end = start + layers_per_shard - 1;
                    ModelShard {
                        shard_id: i as u32,
                        layer_range: (start, end),
                        device: dev.accelerator.clone(),
                        memory_bytes: per_shard,
                    }
                })
                .collect();

            let slowest = gpu_devices
                .iter()
                .map(|d| d.accelerator.throughput_multiplier())
                .fold(f64::INFINITY, f64::min);
            let tps = slowest * 10.0 / (quant.bits_per_param() as f64 / 4.0);

            return ShardingPlan {
                shards,
                strategy: ShardingStrategy::PipelineParallel { num_stages },
                total_memory_bytes: needed,
                estimated_tokens_per_sec: Some(tps),
            };
        }

        // Case 4: CPU fallback.
        let tps = estimate_tokens_per_sec(&AcceleratorType::Cpu, model_params, quant);
        ShardingPlan {
            shards: vec![ModelShard {
                shard_id: 0,
                layer_range: (0, 0),
                device: AcceleratorType::Cpu,
                memory_bytes: needed,
            }],
            strategy: ShardingStrategy::None,
            total_memory_bytes: needed,
            estimated_tokens_per_sec: Some(tps),
        }
    }
}

impl ShardingPlan {
    /// Whether the plan fits within the total available memory.
    pub fn fits_in_memory(&self, registry: &AcceleratorRegistry) -> bool {
        self.total_memory_bytes <= registry.total_memory()
    }
}

/// Rough tokens/sec estimate based on device and model size.
fn estimate_tokens_per_sec(
    accel: &AcceleratorType,
    model_params: u64,
    quant: &QuantizationLevel,
) -> f64 {
    let base = 1_000_000_000.0 / model_params as f64;
    let quant_speedup = quant.memory_reduction_factor();
    base * accel.throughput_multiplier() * quant_speedup
}

/// Estimate device memory for a fine-tuning job on a specific accelerator family.
pub fn estimate_training_memory(
    model_params_millions: u64,
    method: TrainingMethod,
    target: TrainingTarget,
) -> MemoryEstimate {
    let base_gb = (model_params_millions as f64 * 1_000_000.0 * 2.0) / 1_073_741_824.0;

    match target {
        TrainingTarget::Tpu => estimate_tpu_training(base_gb, method),
        TrainingTarget::Gaudi => estimate_gaudi_training(base_gb, method),
        TrainingTarget::Gpu | TrainingTarget::Cpu => estimate_gpu_training(base_gb, method),
    }
}

fn estimate_gpu_training(base_gb: f64, method: TrainingMethod) -> MemoryEstimate {
    let (model, optimizer, activation) = match method {
        TrainingMethod::FullFineTune => (base_gb, base_gb * 2.0, base_gb * 1.0),
        TrainingMethod::LoRA => (base_gb, base_gb * 0.1, base_gb * 0.1),
        TrainingMethod::QLoRA { bits } => {
            let qf = if bits <= 4 { 0.25 } else { 0.5 };
            (base_gb * qf, base_gb * 0.1, base_gb * 0.1 * qf)
        }
        TrainingMethod::Prefix => (base_gb, base_gb * 0.05, base_gb * 0.05),
        TrainingMethod::DPO | TrainingMethod::RLHF => {
            // DPO/RLHF: two model copies + optimizer
            (base_gb * 2.0, base_gb * 2.0, base_gb * 1.5)
        }
        TrainingMethod::Distillation => {
            // Teacher + student
            (base_gb * 1.5, base_gb * 1.0, base_gb * 0.8)
        }
    };
    MemoryEstimate {
        model_gb: model,
        optimizer_gb: optimizer,
        activation_gb: activation,
        total_gb: model + optimizer + activation,
    }
}

fn estimate_tpu_training(base_gb: f64, method: TrainingMethod) -> MemoryEstimate {
    // TPU: BF16 native, XLA fuses activations, BF16 optimizer states (1.5x not 2x)
    let (model, optimizer, activation) = match method {
        TrainingMethod::FullFineTune => (base_gb, base_gb * 1.5, base_gb * 0.8),
        TrainingMethod::LoRA => (base_gb, base_gb * 0.15, base_gb * 0.12),
        TrainingMethod::QLoRA { bits } => {
            let qf = if bits <= 4 { 0.4 } else { 0.6 };
            (base_gb * qf, base_gb * 0.15, base_gb * 0.12 * qf)
        }
        TrainingMethod::Prefix => (base_gb, base_gb * 0.05, base_gb * 0.05),
        TrainingMethod::DPO | TrainingMethod::RLHF => (base_gb * 2.0, base_gb * 1.5, base_gb * 1.2),
        TrainingMethod::Distillation => (base_gb * 1.5, base_gb * 0.8, base_gb * 0.6),
    };
    MemoryEstimate {
        model_gb: model,
        optimizer_gb: optimizer,
        activation_gb: activation,
        total_gb: model + optimizer + activation,
    }
}

fn estimate_gaudi_training(base_gb: f64, method: TrainingMethod) -> MemoryEstimate {
    // Gaudi: BF16 native like TPU, but with different memory controller.
    // Similar to TPU estimates but with slightly higher activation overhead.
    let (model, optimizer, activation) = match method {
        TrainingMethod::FullFineTune => (base_gb, base_gb * 1.5, base_gb * 0.9),
        TrainingMethod::LoRA => (base_gb, base_gb * 0.12, base_gb * 0.12),
        TrainingMethod::QLoRA { bits } => {
            let qf = if bits <= 4 { 0.35 } else { 0.55 };
            (base_gb * qf, base_gb * 0.12, base_gb * 0.12 * qf)
        }
        TrainingMethod::Prefix => (base_gb, base_gb * 0.05, base_gb * 0.06),
        TrainingMethod::DPO | TrainingMethod::RLHF => (base_gb * 2.0, base_gb * 1.5, base_gb * 1.3),
        TrainingMethod::Distillation => (base_gb * 1.5, base_gb * 0.9, base_gb * 0.7),
    };
    MemoryEstimate {
        model_gb: model,
        optimizer_gb: optimizer,
        activation_gb: activation,
        total_gb: model + optimizer + activation,
    }
}

// ---------------------------------------------------------------------------
// Training types (shared across ecosystem)
// ---------------------------------------------------------------------------

/// Fine-tuning / training method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrainingMethod {
    FullFineTune,
    LoRA,
    QLoRA { bits: u8 },
    Prefix,
    DPO,
    RLHF,
    Distillation,
}

impl TrainingMethod {
    /// Preferred accelerator requirement for this training method.
    pub fn preferred_accelerator(&self) -> AcceleratorRequirement {
        match self {
            // LoRA/QLoRA benefit from custom CUDA kernels
            Self::LoRA | Self::QLoRA { .. } => AcceleratorRequirement::Gpu,
            // Full fine-tune, DPO, RLHF, distillation work well on GPU or TPU
            Self::FullFineTune | Self::DPO | Self::RLHF | Self::Distillation | Self::Prefix => {
                AcceleratorRequirement::GpuOrTpu
            }
        }
    }
}

impl std::fmt::Display for TrainingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FullFineTune => write!(f, "full"),
            Self::LoRA => write!(f, "lora"),
            Self::QLoRA { bits } => write!(f, "qlora-{}bit", bits),
            Self::Prefix => write!(f, "prefix"),
            Self::DPO => write!(f, "dpo"),
            Self::RLHF => write!(f, "rlhf"),
            Self::Distillation => write!(f, "distillation"),
        }
    }
}

/// Target accelerator family for training memory estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingTarget {
    Gpu,
    Tpu,
    Gaudi,
    Cpu,
}
