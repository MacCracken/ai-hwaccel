//! Sharding and workload planning logic.

use crate::hardware::AcceleratorType;
use crate::quantization::QuantizationLevel;
use crate::registry::AcceleratorRegistry;
use crate::sharding::{ModelShard, ShardingPlan, ShardingStrategy};

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
        let tpu_devices: Vec<_> = self
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
            // Ceiling division so no bytes are unaccounted for.
            let chips = total_chips.max(1) as u64;
            let per_chip = needed.div_ceil(chips);

            let shards: Vec<ModelShard> = (0..total_chips)
                .map(|i| ModelShard {
                    shard_id: i,
                    layer_range: (0, 0), // tensor parallel: all chips see all layers
                    device: tpu_devices[0].accelerator.clone(),
                    memory_bytes: per_chip,
                })
                .collect();

            // Throughput estimate: base multiplier * chip count, scaled by
            // quantisation memory reduction factor (e.g. FP16 = 2x, INT4 = 8x).
            let tpu_multiplier = tpu_devices
                .iter()
                .map(|d| d.accelerator.throughput_multiplier())
                .fold(f64::INFINITY, f64::min);
            let quant_factor = quant.memory_reduction_factor();
            let tps = tpu_multiplier * total_chips as f64 * quant_factor * 2.0;

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
        let gpu_devices: Vec<_> = self
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
            let per_shard = needed.div_ceil(num_stages as u64);
            let estimated_layers = (model_params / 250_000_000).max(1) as u32;
            let layers_per_shard = estimated_layers.div_ceil(num_stages).max(1);

            let shards: Vec<ModelShard> = gpu_devices
                .iter()
                .enumerate()
                .map(|(i, dev)| {
                    let start = i as u32 * layers_per_shard;
                    // Last shard captures all remaining layers.
                    let end = if i as u32 == num_stages - 1 {
                        estimated_layers.saturating_sub(1)
                    } else {
                        start + layers_per_shard - 1
                    };
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
            let quant_factor = quant.memory_reduction_factor();
            let tps = slowest * quant_factor * 2.5;

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
    if model_params == 0 {
        return 0.0;
    }
    let base = 1_000_000_000.0 / model_params as f64;
    let quant_speedup = quant.memory_reduction_factor();
    base * accel.throughput_multiplier() * quant_speedup
}
