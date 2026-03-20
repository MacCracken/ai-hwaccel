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

        // Single pass: collect TPU and GPU/ASIC device groups + totals.
        let mut tpu_devices: Vec<_> = Vec::with_capacity(8);
        let mut tpu_memory: u64 = 0;
        let mut tpu_chips: u32 = 0;
        let mut tpu_min_mult: f64 = f64::INFINITY;
        let mut gpu_devices: Vec<_> = Vec::with_capacity(16);
        let mut gpu_memory: u64 = 0;

        for p in self.all_profiles() {
            if !p.available {
                continue;
            }
            if p.accelerator.is_tpu() {
                tpu_memory += p.memory_bytes;
                tpu_min_mult = tpu_min_mult.min(p.accelerator.throughput_multiplier());
                if let AcceleratorType::Tpu { chip_count, .. } = &p.accelerator {
                    tpu_chips += chip_count;
                }
                tpu_devices.push(p);
            }
            if p.accelerator.is_gpu() || p.accelerator.is_ai_asic() || p.accelerator.is_tpu() {
                gpu_memory += p.memory_bytes;
                gpu_devices.push(p);
            }
        }

        // Case 2: TPU tensor parallel (ICI mesh has high inter-chip bandwidth).
        if !tpu_devices.is_empty() && tpu_memory >= needed {
            let chips = tpu_chips.max(1) as u64;
            let per_chip = needed.div_ceil(chips);

            let shards: Vec<ModelShard> = (0..tpu_chips)
                .map(|i| ModelShard {
                    shard_id: i,
                    layer_range: (0, 0),
                    device: tpu_devices[0].accelerator.clone(),
                    memory_bytes: per_chip,
                })
                .collect();

            let quant_factor = quant.memory_reduction_factor();
            let tps = tpu_min_mult * tpu_chips as f64 * quant_factor * 2.0;

            return ShardingPlan {
                shards,
                strategy: ShardingStrategy::TensorParallel {
                    num_devices: tpu_chips,
                },
                total_memory_bytes: needed,
                estimated_tokens_per_sec: Some(tps),
            };
        }

        // Case 3: GPU / AI ASIC pipeline parallel.

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
