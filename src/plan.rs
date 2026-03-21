//! Sharding and workload planning logic.

use crate::hardware::AcceleratorType;
use crate::quantization::QuantizationLevel;
use crate::registry::AcceleratorRegistry;
use crate::sharding::{ModelShard, ShardingPlan, ShardingStrategy};
use crate::system_io::InterconnectKind;

impl AcceleratorRegistry {
    /// Generate a sharding plan for a model given its parameter count and quantisation.
    ///
    /// Strategy selection:
    /// 1. If model fits on the best single device → `None` (no sharding).
    /// 2. If TPU pod slice available → `TensorParallel` (ICI mesh favours this).
    /// 3. If NVSwitch or high-bandwidth NVLink connects GPUs → `TensorParallel`.
    /// 4. If multiple GPUs/ASICs available → `PipelineParallel` (topology-ordered).
    /// 5. Fallback → CPU with `None`.
    ///
    /// When interconnect data is available, the planner:
    /// - Prefers tensor parallel for NVSwitch-connected GPU groups.
    /// - Orders pipeline stages to prefer directly-connected GPU pairs (NVLink).
    /// - Adjusts throughput estimates based on interconnect bandwidth.
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

        // Case 3: GPU / AI ASIC with topology awareness.
        if !gpu_devices.is_empty() && gpu_memory >= needed {
            let interconnects = &self.system_io.interconnects;
            let has_nvswitch = interconnects
                .iter()
                .any(|ic| ic.kind == InterconnectKind::NVSwitch);
            let nvlink_bw = interconnects
                .iter()
                .filter(|ic| matches!(ic.kind, InterconnectKind::NVLink | InterconnectKind::NVSwitch))
                .map(|ic| ic.bandwidth_gbps)
                .fold(0.0f64, f64::max);
            let xgmi_bw = interconnects
                .iter()
                .filter(|ic| ic.kind == InterconnectKind::XgmiInfinityFabric)
                .map(|ic| ic.bandwidth_gbps)
                .fold(0.0f64, f64::max);
            let high_bw_interconnect = nvlink_bw + xgmi_bw;

            // NVSwitch or very high NVLink BW (>100 GB/s total) → tensor parallel.
            // Tensor parallel requires all-to-all communication which is only
            // efficient with a full-bisection fabric. The ≤8 device cap is a
            // heuristic: beyond 8 GPUs without NVSwitch, all-reduce collectives
            // hit O(n) scaling on ring topologies. NVSwitch provides full
            // bisection bandwidth regardless of device count, so it bypasses
            // this limit. The 100 GB/s threshold corresponds to ~2 NVLink 4.0
            // connections per direction (50 GB/s each).
            let use_tensor_parallel = has_nvswitch
                || (high_bw_interconnect > 100.0 && gpu_devices.len() <= 8);

            if use_tensor_parallel {
                let num_devices = gpu_devices.len() as u32;
                let per_device = needed.div_ceil(num_devices as u64);

                let shards: Vec<ModelShard> = gpu_devices
                    .iter()
                    .enumerate()
                    .map(|(i, dev)| ModelShard {
                        shard_id: i as u32,
                        layer_range: (0, 0),
                        device: dev.accelerator.clone(),
                        memory_bytes: per_device,
                    })
                    .collect();

                let slowest = gpu_devices
                    .iter()
                    .map(|d| d.accelerator.throughput_multiplier())
                    .fold(f64::INFINITY, f64::min);
                let quant_factor = quant.memory_reduction_factor();
                // Tensor parallel scales better than pipeline with good interconnect.
                // Interconnect bonus: high BW reduces communication overhead.
                let ic_bonus = if has_nvswitch {
                    1.8
                } else {
                    1.0 + (high_bw_interconnect / 200.0).min(0.8)
                };
                let tps = slowest * num_devices as f64 * quant_factor * ic_bonus;

                return ShardingPlan {
                    shards,
                    strategy: ShardingStrategy::TensorParallel { num_devices },
                    total_memory_bytes: needed,
                    estimated_tokens_per_sec: Some(tps),
                };
            }

            // Pipeline parallel — order stages by NUMA locality for minimal
            // cross-link transfers. Devices on the same NUMA node should be
            // adjacent pipeline stages.
            let mut ordered_devices = gpu_devices.clone();
            ordered_devices.sort_by_key(|d| d.numa_node.unwrap_or(u32::MAX));

            let num_stages = ordered_devices.len() as u32;
            let per_shard = needed.div_ceil(num_stages as u64);
            let estimated_layers = (model_params / 250_000_000).max(1) as u32;
            let layers_per_shard = estimated_layers.div_ceil(num_stages).max(1);

            let shards: Vec<ModelShard> = ordered_devices
                .iter()
                .enumerate()
                .map(|(i, dev)| {
                    let start = i as u32 * layers_per_shard;
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

            let slowest = ordered_devices
                .iter()
                .map(|d| d.accelerator.throughput_multiplier())
                .fold(f64::INFINITY, f64::min);
            let quant_factor = quant.memory_reduction_factor();
            // Pipeline parallel throughput: once the pipeline is full, each
            // stage processes a micro-batch concurrently. Steady-state
            // throughput scales with the number of stages (bounded by
            // the slowest stage). Inter-stage communication overhead
            // depends on interconnect bandwidth.
            let ic_factor = if high_bw_interconnect > 0.0 {
                0.85 // ~15% overhead for NVLink/XGMI inter-stage transfers
            } else {
                0.65 // ~35% overhead for PCIe-only transfers
            };
            let tps = slowest * num_stages as f64 * quant_factor * ic_factor;

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
