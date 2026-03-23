//! Sharding and workload planning logic.

use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;
use crate::quantization::QuantizationLevel;
use crate::registry::AcceleratorRegistry;
use crate::sharding::{ModelShard, ShardingPlan, ShardingStrategy};
use crate::system_io::{Interconnect, InterconnectKind};
use crate::units;

/// Aggregated interconnect bandwidth data extracted in a single pass.
#[allow(dead_code)]
struct InterconnectInfo {
    has_nvswitch: bool,
    nvlink_bw: f64,
    xgmi_bw: f64,
    high_bw: f64,
}

impl InterconnectInfo {
    /// Single-pass scan over interconnects to extract all needed values.
    fn scan(interconnects: &[Interconnect]) -> Self {
        let mut has_nvswitch = false;
        let mut nvlink_bw = 0.0f64;
        let mut xgmi_bw = 0.0f64;
        for ic in interconnects {
            match ic.kind {
                InterconnectKind::NVSwitch => {
                    has_nvswitch = true;
                    nvlink_bw = nvlink_bw.max(ic.bandwidth_gbps);
                }
                InterconnectKind::NVLink => {
                    nvlink_bw = nvlink_bw.max(ic.bandwidth_gbps);
                }
                InterconnectKind::XgmiInfinityFabric => {
                    xgmi_bw = xgmi_bw.max(ic.bandwidth_gbps);
                }
                _ => {}
            }
        }
        Self {
            has_nvswitch,
            nvlink_bw,
            xgmi_bw,
            high_bw: nvlink_bw + xgmi_bw,
        }
    }
}

/// Case 2: TPU tensor parallel (ICI mesh has high inter-chip bandwidth).
fn build_tpu_tensor_plan(
    tpu_devices: &[&AcceleratorProfile],
    tpu_chips: u32,
    tpu_min_mult: f64,
    needed: u64,
    quant: &QuantizationLevel,
) -> ShardingPlan {
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
    let tps = tpu_min_mult * tpu_chips as f64 * quant_factor * units::TPU_TP_ICI_BONUS;

    ShardingPlan {
        shards,
        strategy: ShardingStrategy::TensorParallel {
            num_devices: tpu_chips,
        },
        total_memory_bytes: needed,
        estimated_tokens_per_sec: Some(tps),
    }
}

/// Case 3a: GPU tensor parallel with NVSwitch or high-bandwidth interconnect.
fn build_gpu_tensor_plan(
    gpu_devices: &[&AcceleratorProfile],
    ic: &InterconnectInfo,
    needed: u64,
    quant: &QuantizationLevel,
) -> ShardingPlan {
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
    let ic_bonus = if ic.has_nvswitch {
        units::NVSWITCH_TP_BONUS
    } else {
        1.0 + (ic.high_bw / units::TP_INTERCONNECT_BW_DIVISOR).min(units::MAX_NON_NVSWITCH_TP_BONUS)
    };
    let tps = slowest * num_devices as f64 * quant_factor * ic_bonus;

    ShardingPlan {
        shards,
        strategy: ShardingStrategy::TensorParallel { num_devices },
        total_memory_bytes: needed,
        estimated_tokens_per_sec: Some(tps),
    }
}

/// Case 3b: Pipeline parallel with NUMA-ordered stages.
fn build_pipeline_plan(
    gpu_devices: &[&AcceleratorProfile],
    high_bw: f64,
    needed: u64,
    model_params: u64,
    quant: &QuantizationLevel,
) -> ShardingPlan {
    // Pipeline parallel — order stages by NUMA locality for minimal
    // cross-link transfers. Devices on the same NUMA node should be
    // adjacent pipeline stages.
    let mut ordered_devices = gpu_devices.to_vec();
    ordered_devices.sort_by_key(|d| d.numa_node.unwrap_or(u32::MAX));

    let num_stages = ordered_devices.len() as u32;
    let per_shard = needed.div_ceil(num_stages as u64);
    let estimated_layers = (model_params / units::PARAMS_PER_LAYER_ESTIMATE).max(1) as u32;
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
    let ic_factor = if high_bw > 0.0 {
        units::PP_HIGH_BW_EFFICIENCY // ~15% overhead for NVLink/XGMI inter-stage transfers
    } else {
        units::PP_PCIE_ONLY_EFFICIENCY // ~35% overhead for PCIe-only transfers
    };
    let tps = slowest * num_stages as f64 * quant_factor * ic_factor;

    ShardingPlan {
        shards,
        strategy: ShardingStrategy::PipelineParallel { num_stages },
        total_memory_bytes: needed,
        estimated_tokens_per_sec: Some(tps),
    }
}

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
            return build_tpu_tensor_plan(&tpu_devices, tpu_chips, tpu_min_mult, needed, quant);
        }

        // Case 3: GPU / AI ASIC with topology awareness.
        if !gpu_devices.is_empty() && gpu_memory >= needed {
            let ic = InterconnectInfo::scan(&self.system_io.interconnects);

            // NVSwitch or very high NVLink BW (>100 GB/s total) → tensor parallel.
            // Tensor parallel requires all-to-all communication which is only
            // efficient with a full-bisection fabric. The ≤8 device cap is a
            // heuristic: beyond 8 GPUs without NVSwitch, all-reduce collectives
            // hit O(n) scaling on ring topologies. NVSwitch provides full
            // bisection bandwidth regardless of device count, so it bypasses
            // this limit. The 100 GB/s threshold corresponds to ~2 NVLink 4.0
            // connections per direction (50 GB/s each).
            let use_tensor_parallel = ic.has_nvswitch
                || (ic.high_bw > units::TP_MIN_INTERCONNECT_BW
                    && gpu_devices.len() <= units::TP_MAX_DEVICES_WITHOUT_NVSWITCH);

            if use_tensor_parallel {
                return build_gpu_tensor_plan(&gpu_devices, &ic, needed, quant);
            }

            return build_pipeline_plan(&gpu_devices, ic.high_bw, needed, model_params, quant);
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
    let base = units::TOKENS_PER_SEC_BASE / model_params as f64;
    let quant_speedup = quant.memory_reduction_factor();
    base * accel.throughput_multiplier() * quant_speedup
}
