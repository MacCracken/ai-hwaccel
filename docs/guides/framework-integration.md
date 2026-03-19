# Framework Integration Guide

`ai-hwaccel` detects hardware and plans model deployment. It does **not** run
inference or training — that's the job of your ML framework. This guide shows
how to bridge the gap.

## General pattern

```rust
use ai_hwaccel::{AcceleratorRegistry, AcceleratorFamily, QuantizationLevel};

let registry = AcceleratorRegistry::detect();
let quant = registry.suggest_quantization(model_params);
let plan = registry.plan_sharding(model_params, &quant);

// Use plan.strategy, plan.shards, and device info to configure your framework.
```

---

## candle

[candle](https://github.com/huggingface/candle) selects a device at model load
time. Use `ai-hwaccel` to pick the right one:

```rust,ignore
use ai_hwaccel::{AcceleratorRegistry, AcceleratorType};
use candle_core::Device;

let registry = AcceleratorRegistry::detect();
let best = registry.best_available().unwrap();

let device = match &best.accelerator {
    AcceleratorType::CudaGpu { device_id } => Device::cuda(*device_id as usize)?,
    AcceleratorType::MetalGpu => Device::metal(0)?,
    _ => Device::Cpu,
};
// Load model onto `device`...
```

## burn

[burn](https://github.com/tracel-ai/burn) uses backend types at compile time,
but device selection is runtime. Use `ai-hwaccel` to pick the backend device:

```rust,ignore
use ai_hwaccel::{AcceleratorRegistry, AcceleratorFamily};

let registry = AcceleratorRegistry::detect();

if !registry.by_family(AcceleratorFamily::Gpu).is_empty() {
    // Use burn's WGPU or CUDA backend
    println!("GPU available — use burn-wgpu or burn-cuda");
} else {
    // Fall back to CPU
    println!("CPU only — use burn-ndarray");
}
```

## tch-rs (PyTorch bindings)

[tch-rs](https://github.com/LaurentMazare/tch-rs) wraps libtorch. Use
`ai-hwaccel` to select the CUDA device:

```rust,ignore
use ai_hwaccel::{AcceleratorRegistry, AcceleratorType};

let registry = AcceleratorRegistry::detect();

let device = registry
    .best_available()
    .and_then(|p| match &p.accelerator {
        AcceleratorType::CudaGpu { device_id } => {
            Some(tch::Device::Cuda(*device_id as usize))
        }
        _ => None,
    })
    .unwrap_or(tch::Device::Cpu);
```

## ort (ONNX Runtime)

[ort](https://github.com/pykeio/ort) supports multiple execution providers.
Use `ai-hwaccel` to pick the best one:

```rust,ignore
use ai_hwaccel::{AcceleratorRegistry, AcceleratorFamily};

let registry = AcceleratorRegistry::detect();

let provider = if !registry.by_family(AcceleratorFamily::Gpu).is_empty() {
    "CUDAExecutionProvider" // or "ROCMExecutionProvider" / "CoreMLExecutionProvider"
} else if !registry.by_family(AcceleratorFamily::Npu).is_empty() {
    "QNNExecutionProvider"
} else {
    "CPUExecutionProvider"
};
```

---

## Multi-device sharding

For models that don't fit on a single device, use the sharding plan:

```rust,ignore
use ai_hwaccel::{AcceleratorRegistry, QuantizationLevel, ShardingStrategy};

let registry = AcceleratorRegistry::detect();
let plan = registry.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16);

match &plan.strategy {
    ShardingStrategy::None => {
        // Load entire model on plan.shards[0].device
    }
    ShardingStrategy::PipelineParallel { num_stages } => {
        for shard in &plan.shards {
            // Load layers shard.layer_range on shard.device
        }
    }
    ShardingStrategy::TensorParallel { num_devices } => {
        // Split tensors across devices (framework-specific)
    }
    _ => {}
}
```

## Training memory budgeting

Before launching a fine-tuning job, check if you have enough memory:

```rust,ignore
use ai_hwaccel::*;

let registry = AcceleratorRegistry::detect();
let est = estimate_training_memory(7000, TrainingMethod::LoRA, TrainingTarget::Gpu);

let available_gb = registry.total_accelerator_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
if est.total_gb > available_gb {
    eprintln!("Need {:.1} GB but only {:.1} GB available", est.total_gb, available_gb);
    eprintln!("Try QLoRA: {:.1} GB",
        estimate_training_memory(7000, TrainingMethod::QLoRA { bits: 4 }, TrainingTarget::Gpu).total_gb);
}
```
