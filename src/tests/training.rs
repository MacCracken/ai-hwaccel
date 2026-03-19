//! Training memory estimation tests.

use crate::*;

// ---------------------------------------------------------------------------
// Training memory: cross-target comparisons
// ---------------------------------------------------------------------------

#[test]
fn training_memory_tpu_less_optimizer_than_gpu() {
    let gpu = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Gpu);
    let tpu = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Tpu);
    assert!(tpu.optimizer_gb < gpu.optimizer_gb);
}

#[test]
fn training_memory_qlora_less_than_full() {
    let full = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Gpu);
    let qlora =
        estimate_training_memory(7000, TrainingMethod::QLoRA { bits: 4 }, TrainingTarget::Gpu);
    assert!(qlora.total_gb < full.total_gb);
    assert!(qlora.model_gb < full.model_gb);
}

#[test]
fn training_memory_lora_less_than_full() {
    let full = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Gpu);
    let lora = estimate_training_memory(7000, TrainingMethod::LoRA, TrainingTarget::Gpu);
    assert!(lora.total_gb < full.total_gb);
    assert!(lora.optimizer_gb < full.optimizer_gb);
}

#[test]
fn training_memory_gaudi() {
    let est = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Gaudi);
    assert!(est.total_gb > 30.0);
    assert!(est.total_gb < 60.0);
    assert!(est.model_gb > 0.0);
    assert!(est.optimizer_gb > 0.0);
    assert!(est.activation_gb > 0.0);
}

// ---------------------------------------------------------------------------
// All training methods produce positive estimates
// ---------------------------------------------------------------------------

#[test]
fn training_memory_all_methods_positive() {
    let methods = [
        TrainingMethod::FullFineTune,
        TrainingMethod::LoRA,
        TrainingMethod::QLoRA { bits: 4 },
        TrainingMethod::QLoRA { bits: 8 },
        TrainingMethod::Prefix,
        TrainingMethod::DPO,
        TrainingMethod::RLHF,
        TrainingMethod::Distillation,
    ];
    let targets = [
        TrainingTarget::Gpu,
        TrainingTarget::Tpu,
        TrainingTarget::Gaudi,
        TrainingTarget::Cpu,
    ];
    for method in &methods {
        for target in &targets {
            let est = estimate_training_memory(7000, *method, *target);
            assert!(
                est.total_gb > 0.0,
                "{:?} on {:?} should have positive total",
                method,
                target
            );
            assert!(
                (est.model_gb + est.optimizer_gb + est.activation_gb - est.total_gb).abs() < 0.001,
                "{:?} on {:?}: component sum doesn't match total",
                method,
                target
            );
        }
    }
}

// ---------------------------------------------------------------------------
// DPO/RLHF need more memory than LoRA (two model copies)
// ---------------------------------------------------------------------------

#[test]
fn training_memory_dpo_more_than_lora() {
    let lora = estimate_training_memory(7000, TrainingMethod::LoRA, TrainingTarget::Gpu);
    let dpo = estimate_training_memory(7000, TrainingMethod::DPO, TrainingTarget::Gpu);
    assert!(dpo.total_gb > lora.total_gb);
}

#[test]
fn training_memory_rlhf_more_than_lora() {
    let lora = estimate_training_memory(7000, TrainingMethod::LoRA, TrainingTarget::Gpu);
    let rlhf = estimate_training_memory(7000, TrainingMethod::RLHF, TrainingTarget::Gpu);
    assert!(rlhf.total_gb > lora.total_gb);
}

#[test]
fn training_memory_distillation() {
    let full = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Gpu);
    let dist = estimate_training_memory(7000, TrainingMethod::Distillation, TrainingTarget::Gpu);
    // Distillation needs teacher + student, more model memory
    assert!(dist.model_gb > full.model_gb);
}

#[test]
fn training_memory_prefix_is_lightweight() {
    let full = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Gpu);
    let prefix = estimate_training_memory(7000, TrainingMethod::Prefix, TrainingTarget::Gpu);
    assert!(prefix.optimizer_gb < full.optimizer_gb);
    assert!(prefix.total_gb < full.total_gb);
}

// ---------------------------------------------------------------------------
// QLoRA 4-bit vs 8-bit
// ---------------------------------------------------------------------------

#[test]
fn training_memory_qlora_4bit_less_than_8bit() {
    let q4 = estimate_training_memory(7000, TrainingMethod::QLoRA { bits: 4 }, TrainingTarget::Gpu);
    let q8 = estimate_training_memory(7000, TrainingMethod::QLoRA { bits: 8 }, TrainingTarget::Gpu);
    assert!(q4.model_gb < q8.model_gb);
}

// ---------------------------------------------------------------------------
// TrainingMethod
// ---------------------------------------------------------------------------

#[test]
fn training_method_display() {
    assert_eq!(TrainingMethod::LoRA.to_string(), "lora");
    assert_eq!(TrainingMethod::QLoRA { bits: 4 }.to_string(), "qlora-4bit");
    assert_eq!(TrainingMethod::FullFineTune.to_string(), "full");
    assert_eq!(TrainingMethod::DPO.to_string(), "dpo");
    assert_eq!(TrainingMethod::RLHF.to_string(), "rlhf");
    assert_eq!(TrainingMethod::Distillation.to_string(), "distillation");
    assert_eq!(TrainingMethod::Prefix.to_string(), "prefix");
}

#[test]
fn training_method_preferred_accelerator() {
    assert_eq!(
        TrainingMethod::LoRA.preferred_accelerator(),
        AcceleratorRequirement::Gpu
    );
    assert_eq!(
        TrainingMethod::QLoRA { bits: 4 }.preferred_accelerator(),
        AcceleratorRequirement::Gpu
    );
    assert_eq!(
        TrainingMethod::FullFineTune.preferred_accelerator(),
        AcceleratorRequirement::GpuOrTpu
    );
    assert_eq!(
        TrainingMethod::DPO.preferred_accelerator(),
        AcceleratorRequirement::GpuOrTpu
    );
    assert_eq!(
        TrainingMethod::RLHF.preferred_accelerator(),
        AcceleratorRequirement::GpuOrTpu
    );
    assert_eq!(
        TrainingMethod::Distillation.preferred_accelerator(),
        AcceleratorRequirement::GpuOrTpu
    );
    assert_eq!(
        TrainingMethod::Prefix.preferred_accelerator(),
        AcceleratorRequirement::GpuOrTpu
    );
}
