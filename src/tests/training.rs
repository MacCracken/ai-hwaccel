//! Training memory estimation tests.

use crate::*;

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
}

#[test]
fn training_memory_gaudi() {
    let est = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Gaudi);
    assert!(est.total_gb > 30.0);
    assert!(est.total_gb < 60.0);
}

#[test]
fn training_method_display() {
    assert_eq!(TrainingMethod::LoRA.to_string(), "lora");
    assert_eq!(TrainingMethod::QLoRA { bits: 4 }.to_string(), "qlora-4bit");
    assert_eq!(TrainingMethod::FullFineTune.to_string(), "full");
    assert_eq!(TrainingMethod::DPO.to_string(), "dpo");
    assert_eq!(TrainingMethod::RLHF.to_string(), "rlhf");
    assert_eq!(TrainingMethod::Distillation.to_string(), "distillation");
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
}
