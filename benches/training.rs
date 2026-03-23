//! Benchmarks for training memory estimation.

use criterion::{Criterion, criterion_group, criterion_main};

use ai_hwaccel::*;

fn bench_training_memory_all_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_memory");

    let methods: &[(TrainingMethod, &str)] = &[
        (TrainingMethod::FullFineTune, "full_finetune"),
        (TrainingMethod::LoRA, "lora"),
        (TrainingMethod::QLoRA { bits: 4 }, "qlora_4bit"),
        (TrainingMethod::QLoRA { bits: 8 }, "qlora_8bit"),
        (TrainingMethod::Prefix, "prefix"),
        (TrainingMethod::DPO, "dpo"),
        (TrainingMethod::RLHF, "rlhf"),
        (TrainingMethod::Distillation, "distillation"),
    ];

    for (method, name) in methods {
        group.bench_function(format!("7B_{}_gpu", name), |b| {
            b.iter(|| estimate_training_memory(7000, *method, TrainingTarget::Gpu));
        });
    }

    group.finish();
}

fn bench_training_memory_all_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_targets");

    let targets: &[(TrainingTarget, &str)] = &[
        (TrainingTarget::Gpu, "gpu"),
        (TrainingTarget::Tpu, "tpu"),
        (TrainingTarget::Gaudi, "gaudi"),
        (TrainingTarget::Cpu, "cpu"),
    ];

    for (target, name) in targets {
        group.bench_function(format!("7B_full_{}", name), |b| {
            b.iter(|| estimate_training_memory(7000, TrainingMethod::FullFineTune, *target));
        });
    }

    group.finish();
}

fn bench_training_memory_model_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_model_sizes");

    let sizes: &[(u64, &str)] = &[
        (1000, "1B"),
        (7000, "7B"),
        (13000, "13B"),
        (70000, "70B"),
        (405000, "405B"),
    ];

    for (params_m, name) in sizes {
        group.bench_function(format!("{}_lora_gpu", name), |b| {
            b.iter(|| {
                estimate_training_memory(*params_m, TrainingMethod::LoRA, TrainingTarget::Gpu)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_training_memory_all_methods,
    bench_training_memory_all_targets,
    bench_training_memory_model_sizes,
);
criterion_main!(benches);
