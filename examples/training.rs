//! Estimate training memory for different methods and targets.
//!
//! ```sh
//! cargo run --example training
//! ```

use ai_hwaccel::{TrainingMethod, TrainingTarget, estimate_training_memory};

fn main() {
    let model_params_m: u64 = 7000; // 7B

    println!("Training memory estimates for a 7B-parameter model:\n");

    let methods = [
        TrainingMethod::FullFineTune,
        TrainingMethod::LoRA,
        TrainingMethod::QLoRA { bits: 4 },
        TrainingMethod::Prefix,
        TrainingMethod::DPO,
        TrainingMethod::RLHF,
        TrainingMethod::Distillation,
    ];

    let targets = [
        ("GPU", TrainingTarget::Gpu),
        ("TPU", TrainingTarget::Tpu),
        ("Gaudi", TrainingTarget::Gaudi),
    ];

    // Header
    print!("{:<16}", "Method");
    for (name, _) in &targets {
        print!("{:>12}", name);
    }
    println!();
    println!("{}", "-".repeat(52));

    for method in &methods {
        print!("{:<16}", method.to_string());
        for (_, target) in &targets {
            let est = estimate_training_memory(model_params_m, *method, *target);
            print!("{:>10.1} GB", est.total_gb);
        }
        println!();
    }
}
