//! Plan model sharding across detected hardware.
//!
//! ```sh
//! cargo run --example plan
//! cargo run --example plan -- 70   # 70B params
//! ```

use ai_hwaccel::{AcceleratorRegistry, QuantizationLevel};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_params_b: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let model_params = model_params_b * 1_000_000_000;

    let registry = AcceleratorRegistry::detect();
    let quant = registry.suggest_quantization(model_params);

    println!("Model: {}B parameters", model_params_b);
    println!("Suggested quantization: {}", quant);
    println!();

    let plan = registry.plan_sharding(model_params, &quant);
    print!("{}", plan);

    println!("\nMemory estimates at each quantization level:");
    for q in [
        QuantizationLevel::None,
        QuantizationLevel::Float16,
        QuantizationLevel::BFloat16,
        QuantizationLevel::Int8,
        QuantizationLevel::Int4,
    ] {
        let mem = AcceleratorRegistry::estimate_memory(model_params, &q);
        println!("  {}: {:.1} GB", q, mem as f64 / (1024.0 * 1024.0 * 1024.0));
    }
}
