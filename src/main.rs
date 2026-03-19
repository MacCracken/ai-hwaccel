//! CLI binary for ai-hwaccel — outputs the accelerator registry as JSON.
//!
//! Usage:
//!   ai-hwaccel              # Full registry JSON
//!   ai-hwaccel --summary    # Compact summary (available devices, memory, best device)
//!   ai-hwaccel --version    # Print version

use ai_hwaccel::AcceleratorRegistry;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return;
    }

    let registry = AcceleratorRegistry::detect();

    if args.iter().any(|a| a == "--summary" || a == "-s") {
        let summary = build_summary(&registry);
        match serde_json::to_string(&summary) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("serialisation error: {e}");
                std::process::exit(1);
            }
        }
    } else {
        match serde_json::to_string(&registry) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("serialisation error: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn build_summary(registry: &AcceleratorRegistry) -> serde_json::Value {
    let available = registry.available();
    let best = registry.best_available();
    let total_memory = registry.total_memory();
    let accel_memory = registry.total_accelerator_memory();

    serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "device_count": available.len(),
        "has_accelerator": registry.has_accelerator(),
        "total_memory_bytes": total_memory,
        "accelerator_memory_bytes": accel_memory,
        "best_device": best.map(|b| serde_json::json!({
            "accelerator": b.accelerator,
            "memory_bytes": b.memory_bytes,
            "compute_capability": b.compute_capability,
            "driver_version": b.driver_version,
        })),
        "families": {
            "gpu": registry.by_family(ai_hwaccel::AcceleratorFamily::Gpu).len(),
            "tpu": registry.by_family(ai_hwaccel::AcceleratorFamily::Tpu).len(),
            "npu": registry.by_family(ai_hwaccel::AcceleratorFamily::Npu).len(),
            "ai_asic": registry.by_family(ai_hwaccel::AcceleratorFamily::AiAsic).len(),
        }
    })
}
