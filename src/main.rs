//! CLI binary for ai-hwaccel — outputs the accelerator registry as JSON.
//!
//! Usage:
//!   ai-hwaccel              # Full registry JSON
//!   ai-hwaccel --summary    # Compact summary (available devices, memory, best device)
//!   ai-hwaccel --version    # Print version
//!
//! Logging:
//!   Set `RUST_LOG` to control verbosity (e.g. `RUST_LOG=debug ai-hwaccel`).
//!   Use `--json-log` to emit structured JSON logs to stderr.

use tracing::{error, info};
use tracing_subscriber::{EnvFilter, fmt};

use ai_hwaccel::AcceleratorRegistry;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let json_log = args.iter().any(|a| a == "--json-log");
    init_logging(json_log);

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return;
    }

    info!("starting accelerator detection");
    let registry = AcceleratorRegistry::detect();
    info!(
        device_count = registry.all_profiles().len(),
        has_accelerator = registry.has_accelerator(),
        "detection complete"
    );

    if args.iter().any(|a| a == "--summary" || a == "-s") {
        let summary = build_summary(&registry);
        match serde_json::to_string(&summary) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                error!(%e, "serialisation error");
                std::process::exit(1);
            }
        }
    } else {
        match serde_json::to_string(&registry) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                error!(%e, "serialisation error");
                std::process::exit(1);
            }
        }
    }
}

fn init_logging(json: bool) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));

    if json {
        fmt()
            .with_env_filter(filter)
            .json()
            .with_writer(std::io::stderr)
            .init();
    } else {
        fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .init();
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
