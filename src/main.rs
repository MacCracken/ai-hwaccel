//! CLI binary for ai-hwaccel — outputs the accelerator registry as JSON.
//!
//! Usage:
//!   ai-hwaccel              # Full registry JSON (compact)
//!   ai-hwaccel --pretty     # Full registry JSON (formatted)
//!   ai-hwaccel --table      # Human-readable table
//!   ai-hwaccel --summary    # Compact summary JSON
//!   ai-hwaccel --debug      # Verbose detection diagnostics to stderr
//!   ai-hwaccel --version    # Print version
//!
//! Logging:
//!   Set `RUST_LOG` to control verbosity (e.g. `RUST_LOG=debug ai-hwaccel`).
//!   Use `--json-log` to emit structured JSON logs to stderr.

use tracing::{error, info, warn};
use tracing_subscriber::{EnvFilter, fmt};

use ai_hwaccel::{AcceleratorFamily, AcceleratorRegistry};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);

    let debug_mode = has("--debug") || has("-d");
    let json_log = has("--json-log");
    init_logging(json_log, debug_mode);

    if has("--version") || has("-V") {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return;
    }

    info!("starting accelerator detection");
    let registry = AcceleratorRegistry::detect();
    info!(
        device_count = registry.all_profiles().len(),
        has_accelerator = registry.has_accelerator(),
        warnings = registry.warnings().len(),
        "detection complete"
    );

    for w in registry.warnings() {
        warn!("{}", w);
    }

    let pretty = has("--pretty") || has("-p");

    if has("--table") || has("-t") {
        print_table(&registry);
    } else if has("--summary") || has("-s") {
        let summary = build_summary(&registry);
        emit_json(&summary, pretty);
    } else {
        emit_json(&registry, pretty);
    }
}

fn print_table(registry: &AcceleratorRegistry) {
    println!(
        "{:<6} {:<35} {:>10} {:>8} {:>12}",
        "ID", "Device", "Memory", "Family", "Status"
    );
    println!("{}", "-".repeat(75));

    for (i, p) in registry.all_profiles().iter().enumerate() {
        let mem_gb = p.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let family = p.accelerator.family();
        let status = if p.available { "ok" } else { "unavail" };
        println!(
            "{:<6} {:<35} {:>7.1} GB {:>8} {:>12}",
            i,
            truncate(&p.accelerator.to_string(), 35),
            mem_gb,
            family,
            status,
        );
    }

    println!();
    println!(
        "Total: {} device(s), {:.1} GB system, {:.1} GB accelerator",
        registry.all_profiles().len(),
        registry.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
        registry.total_accelerator_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    if let Some(best) = registry.best_available() {
        println!("Best:  {}", best);
    }

    let families = [
        AcceleratorFamily::Gpu,
        AcceleratorFamily::Tpu,
        AcceleratorFamily::Npu,
        AcceleratorFamily::AiAsic,
    ];
    let counts: Vec<String> = families
        .iter()
        .filter_map(|f| {
            let n = registry.by_family(*f).len();
            if n > 0 {
                Some(format!("{} {}", n, f))
            } else {
                None
            }
        })
        .collect();
    if !counts.is_empty() {
        println!("       {}", counts.join(", "));
    }

    if !registry.warnings().is_empty() {
        println!();
        println!("Warnings:");
        for w in registry.warnings() {
            println!("  {}", w);
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}

fn emit_json<T: serde::Serialize>(value: &T, pretty: bool) {
    let result = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    };
    match result {
        Ok(json) => println!("{json}"),
        Err(e) => {
            error!(%e, "serialisation error");
            std::process::exit(1);
        }
    }
}

fn init_logging(json: bool, debug_mode: bool) {
    let filter = if debug_mode {
        EnvFilter::new("debug")
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
    };

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
        "schema_version": 1,
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
        },
        "warnings": registry.warnings().iter().map(|w| w.to_string()).collect::<Vec<_>>(),
    })
}
