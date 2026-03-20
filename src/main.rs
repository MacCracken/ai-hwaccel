//! CLI binary for ai-hwaccel — outputs the accelerator registry as JSON.
//!
//! Usage:
//!   ai-hwaccel                     # Full registry JSON (compact)
//!   ai-hwaccel --pretty            # Full registry JSON (formatted)
//!   ai-hwaccel --table             # Human-readable table
//!   ai-hwaccel --table --sort mem  # Table sorted by memory (desc)
//!   ai-hwaccel --table --family gpu  # Table filtered to GPUs only
//!   ai-hwaccel --summary           # Compact summary JSON
//!   ai-hwaccel --watch 5           # Re-detect every 5 seconds (table)
//!   ai-hwaccel --debug             # Verbose detection diagnostics
//!   ai-hwaccel --version           # Print version
//!
//! Logging:
//!   Set `RUST_LOG` to control verbosity (e.g. `RUST_LOG=debug ai-hwaccel`).
//!   Use `--json-log` to emit structured JSON logs to stderr.

use std::time::Duration;

use tracing::{error, info, warn};
use tracing_subscriber::{EnvFilter, fmt};

use ai_hwaccel::{AcceleratorFamily, AcceleratorProfile, AcceleratorRegistry};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let get_val = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };

    let debug_mode = has("--debug") || has("-d");
    let json_log = has("--json-log");
    init_logging(json_log, debug_mode);

    if has("--version") || has("-V") {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return;
    }

    // --watch <seconds>: re-detect on interval
    if let Some(interval_str) = get_val("--watch") {
        let secs: u64 = interval_str.parse().unwrap_or(10);
        let sort = get_val("--sort");
        let family_filter = get_val("--family");
        run_watch(Duration::from_secs(secs), sort, family_filter);
        return;
    }

    info!("starting accelerator detection");
    let registry = AcceleratorRegistry::detect();
    log_detection(&registry);

    let pretty = has("--pretty") || has("-p");
    let sort = get_val("--sort");
    let family_filter = get_val("--family");

    if has("--table") || has("-t") {
        print_table(&registry, sort.as_deref(), family_filter.as_deref());
    } else if has("--summary") || has("-s") {
        let summary = build_summary(&registry);
        emit_json(&summary, pretty);
    } else {
        emit_json(&registry, pretty);
    }
}

fn log_detection(registry: &AcceleratorRegistry) {
    info!(
        device_count = registry.all_profiles().len(),
        has_accelerator = registry.has_accelerator(),
        warnings = registry.warnings().len(),
        "detection complete"
    );
    for w in registry.warnings() {
        warn!("{}", w);
    }
}

fn run_watch(interval: Duration, sort: Option<String>, family: Option<String>) {
    let mut prev_count = 0usize;
    loop {
        // Clear screen
        print!("\x1b[2J\x1b[H");
        let registry = AcceleratorRegistry::detect();
        let count = registry.all_profiles().len();

        print_table(&registry, sort.as_deref(), family.as_deref());

        if prev_count > 0 && count != prev_count {
            println!("\n  [device count changed: {} -> {}]", prev_count, count);
        }
        prev_count = count;

        println!(
            "\nRefreshing every {}s... (Ctrl+C to stop)",
            interval.as_secs()
        );
        std::thread::sleep(interval);
    }
}

fn print_table(registry: &AcceleratorRegistry, sort_by: Option<&str>, family_filter: Option<&str>) {
    let mut profiles: Vec<&AcceleratorProfile> = registry.all_profiles().iter().collect();

    // Filter by family
    if let Some(f) = family_filter {
        let target = parse_family(f);
        if let Some(fam) = target {
            profiles.retain(|p| p.accelerator.family() == fam);
        }
    }

    // Sort
    match sort_by {
        Some("mem" | "memory") => {
            profiles.sort_by(|a, b| b.memory_bytes.cmp(&a.memory_bytes));
        }
        Some("name" | "device") => {
            profiles.sort_by(|a, b| a.accelerator.to_string().cmp(&b.accelerator.to_string()));
        }
        Some("family") => {
            profiles.sort_by(|a, b| {
                a.accelerator
                    .family()
                    .to_string()
                    .cmp(&b.accelerator.family().to_string())
            });
        }
        _ => {} // default order (detection order)
    }

    println!(
        "{:<6} {:<28} {:>10} {:>10} {:>10} {:>8} {:>6} {:>8}",
        "ID", "Device", "Memory", "Free", "BW", "PCIe", "NUMA", "Status"
    );
    println!("{}", "-".repeat(90));

    for (i, p) in profiles.iter().enumerate() {
        let mem_gb = p.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_str = match p.memory_free_bytes {
            Some(b) => format!("{:.1} GB", b as f64 / (1024.0 * 1024.0 * 1024.0)),
            None => "-".into(),
        };
        let bw_str = match p.memory_bandwidth_gbps {
            Some(bw) if bw >= 1000.0 => format!("{:.1} TB/s", bw / 1000.0),
            Some(bw) => format!("{:.0} GB/s", bw),
            None => "-".into(),
        };
        let pcie_str = match p.pcie_bandwidth_gbps {
            Some(bw) => format!("{:.1}", bw),
            None => "-".into(),
        };
        let numa_str = match p.numa_node {
            Some(n) => n.to_string(),
            None => "-".into(),
        };
        let status = if p.available { "ok" } else { "unavail" };
        println!(
            "{:<6} {:<28} {:>7.1} GB {:>10} {:>10} {:>8} {:>6} {:>8}",
            i,
            truncate(&p.accelerator.to_string(), 28),
            mem_gb,
            free_str,
            bw_str,
            pcie_str,
            numa_str,
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

    // System I/O summary
    let sio = registry.system_io();
    if !sio.interconnects.is_empty() {
        println!();
        println!("Interconnects:");
        for ic in &sio.interconnects {
            let state = ic.state.as_deref().unwrap_or("unknown");
            println!(
                "  {} ({}) — {:.1} GB/s [{}]",
                ic.name, ic.kind, ic.bandwidth_gbps, state
            );
        }
    }
    if !sio.storage.is_empty() {
        println!();
        println!("Storage:");
        for dev in &sio.storage {
            println!(
                "  {} ({}) — {:.1} GB/s est.",
                dev.name, dev.kind, dev.bandwidth_gbps
            );
        }
    }

    if !registry.warnings().is_empty() {
        println!();
        println!("Warnings:");
        for w in registry.warnings() {
            println!("  {}", w);
        }
    }
}

fn parse_family(s: &str) -> Option<AcceleratorFamily> {
    match s.to_lowercase().as_str() {
        "cpu" => Some(AcceleratorFamily::Cpu),
        "gpu" => Some(AcceleratorFamily::Gpu),
        "npu" => Some(AcceleratorFamily::Npu),
        "tpu" => Some(AcceleratorFamily::Tpu),
        "asic" | "ai-asic" | "ai_asic" => Some(AcceleratorFamily::AiAsic),
        _ => None,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if max < 4 || s.len() <= max {
        return s.to_string();
    }
    // UTF-8 safe: use chars to avoid splitting multi-byte characters.
    let truncated: String = s.chars().take(max - 3).collect();
    format!("{}...", truncated)
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
        "schema_version": ai_hwaccel::SCHEMA_VERSION,
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
            "gpu": registry.by_family(AcceleratorFamily::Gpu).len(),
            "tpu": registry.by_family(AcceleratorFamily::Tpu).len(),
            "npu": registry.by_family(AcceleratorFamily::Npu).len(),
            "ai_asic": registry.by_family(AcceleratorFamily::AiAsic).len(),
        },
        "warnings": registry.warnings().iter().map(|w| w.to_string()).collect::<Vec<_>>(),
    })
}
