//! CLI binary for ai-hwaccel — outputs the accelerator registry as JSON.
//!
//! Usage:
//!   ai-hwaccel                             # Full registry JSON (compact)
//!   ai-hwaccel --pretty                    # Full registry JSON (formatted)
//!   ai-hwaccel --table                     # Human-readable table
//!   ai-hwaccel --table --sort mem          # Table sorted by memory (desc)
//!   ai-hwaccel --table --family gpu        # Table filtered to GPUs only
//!   ai-hwaccel --table --columns name,mem  # Select specific columns
//!   ai-hwaccel --table --tsv               # Tab-separated (machine-readable)
//!   ai-hwaccel --summary                   # Compact summary JSON
//!   ai-hwaccel --watch 5                   # Re-detect every 5s with deltas
//!   ai-hwaccel --watch 5 --alert mem>90    # Alert when VRAM usage > 90%
//!   ai-hwaccel --cost 70B --quant bf16     # Cheapest cloud instance for 70B@BF16
//!   ai-hwaccel --profile                   # Show per-backend detection timing
//!   ai-hwaccel --debug                     # Verbose detection diagnostics
//!   ai-hwaccel --version                   # Print version
//!
//! Columns:
//!   id, name, mem, free, bw, pcie, numa, family, status (default: all)
//!
//! Logging:
//!   Set `RUST_LOG` to control verbosity (e.g. `RUST_LOG=debug ai-hwaccel`).
//!   Use `--json-log` to emit structured JSON logs to stderr.

use std::collections::HashMap;
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

    // --cost <model_size> [--quant <level>] [--provider <aws|gcp|azure>]
    if let Some(model_str) = get_val("--cost") {
        let model_params = parse_model_size(&model_str);
        let quant = get_val("--quant")
            .and_then(|q| parse_quant_level(&q))
            .unwrap_or(ai_hwaccel::QuantizationLevel::BFloat16);
        let provider = get_val("--provider").and_then(|p| match p.to_lowercase().as_str() {
            "aws" => Some(ai_hwaccel::cost::CloudProvider::Aws),
            "gcp" | "google" => Some(ai_hwaccel::cost::CloudProvider::Gcp),
            "azure" | "microsoft" => Some(ai_hwaccel::cost::CloudProvider::Azure),
            _ => None,
        });

        let recs = ai_hwaccel::cost::recommend_instance(model_params, &quant, provider);
        if recs.is_empty() {
            println!(
                "No cloud instance has enough GPU memory for {} params at {}.",
                format_params(model_params),
                quant
            );
        } else {
            let needed_gb = recs[0].memory_required_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            println!(
                "Model: {} params at {} — {:.1} GB required\n",
                format_params(model_params),
                quant,
                needed_gb
            );
            println!(
                "{:<32} {:>8} {:>6} {:>10} {:>8} {:>10}",
                "Instance", "Provider", "GPUs", "GPU Mem", "$/hr", "Headroom"
            );
            println!("{}", "-".repeat(80));
            for rec in recs.iter().take(10) {
                println!(
                    "{:<32} {:>8} {:>6} {:>7} GB {:>8.2} {:>9.0}%",
                    rec.instance.name,
                    rec.instance.provider,
                    rec.instance.gpu_count,
                    rec.instance.total_gpu_memory_gb,
                    rec.instance.price_per_hour,
                    rec.memory_headroom_pct
                );
            }
        }
        return;
    }

    let tsv = has("--tsv");
    let columns = get_val("--columns").map(|s| parse_columns(&s));
    let sort = get_val("--sort");
    let family_filter = get_val("--family");
    let alert = get_val("--alert").and_then(|s| parse_alert(&s));

    // --watch <seconds>: re-detect on interval
    if let Some(interval_str) = get_val("--watch") {
        let secs: u64 = interval_str.parse().unwrap_or(10);
        run_watch(
            Duration::from_secs(secs),
            sort,
            family_filter,
            columns,
            tsv,
            alert,
        );
        return;
    }

    let profile_mode = has("--profile");

    info!("starting accelerator detection");
    let (registry, timed_result) = if profile_mode {
        let result = AcceleratorRegistry::detect_with_timing();
        let timings = result.timings.clone();
        let total = result.total;
        (result.registry, Some((timings, total)))
    } else {
        (AcceleratorRegistry::detect(), None)
    };
    log_detection(&registry);

    if let Some((ref timings, total)) = timed_result {
        eprintln!("\nBackend timing:");
        let mut entries: Vec<_> = timings.iter().collect();
        entries.sort_by(|a, b| b.1.cmp(a.1));
        for (name, dur) in &entries {
            eprintln!("  {:<16} {:>8.1}ms", name, dur.as_secs_f64() * 1000.0);
        }
        eprintln!(
            "  {:<16} {:>8.1}ms",
            "TOTAL (wall)",
            total.as_secs_f64() * 1000.0
        );
        eprintln!();
    }

    let pretty = has("--pretty") || has("-p");

    if has("--table") || has("-t") || tsv {
        let cols = columns.as_deref().unwrap_or(Column::ALL);
        print_table(
            &registry,
            sort.as_deref(),
            family_filter.as_deref(),
            cols,
            tsv,
            None,
        );
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

// ---------------------------------------------------------------------------
// Column selection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Column {
    Id,
    Name,
    Memory,
    Free,
    Bandwidth,
    Pcie,
    Numa,
    Family,
    Status,
}

impl Column {
    const ALL: &[Column] = &[
        Column::Id,
        Column::Name,
        Column::Memory,
        Column::Free,
        Column::Bandwidth,
        Column::Pcie,
        Column::Numa,
        Column::Status,
    ];
}

fn parse_columns(s: &str) -> Vec<Column> {
    s.split(',')
        .filter_map(|c| match c.trim().to_lowercase().as_str() {
            "id" => Some(Column::Id),
            "name" | "device" => Some(Column::Name),
            "mem" | "memory" => Some(Column::Memory),
            "free" => Some(Column::Free),
            "bw" | "bandwidth" => Some(Column::Bandwidth),
            "pcie" => Some(Column::Pcie),
            "numa" => Some(Column::Numa),
            "family" => Some(Column::Family),
            "status" => Some(Column::Status),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Alert thresholds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct AlertThreshold {
    metric: AlertMetric,
    value: f64,
}

#[derive(Debug, Clone, Copy)]
enum AlertMetric {
    /// VRAM usage as percentage of total (0–100).
    MemoryPercent,
}

fn parse_alert(s: &str) -> Option<AlertThreshold> {
    // Format: "mem>90" or "mem>90%"
    let s = s.trim().trim_end_matches('%');
    if let Some(rest) = s.strip_prefix("mem>") {
        let value: f64 = rest.parse().ok()?;
        return Some(AlertThreshold {
            metric: AlertMetric::MemoryPercent,
            value,
        });
    }
    None
}

fn check_alerts(profile: &AcceleratorProfile, threshold: &AlertThreshold) -> Option<String> {
    match threshold.metric {
        AlertMetric::MemoryPercent => {
            let used = profile.memory_used_bytes?;
            let total = profile.memory_bytes;
            if total == 0 {
                return None;
            }
            let pct = used as f64 / total as f64 * 100.0;
            if pct > threshold.value {
                Some(format!(
                    "ALERT: {} memory {:.0}% > {:.0}% threshold",
                    profile.accelerator, pct, threshold.value
                ))
            } else {
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Watch mode with deltas
// ---------------------------------------------------------------------------

fn run_watch(
    interval: Duration,
    sort: Option<String>,
    family: Option<String>,
    columns: Option<Vec<Column>>,
    tsv: bool,
    alert: Option<AlertThreshold>,
) {
    let cols = columns.as_deref().unwrap_or(Column::ALL);
    let mut prev_used: HashMap<String, u64> = HashMap::new();

    loop {
        // Clear screen
        print!("\x1b[2J\x1b[H");
        let registry = AcceleratorRegistry::detect();

        // Build delta map: device key → change in memory_used_bytes.
        // Uses Debug format as key for unique device identity across runs.
        let mut deltas: HashMap<String, i64> = HashMap::new();
        for p in registry.all_profiles() {
            let key = format!("{:?}", p.accelerator);
            if let Some(used) = p.memory_used_bytes {
                if let Some(&prev) = prev_used.get(&key) {
                    let delta = used as i64 - prev as i64;
                    if delta != 0 {
                        deltas.insert(key.clone(), delta);
                    }
                }
                prev_used.insert(key, used);
            }
        }

        print_table(
            &registry,
            sort.as_deref(),
            family.as_deref(),
            cols,
            tsv,
            Some(&deltas),
        );

        // Alert checks
        if let Some(ref threshold) = alert {
            for p in registry.all_profiles() {
                if let Some(msg) = check_alerts(p, threshold) {
                    eprintln!("\x1b[1;31m  {}\x1b[0m", msg);
                }
            }
        }

        println!(
            "\nRefreshing every {}s... (Ctrl+C to stop)",
            interval.as_secs()
        );
        std::thread::sleep(interval);
    }
}

// ---------------------------------------------------------------------------
// Table output
// ---------------------------------------------------------------------------

fn print_table(
    registry: &AcceleratorRegistry,
    sort_by: Option<&str>,
    family_filter: Option<&str>,
    columns: &[Column],
    tsv: bool,
    deltas: Option<&HashMap<String, i64>>,
) {
    let mut profiles: Vec<&AcceleratorProfile> = registry.all_profiles().iter().collect();

    // Filter by family
    if let Some(f) = family_filter
        && let Some(fam) = parse_family(f)
    {
        profiles.retain(|p| p.accelerator.family() == fam);
    }

    // Sort
    match sort_by {
        Some("mem" | "memory") => {
            profiles.sort_by(|a, b| b.memory_bytes.cmp(&a.memory_bytes));
        }
        Some("name" | "device") => {
            profiles.sort_by_cached_key(|p| p.accelerator.to_string());
        }
        Some("family") => {
            profiles.sort_by_cached_key(|p| p.accelerator.family().to_string());
        }
        _ => {}
    }

    let sep = if tsv { "\t" } else { " " };

    // Header
    let header: Vec<&str> = columns
        .iter()
        .map(|c| match c {
            Column::Id => "ID",
            Column::Name => "Device",
            Column::Memory => "Memory",
            Column::Free => "Free",
            Column::Bandwidth => "BW",
            Column::Pcie => "PCIe",
            Column::Numa => "NUMA",
            Column::Family => "Family",
            Column::Status => "Status",
        })
        .collect();

    if tsv {
        println!("{}", header.join(sep));
    } else {
        let hdr = format_row(columns, &header);
        println!("{}", hdr);
        println!("{}", "-".repeat(hdr.len()));
    }

    // Rows
    for (i, p) in profiles.iter().enumerate() {
        let mem_gb = format!(
            "{:.1} GB",
            p.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );

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
        let device_name = p.accelerator.to_string();

        // Check for memory delta annotation (keyed by Debug format for unique identity).
        let debug_key = format!("{:?}", p.accelerator);
        let delta_annotation = deltas
            .and_then(|d| d.get(&debug_key))
            .map(|&d| {
                let gb = d.abs() as f64 / (1024.0 * 1024.0 * 1024.0);
                if d > 0 {
                    format!(" (+{:.1})", gb)
                } else {
                    format!(" (-{:.1})", gb)
                }
            })
            .unwrap_or_default();

        let free_with_delta = format!("{}{}", free_str, delta_annotation);

        let values: Vec<String> = columns
            .iter()
            .map(|c| match c {
                Column::Id => i.to_string(),
                Column::Name => {
                    if tsv {
                        device_name.clone()
                    } else {
                        truncate(&device_name, 28)
                    }
                }
                Column::Memory => mem_gb.clone(),
                Column::Free => free_with_delta.clone(),
                Column::Bandwidth => bw_str.clone(),
                Column::Pcie => pcie_str.clone(),
                Column::Numa => numa_str.clone(),
                Column::Family => p.accelerator.family().to_string(),
                Column::Status => status.into(),
            })
            .collect();

        if tsv {
            println!("{}", values.join(sep));
        } else {
            let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
            println!("{}", format_row(columns, &refs));
        }
    }

    if tsv {
        return; // TSV mode: data only, no footer
    }

    // Footer
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

    // Environment info
    if let Some(ref env) = sio.environment {
        println!();
        println!("Environment:");
        if env.is_docker {
            println!("  Container: Docker");
        }
        if env.is_kubernetes {
            let ns = env.kubernetes_namespace.as_deref().unwrap_or("unknown");
            println!("  Kubernetes: namespace={}", ns);
        }
        if let Some(ref cloud) = env.cloud_instance {
            let instance_type = cloud.instance_type.as_deref().unwrap_or("unknown");
            let region = cloud.region.as_deref().unwrap_or("unknown");
            println!(
                "  Cloud: {} (type={}, region={})",
                cloud.provider, instance_type, region
            );
        }
        if !env.is_docker && !env.is_kubernetes && env.cloud_instance.is_none() {
            println!("  Bare metal / local VM");
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

/// Format a row with fixed-width columns for human-readable output.
fn format_row(columns: &[Column], values: &[&str]) -> String {
    columns
        .iter()
        .zip(values.iter())
        .map(|(col, val)| match col {
            Column::Id => format!("{:<6}", val),
            Column::Name => format!("{:<28}", val),
            Column::Memory => format!("{:>10}", val),
            Column::Free => format!("{:>10}", val),
            Column::Bandwidth => format!("{:>10}", val),
            Column::Pcie => format!("{:>8}", val),
            Column::Numa => format!("{:>6}", val),
            Column::Family => format!("{:>8}", val),
            Column::Status => format!("{:>8}", val),
        })
        .collect::<Vec<_>>()
        .join(" ")
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

/// Parse a model size string like "7B", "70b", "405B", "7000M", or a raw number.
fn parse_model_size(s: &str) -> u64 {
    let s = s.trim();
    if let Some(num_str) = s.strip_suffix(['B', 'b'])
        && let Ok(n) = num_str.parse::<f64>()
    {
        return (n * 1_000_000_000.0) as u64;
    }
    if let Some(num_str) = s.strip_suffix(['M', 'm'])
        && let Ok(n) = num_str.parse::<f64>()
    {
        return (n * 1_000_000.0) as u64;
    }
    s.parse::<u64>().unwrap_or(7_000_000_000)
}

/// Format a parameter count for display (e.g. 7000000000 → "7.0B").
fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.0}M", params as f64 / 1_000_000.0)
    } else {
        format!("{}", params)
    }
}

/// Parse a quantisation level string.
fn parse_quant_level(s: &str) -> Option<ai_hwaccel::QuantizationLevel> {
    match s.to_lowercase().as_str() {
        "fp32" | "f32" | "none" => Some(ai_hwaccel::QuantizationLevel::None),
        "fp16" | "f16" | "float16" => Some(ai_hwaccel::QuantizationLevel::Float16),
        "bf16" | "bfloat16" => Some(ai_hwaccel::QuantizationLevel::BFloat16),
        "int8" | "i8" | "q8" => Some(ai_hwaccel::QuantizationLevel::Int8),
        "int4" | "i4" | "q4" => Some(ai_hwaccel::QuantizationLevel::Int4),
        _ => None,
    }
}
