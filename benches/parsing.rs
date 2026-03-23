//! Benchmarks for hardware detection parsing routines.

use criterion::{Criterion, criterion_group, criterion_main};

use ai_hwaccel::detect::bandwidth::{
    estimate_nvidia_bandwidth_from_cc, nvidia_bus_width_bits, parse_max_dpm_clock,
    parse_nvidia_bandwidth_output,
};
use ai_hwaccel::detect::interconnect::{parse_ib_rate, parse_nvlink_output};
use ai_hwaccel::detect::pcie::parse_link_speed;

fn bench_parse_nvidia_bandwidth(c: &mut Criterion) {
    let output = "1593, 8.0\n2619, 9.0\n10501, 8.9\n1593, 8.0\n2619, 9.0\n10501, 8.9\n1593, 8.0\n2619, 9.0\n";
    c.bench_function("parse_nvidia_bandwidth_8gpu", |b| {
        b.iter(|| parse_nvidia_bandwidth_output(output));
    });
}

fn bench_nvidia_bus_width(c: &mut Criterion) {
    let ccs = ["9.0", "8.9", "8.6", "8.0", "7.5", "7.0", "6.1", "6.0", "10.0"];
    c.bench_function("nvidia_bus_width_all_ccs", |b| {
        b.iter(|| {
            for cc in &ccs {
                let _ = nvidia_bus_width_bits(cc);
            }
        });
    });
}

fn bench_estimate_nvidia_bw_from_cc(c: &mut Criterion) {
    let ccs = ["10.0", "9.0", "8.9", "8.6", "8.0", "7.5", "7.0", "6.1", "6.0"];
    c.bench_function("estimate_bw_from_cc_all", |b| {
        b.iter(|| {
            for cc in &ccs {
                let _ = estimate_nvidia_bandwidth_from_cc(cc);
            }
        });
    });
}

fn bench_parse_max_dpm_clock(c: &mut Criterion) {
    let input = "0: 96Mhz\n1: 500Mhz\n2: 900Mhz\n3: 1600Mhz *\n";
    c.bench_function("parse_max_dpm_clock", |b| {
        b.iter(|| parse_max_dpm_clock(input));
    });
}

fn bench_parse_link_speed(c: &mut Criterion) {
    c.bench_function("parse_link_speed", |b| {
        b.iter(|| {
            let _ = parse_link_speed("16 GT/s");
            let _ = parse_link_speed("8.0 GT/s PCIe");
            let _ = parse_link_speed("2.5 GT/s");
        });
    });
}

fn bench_parse_ib_rate(c: &mut Criterion) {
    c.bench_function("parse_ib_rate", |b| {
        b.iter(|| {
            let _ = parse_ib_rate("200 Gb/sec (4X HDR)");
            let _ = parse_ib_rate("400 Gb/sec (4X NDR)");
            let _ = parse_ib_rate("100 Gb/sec (4X EDR)");
        });
    });
}

fn bench_parse_nvlink_output(c: &mut Criterion) {
    let output = "\
GPU 0: NVIDIA H100 (UUID: GPU-abc123)
\tLink 0: 25 GB/s
\tLink 1: 25 GB/s
\tLink 2: 25 GB/s
\tLink 3: 25 GB/s
GPU 1: NVIDIA H100 (UUID: GPU-def456)
\tLink 0: 25 GB/s
\tLink 1: 25 GB/s
\tLink 2: 25 GB/s
\tLink 3: 25 GB/s
";
    c.bench_function("parse_nvlink_output_2gpu", |b| {
        b.iter(|| {
            let mut interconnects = Vec::new();
            parse_nvlink_output(output, &mut interconnects);
        });
    });
}

criterion_group!(
    benches,
    bench_parse_nvidia_bandwidth,
    bench_nvidia_bus_width,
    bench_estimate_nvidia_bw_from_cc,
    bench_parse_max_dpm_clock,
    bench_parse_link_speed,
    bench_parse_ib_rate,
    bench_parse_nvlink_output,
);
criterion_main!(benches);
