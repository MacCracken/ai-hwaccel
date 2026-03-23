//! Unit tests for parsing functions in detect submodules.
//!
//! These functions are benchmarked but previously lacked unit tests.

use crate::detect::bandwidth::{
    estimate_nvidia_bandwidth_from_cc, nvidia_bus_width_bits, parse_max_dpm_clock,
};
use crate::detect::interconnect::parse_ib_rate;
use crate::detect::pcie::parse_link_speed;

// ---------------------------------------------------------------------------
// nvidia_bus_width_bits — all known compute capabilities
// ---------------------------------------------------------------------------

#[test]
fn bus_width_blackwell() {
    assert_eq!(nvidia_bus_width_bits("10.0"), Some(8192));
}

#[test]
fn bus_width_hopper() {
    assert_eq!(nvidia_bus_width_bits("9.0"), Some(5120));
}

#[test]
fn bus_width_ada_lovelace() {
    assert_eq!(nvidia_bus_width_bits("8.9"), Some(384));
}

#[test]
fn bus_width_ampere_datacenter() {
    assert_eq!(nvidia_bus_width_bits("8.0"), Some(5120));
}

#[test]
fn bus_width_ampere_consumer() {
    assert_eq!(nvidia_bus_width_bits("8.6"), Some(384));
}

#[test]
fn bus_width_turing() {
    assert_eq!(nvidia_bus_width_bits("7.5"), Some(352));
}

#[test]
fn bus_width_volta() {
    assert_eq!(nvidia_bus_width_bits("7.0"), Some(4096));
}

#[test]
fn bus_width_pascal_datacenter() {
    assert_eq!(nvidia_bus_width_bits("6.0"), Some(4096));
}

#[test]
fn bus_width_pascal_consumer() {
    assert_eq!(nvidia_bus_width_bits("6.1"), Some(352));
}

#[test]
fn bus_width_unknown_cc() {
    assert_eq!(nvidia_bus_width_bits("5.0"), None);
    assert_eq!(nvidia_bus_width_bits(""), None);
    assert_eq!(nvidia_bus_width_bits("abc"), None);
}

// ---------------------------------------------------------------------------
// estimate_nvidia_bandwidth_from_cc — all known capabilities
// ---------------------------------------------------------------------------

#[test]
fn bw_estimate_all_known_ccs() {
    let known = [
        ("10.0", 8000.0),
        ("9.0", 3350.0),
        ("8.9", 1008.0),
        ("8.6", 936.0),
        ("8.0", 2039.0),
        ("7.5", 616.0),
        ("7.0", 900.0),
        ("6.1", 484.0),
        ("6.0", 732.0),
    ];
    for (cc, expected) in &known {
        let bw = estimate_nvidia_bandwidth_from_cc(cc);
        assert_eq!(
            bw,
            Some(*expected),
            "CC {} expected {} GB/s",
            cc,
            expected
        );
    }
}

#[test]
fn bw_estimate_unknown_cc() {
    assert_eq!(estimate_nvidia_bandwidth_from_cc("5.0"), None);
    assert_eq!(estimate_nvidia_bandwidth_from_cc("99.9"), None);
}

// ---------------------------------------------------------------------------
// parse_max_dpm_clock — AMD sysfs format
// ---------------------------------------------------------------------------

#[test]
fn dpm_clock_standard_format() {
    assert_eq!(
        parse_max_dpm_clock("0: 96Mhz\n1: 1000Mhz *\n"),
        Some(1000.0)
    );
}

#[test]
fn dpm_clock_hbm_three_levels() {
    assert_eq!(
        parse_max_dpm_clock("0: 500Mhz\n1: 900Mhz\n2: 1600Mhz *\n"),
        Some(1600.0)
    );
}

#[test]
fn dpm_clock_case_insensitive_mhz() {
    assert_eq!(parse_max_dpm_clock("0: 500MHz\n"), Some(500.0));
}

#[test]
fn dpm_clock_empty_returns_none() {
    assert_eq!(parse_max_dpm_clock(""), None);
}

#[test]
fn dpm_clock_garbage_returns_none() {
    assert_eq!(parse_max_dpm_clock("not a clock\nstill not\n"), None);
}

#[test]
fn dpm_clock_single_entry() {
    assert_eq!(parse_max_dpm_clock("0: 800Mhz *\n"), Some(800.0));
}

#[test]
fn dpm_clock_picks_highest_not_starred() {
    // Even without *, should return the max.
    assert_eq!(
        parse_max_dpm_clock("0: 100Mhz\n1: 900Mhz\n2: 500Mhz\n"),
        Some(900.0)
    );
}

// ---------------------------------------------------------------------------
// parse_link_speed — PCIe sysfs format
// ---------------------------------------------------------------------------

#[test]
fn link_speed_gen5() {
    assert_eq!(parse_link_speed("32 GT/s"), Some(32.0));
}

#[test]
fn link_speed_gen4() {
    assert_eq!(parse_link_speed("16 GT/s"), Some(16.0));
}

#[test]
fn link_speed_gen3() {
    assert_eq!(parse_link_speed("8.0 GT/s PCIe"), Some(8.0));
}

#[test]
fn link_speed_gen2() {
    assert_eq!(parse_link_speed("5 GT/s"), Some(5.0));
}

#[test]
fn link_speed_gen1() {
    assert_eq!(parse_link_speed("2.5 GT/s"), Some(2.5));
}

#[test]
fn link_speed_empty() {
    assert_eq!(parse_link_speed(""), None);
}

#[test]
fn link_speed_no_number() {
    assert_eq!(parse_link_speed("Unknown"), None);
}

// ---------------------------------------------------------------------------
// parse_ib_rate — InfiniBand sysfs format
// ---------------------------------------------------------------------------

#[test]
fn ib_rate_hdr_200g() {
    let bw = parse_ib_rate("200 Gb/sec (4X HDR)");
    assert!((bw - 25.0).abs() < 0.01);
}

#[test]
fn ib_rate_ndr_400g() {
    let bw = parse_ib_rate("400 Gb/sec (4X NDR)");
    assert!((bw - 50.0).abs() < 0.01);
}

#[test]
fn ib_rate_edr_100g() {
    let bw = parse_ib_rate("100 Gb/sec (4X EDR)");
    assert!((bw - 12.5).abs() < 0.01);
}

#[test]
fn ib_rate_fdr_56g() {
    let bw = parse_ib_rate("56 Gb/sec (4X FDR)");
    assert!((bw - 7.0).abs() < 0.01);
}

#[test]
fn ib_rate_qdr_40g() {
    let bw = parse_ib_rate("40 Gb/sec (4X QDR)");
    assert!((bw - 5.0).abs() < 0.01);
}

#[test]
fn ib_rate_empty() {
    assert_eq!(parse_ib_rate(""), 0.0);
}

#[test]
fn ib_rate_garbage() {
    assert_eq!(parse_ib_rate("not a rate"), 0.0);
}

// ---------------------------------------------------------------------------
// CUDA parser: normal path (11-field output)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[test]
fn cuda_parser_normal_h100() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let line = "0, 81920, 1024, 80896, 9.0, 550.54.15, NVIDIA H100 80GB HBM3, 42, 280.50, 15, 2619\n";
    crate::detect::cuda::parse_cuda_output(line, &mut profiles, &mut warnings);
    assert!(warnings.is_empty(), "unexpected warnings: {:?}", warnings);
    assert_eq!(profiles.len(), 1);
    let p = &profiles[0];
    assert!(matches!(
        p.accelerator,
        crate::hardware::AcceleratorType::CudaGpu { device_id: 0 }
    ));
    assert_eq!(p.compute_capability.as_deref(), Some("9.0"));
    assert_eq!(p.driver_version.as_deref(), Some("550.54.15"));
    assert_eq!(p.temperature_c, Some(42));
    assert!((p.power_watts.unwrap() - 280.50).abs() < 0.01);
    assert_eq!(p.gpu_utilization_percent, Some(15));
    assert!(p.memory_bandwidth_gbps.is_some());
    // H100 bandwidth: 2619 MHz * 5120 bit * 2 / 8 / 1000 ≈ 3352 GB/s
    assert!(p.memory_bandwidth_gbps.unwrap() > 3000.0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_parser_multi_gpu() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let lines = "\
0, 81920, 1024, 80896, 9.0, 550.54, NVIDIA H100, 42, 280, 15, 2619
1, 81920, 2048, 79872, 9.0, 550.54, NVIDIA H100, 45, 290, 20, 2619
";
    crate::detect::cuda::parse_cuda_output(lines, &mut profiles, &mut warnings);
    assert_eq!(profiles.len(), 2);
    assert!(matches!(
        profiles[0].accelerator,
        crate::hardware::AcceleratorType::CudaGpu { device_id: 0 }
    ));
    assert!(matches!(
        profiles[1].accelerator,
        crate::hardware::AcceleratorType::CudaGpu { device_id: 1 }
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_parser_legacy_6_fields() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::cuda::parse_cuda_output(
        "0, 8192, 1000, 7192, 8.6, 535.00\n",
        &mut profiles,
        &mut warnings,
    );
    assert_eq!(profiles.len(), 1);
    assert!(profiles[0].temperature_c.is_none());
    assert!(profiles[0].power_watts.is_none());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_parser_invalid_device_id() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::cuda::parse_cuda_output(
        "abc, 8192, 1000, 7192, 8.6, 535.00\n",
        &mut profiles,
        &mut warnings,
    );
    assert!(profiles.is_empty());
    assert_eq!(warnings.len(), 1);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_parser_invalid_memory() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::cuda::parse_cuda_output(
        "0, notanumber, 1000, 7192, 8.6, 535.00\n",
        &mut profiles,
        &mut warnings,
    );
    assert!(profiles.is_empty());
    assert_eq!(warnings.len(), 1);
}
