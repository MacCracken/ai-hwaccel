//! Unit tests for parsing functions in detect submodules.
//!
//! These functions are benchmarked but previously lacked unit tests.

use crate::detect::bandwidth::{
    estimate_nvidia_bandwidth_from_cc, nvidia_bus_width_bits, parse_max_dpm_clock,
};
use crate::detect::interconnect::parse_ib_rate;
use crate::detect::pcie::parse_link_speed;
use crate::hardware::AcceleratorType;

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
        assert_eq!(bw, Some(*expected), "CC {} expected {} GB/s", cc, expected);
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
    let line =
        "0, 81920, 1024, 80896, 9.0, 550.54.15, NVIDIA H100 80GB HBM3, 42, 280.50, 15, 2619\n";
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

#[cfg(feature = "cuda")]
#[test]
fn cuda_parser_zero_clock_no_bandwidth() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    // Clock speed 0 should not produce NaN or panic in bandwidth calc.
    crate::detect::cuda::parse_cuda_output(
        "0, 81920, 0, 81920, 9.0, 550.54, NVIDIA H100, 42, 280, 10, 0\n",
        &mut profiles,
        &mut warnings,
    );
    assert_eq!(profiles.len(), 1);
    // Bandwidth with 0 MHz clock should be 0.0, then fallback to estimate_from_cc.
    assert!(profiles[0].memory_bandwidth_gbps.is_some());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_parser_empty_optional_fields() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    // temp, power, util, clock fields are empty/missing.
    crate::detect::cuda::parse_cuda_output(
        "0, 24576, 1000, 23576, 8.9, 545.00, RTX 4090, , , , \n",
        &mut profiles,
        &mut warnings,
    );
    assert_eq!(profiles.len(), 1);
    assert!(profiles[0].temperature_c.is_none());
    assert!(profiles[0].power_watts.is_none());
    assert!(profiles[0].gpu_utilization_percent.is_none());
}

// ---------------------------------------------------------------------------
// Vulkan parser — parse_vulkan_output with fixture data
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_parser_single_gpu_summary() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    // The vulkan summary parser expects "size = <MB>" as standalone lines.
    let summary = "\
GPU0:
\tdeviceName = AMD Radeon RX 7900 XTX
\tapiVersion = 1.3.274
\tdriverVersion = 24.1.2
\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tsize = 24560 MiB
";
    crate::detect::vulkan::parse_vulkan_output(summary, None, &mut profiles, &mut warnings);
    assert_eq!(profiles.len(), 1);
    assert!(matches!(
        &profiles[0].accelerator,
        crate::hardware::AcceleratorType::VulkanGpu { .. }
    ));
    assert!(
        profiles[0]
            .device_name
            .as_deref()
            .is_some_and(|n| n.contains("AMD Radeon")),
        "device_name should contain 'AMD Radeon', got {:?}",
        profiles[0].device_name
    );
    assert!(profiles[0].memory_bytes > 20 * 1024 * 1024 * 1024);
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_parser_multi_gpu() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let summary = "\
GPU0:
\tdeviceName = NVIDIA GeForce RTX 4090
\tapiVersion = 1.3.280
\tsize = 24564 MiB
GPU1:
\tdeviceName = NVIDIA GeForce RTX 4090
\tapiVersion = 1.3.280
\tsize = 24564 MiB
";
    crate::detect::vulkan::parse_vulkan_output(summary, None, &mut profiles, &mut warnings);
    assert_eq!(profiles.len(), 2);
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_parser_empty_output_creates_fallback() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::vulkan::parse_vulkan_output(
        "Vulkan Instance Version: 1.3.0\n",
        None,
        &mut profiles,
        &mut warnings,
    );
    assert_eq!(profiles.len(), 1);
}

// ---------------------------------------------------------------------------
// Apple: parse_displays_json (system_profiler SPDisplaysDataType -json)
// ---------------------------------------------------------------------------

#[cfg(feature = "apple")]
#[test]
fn apple_displays_json_m4_max() {
    let json = r#"{
        "SPDisplaysDataType": [{
            "_name": "Apple M4 Max",
            "sppci_model": "Apple M4 Max",
            "sppci_vendor": "Apple",
            "spdisplays_metal_family": "Metal 3, Metal Family - Apple 9",
            "sppci_cores": "40"
        }]
    }"#;
    let gpus = crate::detect::apple::parse_displays_json(json);
    assert_eq!(gpus.len(), 1);
    assert_eq!(gpus[0].name, "Apple M4 Max");
    assert_eq!(gpus[0].vendor, "Apple");
    assert_eq!(
        gpus[0].metal_family.as_deref(),
        Some("Metal 3, Metal Family - Apple 9")
    );
    assert_eq!(gpus[0].cores, Some(40));
    assert!(gpus[0].vram_bytes.is_none()); // integrated — no discrete VRAM
}

#[cfg(feature = "apple")]
#[test]
fn apple_displays_json_discrete_gpu() {
    // Intel Mac with discrete AMD GPU
    let json = r#"{
        "SPDisplaysDataType": [{
            "_name": "AMD Radeon Pro 5500M",
            "sppci_model": "AMD Radeon Pro 5500M",
            "sppci_vendor": "AMD (0x1002)",
            "spdisplays_metal_family": "Metal 3, Metal Family - Common 3",
            "sppci_vram": "8192 MB"
        }]
    }"#;
    let gpus = crate::detect::apple::parse_displays_json(json);
    assert_eq!(gpus.len(), 1);
    assert_eq!(gpus[0].name, "AMD Radeon Pro 5500M");
    assert_eq!(gpus[0].vram_bytes, Some(8192 * 1024 * 1024));
}

#[cfg(feature = "apple")]
#[test]
fn apple_displays_json_multi_gpu() {
    // MacBook Pro with integrated + discrete
    let json = r#"{
        "SPDisplaysDataType": [
            {
                "sppci_model": "Intel UHD Graphics 630",
                "sppci_vendor": "Intel",
                "spdisplays_metal_family": "Metal 3, Metal Family - Common 2"
            },
            {
                "sppci_model": "AMD Radeon Pro 5600M",
                "sppci_vendor": "AMD (0x1002)",
                "spdisplays_metal_family": "Metal 3, Metal Family - Common 3",
                "sppci_vram": "8192 MB"
            }
        ]
    }"#;
    let gpus = crate::detect::apple::parse_displays_json(json);
    assert_eq!(gpus.len(), 2);
    assert!(gpus[0].vram_bytes.is_none()); // integrated
    assert_eq!(gpus[1].vram_bytes, Some(8192 * 1024 * 1024)); // discrete
}

#[cfg(feature = "apple")]
#[test]
fn apple_displays_json_empty() {
    assert!(crate::detect::apple::parse_displays_json("{}").is_empty());
    assert!(crate::detect::apple::parse_displays_json("invalid").is_empty());
    assert!(crate::detect::apple::parse_displays_json(r#"{"SPDisplaysDataType": []}"#).is_empty());
}

// ---------------------------------------------------------------------------
// Apple: parse_sysctl_output (macOS CPU topology)
// ---------------------------------------------------------------------------

#[cfg(feature = "apple")]
#[test]
fn apple_sysctl_m4_max() {
    let output = "\
hw.memsize: 137438953472
hw.ncpu: 16
hw.cpufrequency: 4400000000
hw.perflevel0.logicalcpu: 12
hw.perflevel1.logicalcpu: 4
";
    let info = crate::detect::apple::parse_sysctl_output(output);
    assert_eq!(info.memory_bytes, Some(128 * 1024 * 1024 * 1024)); // 128 GB
    assert_eq!(info.cpu_count, Some(16));
    assert_eq!(info.cpu_freq_hz, Some(4_400_000_000));
    assert_eq!(info.perf_cores, Some(12));
    assert_eq!(info.eff_cores, Some(4));
}

#[cfg(feature = "apple")]
#[test]
fn apple_sysctl_intel_mac() {
    // Intel Mac: no perflevel keys
    let output = "\
hw.memsize: 17179869184
hw.ncpu: 8
hw.cpufrequency: 2300000000
";
    let info = crate::detect::apple::parse_sysctl_output(output);
    assert_eq!(info.memory_bytes, Some(16 * 1024 * 1024 * 1024));
    assert_eq!(info.cpu_count, Some(8));
    assert!(info.perf_cores.is_none());
    assert!(info.eff_cores.is_none());
}

#[cfg(feature = "apple")]
#[test]
fn apple_sysctl_empty() {
    let info = crate::detect::apple::parse_sysctl_output("");
    assert!(info.memory_bytes.is_none());
    assert!(info.cpu_count.is_none());
}

// ---------------------------------------------------------------------------
// Apple: parse_system_profiler_output (existing text parser)
// ---------------------------------------------------------------------------

#[cfg(feature = "apple")]
#[test]
fn apple_parser_m4_max() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let output = "\
Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Chip: Apple M4 Max
      Total Number of Cores: 16 (12 performance and 4 efficiency)
      Memory: 48 GB
";
    let is_mac =
        crate::detect::apple::parse_system_profiler_output(output, &mut profiles, &mut warnings);
    assert!(is_mac);
    assert_eq!(profiles.len(), 2); // Metal GPU + ANE
    assert!(matches!(
        profiles[0].accelerator,
        crate::hardware::AcceleratorType::MetalGpu
    ));
    assert!(matches!(
        profiles[1].accelerator,
        crate::hardware::AcceleratorType::AppleNpu
    ));
    assert_eq!(profiles[0].memory_bytes, 48 * 1024 * 1024 * 1024);
    assert_eq!(
        profiles[0].compute_capability.as_deref(),
        Some("Apple M4 Max")
    );
}

#[cfg(feature = "apple")]
#[test]
fn apple_parser_intel_mac_no_chip() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let output = "\
Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Processor Name: Quad-Core Intel Core i7
      Memory: 16 GB
";
    let is_mac =
        crate::detect::apple::parse_system_profiler_output(output, &mut profiles, &mut warnings);
    assert!(is_mac);
    assert!(profiles.is_empty()); // No Apple Silicon → no profiles
}

#[cfg(feature = "apple")]
#[test]
fn apple_parser_m1_8gb() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let output = "\
Hardware:

    Hardware Overview:

      Model Name: MacBook Air
      Chip: Apple M1
      Memory: 8 GB
";
    let is_mac =
        crate::detect::apple::parse_system_profiler_output(output, &mut profiles, &mut warnings);
    assert!(is_mac);
    assert_eq!(profiles.len(), 2);
    assert_eq!(profiles[0].memory_bytes, 8 * 1024 * 1024 * 1024);
}

// ---------------------------------------------------------------------------
// Gaudi parser — extended fixtures
// ---------------------------------------------------------------------------

#[cfg(feature = "gaudi")]
#[test]
fn gaudi_parser_multi_device() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let output = "\
0, hl-325-gaudi3, 131072, 100000
1, hl-325-gaudi3, 131072, 100000
2, hl-325-gaudi3, 131072, 100000
3, hl-325-gaudi3, 131072, 100000
";
    crate::detect::gaudi::parse_gaudi_output(output, &mut profiles, &mut warnings);
    assert_eq!(profiles.len(), 4);
    for (i, p) in profiles.iter().enumerate() {
        assert!(matches!(
            &p.accelerator,
            crate::hardware::AcceleratorType::Gaudi {
                device_id, generation: crate::hardware::GaudiGeneration::Gaudi3,
            } if *device_id == i as u32
        ));
    }
}

#[cfg(feature = "gaudi")]
#[test]
fn gaudi_parser_gaudi2() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::gaudi::parse_gaudi_output(
        "0, hl-225-gaudi2, 98304, 90000\n",
        &mut profiles,
        &mut warnings,
    );
    assert_eq!(profiles.len(), 1);
    assert!(matches!(
        profiles[0].accelerator,
        crate::hardware::AcceleratorType::Gaudi {
            generation: crate::hardware::GaudiGeneration::Gaudi2,
            ..
        }
    ));
}

#[cfg(feature = "gaudi")]
#[test]
fn gaudi_parser_too_few_fields() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::gaudi::parse_gaudi_output("0, hl-325\n", &mut profiles, &mut warnings);
    assert!(profiles.is_empty());
    assert_eq!(warnings.len(), 1);
}

#[cfg(feature = "gaudi")]
#[test]
fn gaudi_parser_invalid_device_id() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::gaudi::parse_gaudi_output(
        "xyz, hl-325-gaudi3, 131072, 100000\n",
        &mut profiles,
        &mut warnings,
    );
    assert!(profiles.is_empty());
    assert_eq!(warnings.len(), 1);
}

// ---------------------------------------------------------------------------
// AWS Neuron parser
// ---------------------------------------------------------------------------

#[cfg(feature = "aws-neuron")]
#[test]
fn neuron_parser_inferentia_single() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let json = r#"[{"model":"inf2.xlarge","nc_count":1,"memory_per_nc_mb":32768}]"#;
    let ok = crate::detect::neuron::parse_neuron_output(json, &mut profiles, &mut warnings);
    assert!(ok);
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].memory_bytes, 32768 * 1024 * 1024);
}

#[cfg(feature = "aws-neuron")]
#[test]
fn neuron_parser_trainium_multi() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let json = r#"[
        {"model":"trn1.32xlarge","nc_count":32,"memory_per_nc_mb":16384},
        {"model":"trn1.2xlarge","nc_count":2,"memory_per_nc_mb":16384}
    ]"#;
    let ok = crate::detect::neuron::parse_neuron_output(json, &mut profiles, &mut warnings);
    assert!(ok);
    assert_eq!(profiles.len(), 2);
    assert!(matches!(
        &profiles[0].accelerator,
        AcceleratorType::AwsNeuron {
            chip_type: crate::hardware::NeuronChipType::Trainium,
            core_count: 32,
            ..
        }
    ));
}

#[cfg(feature = "aws-neuron")]
#[test]
fn neuron_parser_empty_array() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let ok = crate::detect::neuron::parse_neuron_output("[]", &mut profiles, &mut warnings);
    assert!(ok);
    assert!(profiles.is_empty());
}

#[cfg(feature = "aws-neuron")]
#[test]
fn neuron_parser_invalid_json_returns_false() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let ok = crate::detect::neuron::parse_neuron_output("not json", &mut profiles, &mut warnings);
    assert!(!ok);
}

#[cfg(feature = "aws-neuron")]
#[test]
fn neuron_parser_missing_nc_count_skips() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let json = r#"[{"model":"inf2","memory_per_nc_mb":32768}]"#;
    let ok = crate::detect::neuron::parse_neuron_output(json, &mut profiles, &mut warnings);
    assert!(ok);
    assert!(profiles.is_empty());
}

// ---------------------------------------------------------------------------
// Intel oneAPI parser
// ---------------------------------------------------------------------------

#[cfg(feature = "intel-oneapi")]
#[test]
fn xpu_smi_single_device() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::intel_oneapi::parse_xpu_smi_output(
        "DeviceId,Name,MemorySize,Other\n0, Intel Arc A770, 16384, value\n",
        &mut profiles,
        &mut warnings,
    );
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].memory_bytes, 16384 * 1024 * 1024);
    assert!(matches!(
        profiles[0].accelerator,
        AcceleratorType::IntelOneApi { device_id: 0 }
    ));
}

#[cfg(feature = "intel-oneapi")]
#[test]
fn xpu_smi_skips_header() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::intel_oneapi::parse_xpu_smi_output(
        "DeviceId,Name,MemorySize,Other\n",
        &mut profiles,
        &mut warnings,
    );
    assert!(profiles.is_empty());
}

#[cfg(feature = "intel-oneapi")]
#[test]
fn xpu_smi_too_few_fields_skips() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::intel_oneapi::parse_xpu_smi_output("0, Arc\n", &mut profiles, &mut warnings);
    assert!(profiles.is_empty());
}

#[cfg(feature = "intel-oneapi")]
#[test]
fn xpu_smi_invalid_device_id() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::intel_oneapi::parse_xpu_smi_output(
        "abc, Arc, 8192, v\n",
        &mut profiles,
        &mut warnings,
    );
    assert!(profiles.is_empty());
    assert_eq!(warnings.len(), 1);
}

// ---------------------------------------------------------------------------
// Cerebras parser
// ---------------------------------------------------------------------------

#[cfg(feature = "cerebras")]
#[test]
fn cerebras_memory_gb() {
    let result = crate::detect::cerebras::parse_memory_from_cli("Memory: 44GB SRAM\n");
    assert_eq!(result, Some(44 * 1024 * 1024 * 1024));
}

#[cfg(feature = "cerebras")]
#[test]
fn cerebras_memory_sram_keyword() {
    let result = crate::detect::cerebras::parse_memory_from_cli("SRAM: 44GB\n");
    assert_eq!(result, Some(44 * 1024 * 1024 * 1024));
}

#[cfg(feature = "cerebras")]
#[test]
fn cerebras_memory_no_match() {
    let result = crate::detect::cerebras::parse_memory_from_cli("Cores: 850000\n");
    assert_eq!(result, None);
}

#[cfg(feature = "cerebras")]
#[test]
fn cerebras_memory_empty() {
    assert_eq!(crate::detect::cerebras::parse_memory_from_cli(""), None);
}

// ---------------------------------------------------------------------------
// Graphcore parser
// ---------------------------------------------------------------------------

#[cfg(feature = "graphcore")]
#[test]
fn graphcore_json_memory() {
    let result = crate::detect::graphcore::parse_memory_from_gcinfo(r#"{"memory": 943718400}"#);
    assert_eq!(result, Some(943718400));
}

#[cfg(feature = "graphcore")]
#[test]
fn graphcore_json_sram_size() {
    let result = crate::detect::graphcore::parse_memory_from_gcinfo(r#"{"sram_size": 943718400}"#);
    assert_eq!(result, Some(943718400));
}

#[cfg(feature = "graphcore")]
#[test]
fn graphcore_text_mb() {
    let result = crate::detect::graphcore::parse_memory_from_gcinfo("Memory: 900MB\nTiles: 1472\n");
    assert_eq!(result, Some(900 * 1024 * 1024));
}

#[cfg(feature = "graphcore")]
#[test]
fn graphcore_text_gb() {
    let result = crate::detect::graphcore::parse_memory_from_gcinfo("SRAM: 1GB\n");
    assert_eq!(result, Some(1024 * 1024 * 1024));
}

#[cfg(feature = "graphcore")]
#[test]
fn graphcore_empty() {
    assert_eq!(crate::detect::graphcore::parse_memory_from_gcinfo(""), None);
}

#[cfg(feature = "graphcore")]
#[test]
fn graphcore_no_memory_fields() {
    let result = crate::detect::graphcore::parse_memory_from_gcinfo(r#"{"tiles": 1472}"#);
    assert_eq!(result, None);
}

// ===========================================================================
// Cloud hardware validation fixtures
//
// Realistic tool output for production cloud accelerators. Validates parsers
// against documented specs (memory, compute capability, device names).
// ===========================================================================

// ---------------------------------------------------------------------------
// NVIDIA A100 80GB SXM (AWS p4d.24xlarge / GCP a2-megagpu-16g)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[test]
fn cloud_fixture_a100_80gb_8gpu() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    // 8x A100 80GB SXM, nvidia-smi 11-field CSV
    let output = "\
0, 81920, 512, 81408, 8.0, 535.129.03, NVIDIA A100-SXM4-80GB, 34, 62.30, 0, 1593
1, 81920, 0, 81920, 8.0, 535.129.03, NVIDIA A100-SXM4-80GB, 33, 58.10, 0, 1593
2, 81920, 256, 81664, 8.0, 535.129.03, NVIDIA A100-SXM4-80GB, 35, 63.50, 0, 1593
3, 81920, 0, 81920, 8.0, 535.129.03, NVIDIA A100-SXM4-80GB, 32, 56.20, 0, 1593
4, 81920, 0, 81920, 8.0, 535.129.03, NVIDIA A100-SXM4-80GB, 34, 61.70, 0, 1593
5, 81920, 128, 81792, 8.0, 535.129.03, NVIDIA A100-SXM4-80GB, 35, 64.30, 0, 1593
6, 81920, 0, 81920, 8.0, 535.129.03, NVIDIA A100-SXM4-80GB, 33, 59.10, 0, 1593
7, 81920, 0, 81920, 8.0, 535.129.03, NVIDIA A100-SXM4-80GB, 34, 60.80, 0, 1593
";
    crate::detect::cuda::parse_cuda_output(output, &mut profiles, &mut warnings);
    assert!(warnings.is_empty(), "unexpected warnings: {:?}", warnings);
    assert_eq!(profiles.len(), 8);
    for (i, p) in profiles.iter().enumerate() {
        assert!(
            matches!(p.accelerator, AcceleratorType::CudaGpu { device_id } if device_id == i as u32)
        );
        assert_eq!(p.compute_capability.as_deref(), Some("8.0"));
        assert_eq!(p.device_name.as_deref(), Some("NVIDIA A100-SXM4-80GB"));
        // A100 SXM: 1593 MHz * 5120 bit * 2 / 8 / 1000 ≈ 2039 GB/s
        assert!(p.memory_bandwidth_gbps.unwrap() > 1900.0, "A100 BW too low");
        // ~80 GB VRAM
        assert!(p.memory_bytes >= 80 * 1024 * 1024 * 1024);
    }
}

// ---------------------------------------------------------------------------
// NVIDIA H100 80GB SXM (AWS p5.48xlarge / GCP a3-highgpu-8g)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[test]
fn cloud_fixture_h100_80gb_sxm() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let output = "\
0, 81920, 1024, 80896, 9.0, 550.90.07, NVIDIA H100 80GB HBM3, 38, 275.00, 5, 2619
1, 81920, 0, 81920, 9.0, 550.90.07, NVIDIA H100 80GB HBM3, 37, 268.50, 3, 2619
";
    crate::detect::cuda::parse_cuda_output(output, &mut profiles, &mut warnings);
    assert!(warnings.is_empty());
    assert_eq!(profiles.len(), 2);
    let p = &profiles[0];
    assert_eq!(p.compute_capability.as_deref(), Some("9.0"));
    assert_eq!(p.device_name.as_deref(), Some("NVIDIA H100 80GB HBM3"));
    // H100 SXM: 2619 MHz * 5120 bit * 2 / 8 / 1000 ≈ 3352 GB/s
    assert!(p.memory_bandwidth_gbps.unwrap() > 3300.0, "H100 BW too low");
    assert_eq!(p.temperature_c, Some(38));
    assert!((p.power_watts.unwrap() - 275.0).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// NVIDIA Grace Hopper GH200 (unified memory: 96 GB HBM3 + 480 GB LPDDR5X)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[test]
fn cloud_fixture_grace_hopper_gh200() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    // GH200: nvidia-smi reports ~96 GB HBM, gpu_name contains "GH200"
    // Parser should add 480 GB unified memory for Grace Hopper
    let output =
        "0, 98304, 1024, 97280, 9.0, 550.90.07, NVIDIA GH200 120GB, 42, 300.00, 10, 2619\n";
    crate::detect::cuda::parse_cuda_output(output, &mut profiles, &mut warnings);
    assert!(warnings.is_empty());
    assert_eq!(profiles.len(), 1);
    let p = &profiles[0];
    assert_eq!(p.device_name.as_deref(), Some("NVIDIA GH200 120GB"));
    // Should have 96 GB HBM + 480 GB unified ≈ 576 GB
    let mem_gb = p.memory_bytes / (1024 * 1024 * 1024);
    assert!(
        mem_gb > 500,
        "Grace Hopper should report >500 GB unified memory, got {} GB",
        mem_gb
    );
}

// ---------------------------------------------------------------------------
// Gaudi 3 8-device (AWS DL2q or Intel Developer Cloud)
// ---------------------------------------------------------------------------

#[cfg(feature = "gaudi")]
#[test]
fn cloud_fixture_gaudi3_8x() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    // Gaudi 3: 128 GB HBM2e per device, hl-smi CSV format
    let output = "\
0, hl-325-gaudi3, 131072, 100000
1, hl-325-gaudi3, 131072, 100000
2, hl-325-gaudi3, 131072, 100000
3, hl-325-gaudi3, 131072, 100000
4, hl-325-gaudi3, 131072, 100000
5, hl-325-gaudi3, 131072, 100000
6, hl-325-gaudi3, 131072, 100000
7, hl-325-gaudi3, 131072, 100000
";
    crate::detect::gaudi::parse_gaudi_output(output, &mut profiles, &mut warnings);
    assert!(warnings.is_empty());
    assert_eq!(profiles.len(), 8);
    for (i, p) in profiles.iter().enumerate() {
        assert!(matches!(
            p.accelerator,
            AcceleratorType::Gaudi {
                device_id,
                generation: crate::hardware::GaudiGeneration::Gaudi3,
            } if device_id == i as u32
        ));
        // 131072 MB = 128 GB HBM2e
        assert_eq!(p.memory_bytes, 131072 * 1024 * 1024);
    }
}

// ---------------------------------------------------------------------------
// AWS Neuron: trn1.32xlarge (16 Trainium chips, 32 NeuronCores)
// ---------------------------------------------------------------------------

#[cfg(feature = "aws-neuron")]
#[test]
fn cloud_fixture_trn1_32xlarge() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    // neuron-ls JSON: trn1.32xlarge has 16 Trainium chips, 32 NeuronCores
    let json = r#"[
        {"model":"trn1.32xlarge","nc_count":32,"memory_per_nc_mb":16384}
    ]"#;
    let ok = crate::detect::neuron::parse_neuron_output(json, &mut profiles, &mut warnings);
    assert!(ok);
    assert_eq!(profiles.len(), 1);
    let p = &profiles[0];
    assert!(matches!(
        p.accelerator,
        AcceleratorType::AwsNeuron {
            chip_type: crate::hardware::NeuronChipType::Trainium,
            core_count: 32,
            ..
        }
    ));
    // 32 cores * 16384 MB = 512 GB total
    assert_eq!(p.memory_bytes, 32 * 16384 * 1024 * 1024);
}

// ---------------------------------------------------------------------------
// AWS Neuron: inf2.48xlarge (12 Inferentia2 chips, 24 NeuronCores)
// ---------------------------------------------------------------------------

#[cfg(feature = "aws-neuron")]
#[test]
fn cloud_fixture_inf2_48xlarge() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    let json = r#"[
        {"model":"inf2.48xlarge","nc_count":24,"memory_per_nc_mb":32768}
    ]"#;
    let ok = crate::detect::neuron::parse_neuron_output(json, &mut profiles, &mut warnings);
    assert!(ok);
    assert_eq!(profiles.len(), 1);
    let p = &profiles[0];
    assert!(matches!(
        p.accelerator,
        AcceleratorType::AwsNeuron {
            chip_type: crate::hardware::NeuronChipType::Inferentia,
            core_count: 24,
            ..
        }
    ));
    // 24 cores * 32768 MB = 768 GB total
    assert_eq!(p.memory_bytes, 24 * 32768 * 1024 * 1024);
}

// ---------------------------------------------------------------------------
// Vulkan: AMD MI300X 192GB (Azure ND MI300X v5)
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan")]
#[test]
fn cloud_fixture_mi300x_vulkan() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    // MI300X appears in vulkaninfo with 192 GB HBM3
    let summary = "\
GPU0:
\tdeviceName = AMD Instinct MI300X
\tapiVersion = 1.3.280
\tdriverVersion = 24.20.3
\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tsize = 196608 MiB
";
    crate::detect::vulkan::parse_vulkan_output(summary, None, &mut profiles, &mut warnings);
    assert_eq!(profiles.len(), 1);
    let p = &profiles[0];
    assert!(matches!(
        p.accelerator,
        AcceleratorType::VulkanGpu { device_id: 0 }
    ));
    assert_eq!(p.device_name.as_deref(), Some("AMD Instinct MI300X"));
    // 196608 MiB ≈ 192 GB
    assert!(p.memory_bytes >= 192 * 1024 * 1024 * 1024);
}

// ===========================================================================
// Windows WMI/PowerShell parser tests
// ===========================================================================

#[cfg(feature = "windows-wmi")]
#[test]
fn wmic_parser_single_nvidia_gpu() {
    let mut profiles = Vec::new();
    let output = "\
Node,AdapterRAM,DriverVersion,Name,VideoProcessor
DESKTOP-ABC,8589934592,31.0.15.5250,NVIDIA GeForce RTX 4070,AD104
";
    let ok = crate::detect::windows::parse_wmic_output(output, &mut profiles);
    assert!(ok);
    assert_eq!(profiles.len(), 1);
    assert_eq!(
        profiles[0].device_name.as_deref(),
        Some("NVIDIA GeForce RTX 4070")
    );
    assert_eq!(profiles[0].memory_bytes, 8589934592); // 8 GB
    assert_eq!(profiles[0].driver_version.as_deref(), Some("31.0.15.5250"));
}

#[cfg(feature = "windows-wmi")]
#[test]
fn wmic_parser_multi_gpu() {
    let mut profiles = Vec::new();
    let output = "\
Node,AdapterRAM,DriverVersion,Name,VideoProcessor
DESKTOP-ABC,8589934592,31.0.15.5250,NVIDIA GeForce RTX 4070,AD104
DESKTOP-ABC,4293918720,27.21.14.5671,AMD Radeon RX 580,Polaris 20
";
    let ok = crate::detect::windows::parse_wmic_output(output, &mut profiles);
    assert!(ok);
    assert_eq!(profiles.len(), 2);
    assert_eq!(
        profiles[0].device_name.as_deref(),
        Some("NVIDIA GeForce RTX 4070")
    );
    assert_eq!(
        profiles[1].device_name.as_deref(),
        Some("AMD Radeon RX 580")
    );
}

#[cfg(feature = "windows-wmi")]
#[test]
fn wmic_parser_skips_virtual_devices() {
    let mut profiles = Vec::new();
    let output = "\
Node,AdapterRAM,DriverVersion,Name,VideoProcessor
DESKTOP,1048576,,Microsoft Basic Display Adapter,
DESKTOP,8589934592,31.0.15.5250,NVIDIA GeForce RTX 4090,AD102
DESKTOP,0,,Microsoft Remote Desktop Virtual Adapter,
";
    let ok = crate::detect::windows::parse_wmic_output(output, &mut profiles);
    assert!(ok);
    assert_eq!(profiles.len(), 1); // Only the real GPU
    assert_eq!(
        profiles[0].device_name.as_deref(),
        Some("NVIDIA GeForce RTX 4090")
    );
}

#[cfg(feature = "windows-wmi")]
#[test]
fn wmic_parser_empty() {
    let mut profiles = Vec::new();
    assert!(!crate::detect::windows::parse_wmic_output(
        "",
        &mut profiles
    ));
    assert!(profiles.is_empty());
}

#[cfg(feature = "windows-wmi")]
#[test]
fn powershell_csv_parser_nvidia() {
    let mut profiles = Vec::new();
    let output = r#""Name","AdapterRAM","DriverVersion"
"NVIDIA GeForce RTX 3080","10737418240","31.0.15.3623"
"#;
    let ok = crate::detect::windows::parse_powershell_csv(output, &mut profiles);
    assert!(ok);
    assert_eq!(profiles.len(), 1);
    assert_eq!(
        profiles[0].device_name.as_deref(),
        Some("NVIDIA GeForce RTX 3080")
    );
    assert_eq!(profiles[0].memory_bytes, 10737418240); // 10 GB
}

#[cfg(feature = "windows-wmi")]
#[test]
fn powershell_csv_parser_skips_virtual() {
    let mut profiles = Vec::new();
    let output = r#""Name","AdapterRAM","DriverVersion"
"Microsoft Basic Display Adapter","1048576",""
"NVIDIA GeForce RTX 4090","25769803776","32.0.15.6081"
"#;
    let ok = crate::detect::windows::parse_powershell_csv(output, &mut profiles);
    assert!(ok);
    assert_eq!(profiles.len(), 1);
    assert_eq!(
        profiles[0].device_name.as_deref(),
        Some("NVIDIA GeForce RTX 4090")
    );
}
