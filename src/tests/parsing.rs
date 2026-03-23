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
        crate::hardware::AcceleratorType::VulkanGpu { device_name, .. }
        if device_name.contains("AMD Radeon")
    ));
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
// Apple parser — parse_system_profiler_output with fixture data
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
