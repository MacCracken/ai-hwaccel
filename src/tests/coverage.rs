//! Tests specifically targeting coverage gaps identified by llvm-cov.

use crate::*;

// ---------------------------------------------------------------------------
// Registry: suggest_quantization edge cases
// ---------------------------------------------------------------------------

#[test]
fn suggest_quantization_tpu_drops_to_int8() {
    // TPU V5e: 16 GB HBM. 10B at BF16 = 24 GB (doesn't fit).
    // INT8 = 12 GB (fits!) → should return INT8.
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::tpu(0, 1, TpuVersion::V5e), // 16 GB HBM
    ]);
    let q = reg.suggest_quantization(10_000_000_000);
    assert_eq!(q, QuantizationLevel::Int8);
}

#[test]
fn suggest_quantization_gaudi_drops_to_int8() {
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi2), // 96 GB HBM
    ]);
    // ~60B params at BF16 = 120 GB (doesn't fit in 96 GB) → drops to INT8
    let q = reg.suggest_quantization(60_000_000_000);
    assert_eq!(q, QuantizationLevel::Int8);
}

#[test]
fn suggest_quantization_npu_drops_to_int4() {
    // NPU with very small memory
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(4 * 1024 * 1024 * 1024),
        AcceleratorProfile {
            accelerator: AcceleratorType::IntelNpu,
            available: true,
            memory_bytes: 1 * 1024 * 1024 * 1024, // 1 GB
            compute_capability: None,
            driver_version: None,
            memory_bandwidth_gbps: None,
            memory_used_bytes: None,
            memory_free_bytes: None,
            pcie_bandwidth_gbps: None,
            numa_node: None,
            temperature_c: None,
            power_watts: None,
            gpu_utilization_percent: None,
        },
    ]);
    // 7B at INT8 = ~8.4 GB (doesn't fit in 1 GB NPU) → drops to INT4
    let q = reg.suggest_quantization(7_000_000_000);
    assert_eq!(q, QuantizationLevel::Int4);
}

#[test]
fn suggest_quantization_cpu_fallback_tiny_memory() {
    // Only CPU with very limited memory
    let reg = AcceleratorRegistry::from_profiles(vec![AcceleratorProfile::cpu(
        2 * 1024 * 1024 * 1024,
    )]);
    // 70B model: won't fit even at INT4 on 2 GB → returns INT4 anyway (best effort)
    let q = reg.suggest_quantization(70_000_000_000);
    assert_eq!(q, QuantizationLevel::Int4);
}

#[test]
fn suggest_quantization_unavailable_devices_skipped() {
    let mut gpu = AcceleratorProfile::cuda(0, 80 * 1024 * 1024 * 1024);
    gpu.available = false;
    let reg = AcceleratorRegistry::from_profiles(vec![
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        gpu,
    ]);
    // GPU is unavailable, so CPU fallback
    let q = reg.suggest_quantization(7_000_000_000);
    // Should pick FP16 (CPU has 16 GB, 7B FP16 ≈ 16.8 GB → drops to INT8)
    assert!(matches!(
        q,
        QuantizationLevel::Int8 | QuantizationLevel::Float16
    ));
}

// ---------------------------------------------------------------------------
// Registry: Default impl
// ---------------------------------------------------------------------------

#[test]
fn registry_default_impl() {
    let reg = AcceleratorRegistry::default();
    assert!(!reg.all_profiles().is_empty());
    assert!(matches!(
        reg.all_profiles()[0].accelerator,
        AcceleratorType::Cpu
    ));
}

// ---------------------------------------------------------------------------
// Registry: from_json deserialization triggers default_schema_version
// ---------------------------------------------------------------------------

#[test]
fn from_json_without_schema_version_uses_default() {
    // JSON without schema_version field — should use default
    let json = r#"{"profiles":[],"system_io":{"interconnects":[],"storage":[]}}"#;
    let reg = AcceleratorRegistry::from_json(json).unwrap();
    assert_eq!(reg.schema_version(), SCHEMA_VERSION);
}

// ---------------------------------------------------------------------------
// DetectBuilder: all with_* methods
// ---------------------------------------------------------------------------

#[test]
fn builder_all_with_methods() {
    // Exercise every with_* method to hit coverage
    let builder = DetectBuilder::none()
        .with_cuda()
        .with_rocm()
        .with_apple()
        .with_vulkan()
        .with_intel_npu()
        .with_amd_xdna()
        .with_tpu()
        .with_gaudi()
        .with_aws_neuron()
        .with_intel_oneapi()
        .with_qualcomm()
        .with_cerebras()
        .with_graphcore()
        .with_groq()
        .with_samsung_npu()
        .with_mediatek_apu();
    assert_eq!(builder.enabled_count(), Backend::ALL.len());
}

#[test]
fn builder_all_without_methods() {
    let builder = DetectBuilder::new()
        .without_cuda()
        .without_rocm()
        .without_apple()
        .without_vulkan()
        .without_intel_npu()
        .without_amd_xdna()
        .without_tpu()
        .without_gaudi()
        .without_aws_neuron()
        .without_intel_oneapi()
        .without_qualcomm()
        .without_cerebras()
        .without_graphcore()
        .without_groq()
        .without_samsung_npu()
        .without_mediatek_apu();
    assert_eq!(builder.enabled_count(), 0);
}

// ---------------------------------------------------------------------------
// Bandwidth: parse_nvidia_bandwidth_output
// ---------------------------------------------------------------------------

#[test]
fn parse_nvidia_bandwidth_output_valid() {
    let output = "2619, 9.0\n10501, 8.9\n";
    let results = crate::detect::bandwidth::parse_nvidia_bandwidth_output(output);
    assert_eq!(results.len(), 2);
    // H100: 2619 MHz, CC 9.0, 5120-bit → ~3352 GB/s
    assert!(results[0].is_some());
    assert!(results[0].unwrap() > 3000.0);
    // RTX 4090: 10501 MHz, CC 8.9, 384-bit → ~1008 GB/s
    assert!(results[1].is_some());
    assert!(results[1].unwrap() > 900.0);
}

#[test]
fn parse_nvidia_bandwidth_output_unknown_cc() {
    let output = "1000, 5.0\n"; // Unknown compute capability
    let results = crate::detect::bandwidth::parse_nvidia_bandwidth_output(output);
    assert_eq!(results.len(), 1);
    assert!(results[0].is_none()); // No bus width known for CC 5.0
}

#[test]
fn parse_nvidia_bandwidth_output_empty() {
    let results = crate::detect::bandwidth::parse_nvidia_bandwidth_output("");
    assert!(results.is_empty());
}

#[test]
fn parse_nvidia_bandwidth_output_malformed() {
    let results = crate::detect::bandwidth::parse_nvidia_bandwidth_output("garbage\n\n");
    assert_eq!(results.len(), 2); // Two lines → two None results
    assert!(results[0].is_none());
}

// ---------------------------------------------------------------------------
// Bandwidth: estimate_nvidia_bandwidth_from_cc
// ---------------------------------------------------------------------------

#[test]
fn estimate_bandwidth_from_cc_known() {
    let bw = crate::detect::bandwidth::estimate_nvidia_bandwidth_from_cc("9.0");
    assert!(bw.is_some());
    assert!(bw.unwrap() > 3000.0); // H100: 3350 GB/s
}

#[test]
fn estimate_bandwidth_from_cc_unknown() {
    let bw = crate::detect::bandwidth::estimate_nvidia_bandwidth_from_cc("99.9");
    assert!(bw.is_none());
}

// ---------------------------------------------------------------------------
// Interconnect: parse helpers
// ---------------------------------------------------------------------------

#[test]
fn parse_nvlink_output_multi_gpu() {
    let output = "\
GPU 0: NVIDIA H100 (UUID: GPU-aaa)
    Link 0: 25 GB/s
    Link 1: 25 GB/s
GPU 1: NVIDIA H100 (UUID: GPU-bbb)
    Link 0: 25 GB/s
";
    let mut interconnects = Vec::new();
    crate::detect::interconnect::parse_nvlink_output(output, &mut interconnects);
    assert_eq!(interconnects.len(), 2);
    assert_eq!(interconnects[0].bandwidth_gbps, 50.0); // 2 links × 25
    assert_eq!(interconnects[1].bandwidth_gbps, 25.0); // 1 link × 25
}

#[test]
fn parse_nvlink_output_empty() {
    let mut interconnects = Vec::new();
    crate::detect::interconnect::parse_nvlink_output("", &mut interconnects);
    assert!(interconnects.is_empty());
}

#[test]
fn parse_nvlink_output_no_links() {
    let output = "GPU 0: NVIDIA RTX 4090\n";
    let mut interconnects = Vec::new();
    crate::detect::interconnect::parse_nvlink_output(output, &mut interconnects);
    assert!(interconnects.is_empty()); // No links → no interconnect entry
}

// ---------------------------------------------------------------------------
// Hardware: new backend ranks and multipliers
// ---------------------------------------------------------------------------

#[test]
fn hardware_mod_all_variants_have_rank() {
    // Ensure every type has a non-zero rank (except CPU which is 10)
    let types: Vec<AcceleratorType> = vec![
        AcceleratorType::Cpu,
        AcceleratorType::CudaGpu { device_id: 0 },
        AcceleratorType::RocmGpu { device_id: 0 },
        AcceleratorType::MetalGpu,
        AcceleratorType::VulkanGpu { device_id: 0, device_name: "x".into() },
        AcceleratorType::IntelNpu,
        AcceleratorType::AppleNpu,
        AcceleratorType::AmdXdnaNpu { device_id: 0 },
        AcceleratorType::Tpu { device_id: 0, chip_count: 1, version: TpuVersion::V4 },
        AcceleratorType::Tpu { device_id: 0, chip_count: 1, version: TpuVersion::V5e },
        AcceleratorType::Tpu { device_id: 0, chip_count: 1, version: TpuVersion::V5p },
        AcceleratorType::Gaudi { device_id: 0, generation: GaudiGeneration::Gaudi2 },
        AcceleratorType::Gaudi { device_id: 0, generation: GaudiGeneration::Gaudi3 },
        AcceleratorType::AwsNeuron { device_id: 0, chip_type: NeuronChipType::Inferentia, core_count: 2 },
        AcceleratorType::AwsNeuron { device_id: 0, chip_type: NeuronChipType::Trainium, core_count: 2 },
        AcceleratorType::QualcommAi100 { device_id: 0 },
        AcceleratorType::IntelOneApi { device_id: 0 },
        AcceleratorType::CerebrasWse { device_id: 0 },
        AcceleratorType::GraphcoreIpu { device_id: 0 },
        AcceleratorType::GroqLpu { device_id: 0 },
        AcceleratorType::SamsungNpu { device_id: 0 },
        AcceleratorType::MediaTekApu { device_id: 0 },
    ];
    for t in &types {
        assert!(t.rank() > 0, "{:?} should have rank > 0", t);
        assert!(t.throughput_multiplier() > 0.0, "{:?} should have positive throughput", t);
    }
}

// ---------------------------------------------------------------------------
// CUDA parser: edge cases
// ---------------------------------------------------------------------------

#[test]
fn cuda_parser_empty_fields() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    // 6 fields minimum, empty compute_cap and driver_version
    crate::detect::cuda::parse_cuda_output(
        "0, 8192, 4096, 4096, , ,\n",
        &mut profiles,
        &mut warnings,
    );
    assert_eq!(profiles.len(), 1);
    assert!(profiles[0].compute_capability.is_none());
    assert!(profiles[0].driver_version.is_none());
}

#[test]
fn cuda_parser_too_few_fields() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::cuda::parse_cuda_output("0, 8192\n", &mut profiles, &mut warnings);
    assert!(profiles.is_empty());
    assert_eq!(warnings.len(), 1);
}

#[test]
fn cuda_parser_grace_hopper() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    // Simulate GH200: 96 GB HBM, CC 9.0, name contains "GH200"
    crate::detect::cuda::parse_cuda_output(
        "0, 98304, 1000, 97304, 9.0, 550.00, NVIDIA GH200, 45, 300, 10, 2619\n",
        &mut profiles,
        &mut warnings,
    );
    assert_eq!(profiles.len(), 1);
    // Should have added 480 GB unified memory
    let mem_gb = profiles[0].memory_bytes / (1024 * 1024 * 1024);
    assert!(mem_gb > 500, "Grace Hopper should report >500 GB, got {}", mem_gb);
}

// ---------------------------------------------------------------------------
// Gaudi parser
// ---------------------------------------------------------------------------

#[test]
fn gaudi_parser_valid_line() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::gaudi::parse_gaudi_output(
        "0, hl-325-gaudi3, 131072, 100000\n",
        &mut profiles,
        &mut warnings,
    );
    assert_eq!(profiles.len(), 1);
    assert!(matches!(
        profiles[0].accelerator,
        AcceleratorType::Gaudi { generation: GaudiGeneration::Gaudi3, .. }
    ));
}

// ---------------------------------------------------------------------------
// Vulkan parser
// ---------------------------------------------------------------------------

#[test]
fn vulkan_parser_no_devices() {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::vulkan::parse_vulkan_output(
        "Vulkan Instance Version: 1.3.0\n",
        None,
        &mut profiles,
        &mut warnings,
    );
    assert_eq!(profiles.len(), 1); // Fallback generic device
    assert!(matches!(
        profiles[0].accelerator,
        AcceleratorType::VulkanGpu { .. }
    ));
}
