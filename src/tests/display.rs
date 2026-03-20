//! Display formatting tests.

use crate::*;

// ---------------------------------------------------------------------------
// AcceleratorType — all 13 variants
// ---------------------------------------------------------------------------

#[test]
fn display_accelerator_type_all_variants() {
    assert_eq!(AcceleratorType::Cpu.to_string(), "CPU");
    assert_eq!(
        AcceleratorType::CudaGpu { device_id: 3 }.to_string(),
        "CUDA GPU (device 3)"
    );
    assert_eq!(
        AcceleratorType::RocmGpu { device_id: 1 }.to_string(),
        "ROCm GPU (device 1)"
    );
    assert_eq!(AcceleratorType::MetalGpu.to_string(), "Metal GPU");
    assert_eq!(
        AcceleratorType::VulkanGpu {
            device_id: 0,
            device_name: "RTX 4090".into()
        }
        .to_string(),
        "Vulkan GPU (device 0, RTX 4090)"
    );
    assert_eq!(AcceleratorType::IntelNpu.to_string(), "Intel NPU");
    assert_eq!(
        AcceleratorType::AmdXdnaNpu { device_id: 2 }.to_string(),
        "AMD XDNA NPU (device 2)"
    );
    assert_eq!(AcceleratorType::AppleNpu.to_string(), "Apple Neural Engine");
    assert_eq!(
        AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 4,
            version: TpuVersion::V5p
        }
        .to_string(),
        "TPU v5p (device 0, 4 chips)"
    );
    assert_eq!(
        AcceleratorType::Gaudi {
            device_id: 0,
            generation: GaudiGeneration::Gaudi3
        }
        .to_string(),
        "Intel Gaudi3 (device 0)"
    );
    assert_eq!(
        AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Trainium,
            core_count: 4
        }
        .to_string(),
        "AWS Trainium (device 0, 4 cores)"
    );
    assert_eq!(
        AcceleratorType::AwsNeuron {
            device_id: 1,
            chip_type: NeuronChipType::Inferentia,
            core_count: 2
        }
        .to_string(),
        "AWS Inferentia (device 1, 2 cores)"
    );
    assert_eq!(
        AcceleratorType::QualcommAi100 { device_id: 0 }.to_string(),
        "Qualcomm Cloud AI 100 (device 0)"
    );
    assert_eq!(
        AcceleratorType::IntelOneApi { device_id: 0 }.to_string(),
        "Intel oneAPI GPU (device 0)"
    );
}

// ---------------------------------------------------------------------------
// Sub-types
// ---------------------------------------------------------------------------

#[test]
fn display_tpu_version() {
    assert_eq!(TpuVersion::V4.to_string(), "v4");
    assert_eq!(TpuVersion::V5e.to_string(), "v5e");
    assert_eq!(TpuVersion::V5p.to_string(), "v5p");
}

#[test]
fn display_gaudi_generation() {
    assert_eq!(GaudiGeneration::Gaudi2.to_string(), "Gaudi2");
    assert_eq!(GaudiGeneration::Gaudi3.to_string(), "Gaudi3");
}

#[test]
fn display_neuron_chip_type() {
    assert_eq!(NeuronChipType::Inferentia.to_string(), "Inferentia");
    assert_eq!(NeuronChipType::Trainium.to_string(), "Trainium");
}

// ---------------------------------------------------------------------------
// AcceleratorFamily
// ---------------------------------------------------------------------------

#[test]
fn display_accelerator_family() {
    assert_eq!(AcceleratorFamily::Cpu.to_string(), "CPU");
    assert_eq!(AcceleratorFamily::Gpu.to_string(), "GPU");
    assert_eq!(AcceleratorFamily::Npu.to_string(), "NPU");
    assert_eq!(AcceleratorFamily::Tpu.to_string(), "TPU");
    assert_eq!(AcceleratorFamily::AiAsic.to_string(), "AI ASIC");
}

// ---------------------------------------------------------------------------
// QuantizationLevel
// ---------------------------------------------------------------------------

#[test]
fn display_quantization() {
    assert_eq!(QuantizationLevel::None.to_string(), "FP32");
    assert_eq!(QuantizationLevel::Float16.to_string(), "FP16");
    assert_eq!(QuantizationLevel::BFloat16.to_string(), "BF16");
    assert_eq!(QuantizationLevel::Int8.to_string(), "INT8");
    assert_eq!(QuantizationLevel::Int4.to_string(), "INT4");
}

// ---------------------------------------------------------------------------
// AcceleratorRequirement — all 7 variants
// ---------------------------------------------------------------------------

#[test]
fn display_requirement_all_variants() {
    assert_eq!(AcceleratorRequirement::None.to_string(), "none");
    assert_eq!(AcceleratorRequirement::Gpu.to_string(), "gpu");
    assert_eq!(
        AcceleratorRequirement::Tpu { min_chips: 4 }.to_string(),
        "tpu(4+ chips)"
    );
    assert_eq!(AcceleratorRequirement::Gaudi.to_string(), "gaudi");
    assert_eq!(AcceleratorRequirement::AwsNeuron.to_string(), "aws-neuron");
    assert_eq!(AcceleratorRequirement::GpuOrTpu.to_string(), "gpu-or-tpu");
    assert_eq!(
        AcceleratorRequirement::AnyAccelerator.to_string(),
        "any-accelerator"
    );
}

// ---------------------------------------------------------------------------
// ShardingStrategy
// ---------------------------------------------------------------------------

#[test]
fn display_sharding_strategy_all_variants() {
    assert_eq!(ShardingStrategy::None.to_string(), "None");
    assert_eq!(
        ShardingStrategy::PipelineParallel { num_stages: 4 }.to_string(),
        "Pipeline Parallel (4 stages)"
    );
    assert_eq!(
        ShardingStrategy::TensorParallel { num_devices: 8 }.to_string(),
        "Tensor Parallel (8 devices)"
    );
    assert_eq!(
        ShardingStrategy::DataParallel { num_replicas: 2 }.to_string(),
        "Data Parallel (2 replicas)"
    );
}

// ---------------------------------------------------------------------------
// AcceleratorProfile
// ---------------------------------------------------------------------------

#[test]
fn display_accelerator_profile() {
    let p = AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024);
    let s = p.to_string();
    assert!(s.contains("CUDA GPU"));
    assert!(s.contains("24.0 GB"));
    assert!(!s.contains("unavailable"));

    let mut unavail = AcceleratorProfile::cuda(0, 8 * 1024 * 1024 * 1024);
    unavail.available = false;
    let s = unavail.to_string();
    assert!(s.contains("unavailable"));
}

// ---------------------------------------------------------------------------
// DetectionError
// ---------------------------------------------------------------------------

#[test]
fn display_detection_error_all_variants() {
    let e = DetectionError::ToolNotFound {
        tool: "nvidia-smi".into(),
    };
    assert!(e.to_string().contains("nvidia-smi"));
    assert!(e.to_string().contains("not found"));

    let e = DetectionError::ToolFailed {
        tool: "hl-smi".into(),
        exit_code: Some(1),
        stderr: "device not found\nsecond line".into(),
    };
    let s = e.to_string();
    assert!(s.contains("hl-smi"));
    assert!(s.contains("1"));
    assert!(s.contains("device not found"));
    // Only first line of stderr shown
    assert!(!s.contains("second line"));

    let e = DetectionError::ToolFailed {
        tool: "x".into(),
        exit_code: None,
        stderr: String::new(),
    };
    assert!(e.to_string().contains("signal"));

    let e = DetectionError::ParseError {
        backend: "cuda".into(),
        message: "bad csv".into(),
    };
    assert!(e.to_string().contains("cuda"));
    assert!(e.to_string().contains("bad csv"));

    let e = DetectionError::Timeout {
        tool: "nvidia-smi".into(),
        timeout_secs: 5.0,
    };
    let s = e.to_string();
    assert!(s.contains("nvidia-smi"));
    assert!(s.contains("timed out"));
    assert!(s.contains("5.0s"));

    let e = DetectionError::SysfsReadError {
        path: "/sys/class/drm".into(),
        message: "permission denied".into(),
    };
    assert!(e.to_string().contains("/sys/class/drm"));
    assert!(e.to_string().contains("permission denied"));
}
