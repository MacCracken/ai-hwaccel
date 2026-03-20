//! Quantisation level and device support tests.

use crate::*;

// ---------------------------------------------------------------------------
// bits_per_param and memory_reduction_factor
// ---------------------------------------------------------------------------

#[test]
fn quantization_bits() {
    assert_eq!(QuantizationLevel::None.bits_per_param(), 32);
    assert_eq!(QuantizationLevel::Float16.bits_per_param(), 16);
    assert_eq!(QuantizationLevel::BFloat16.bits_per_param(), 16);
    assert_eq!(QuantizationLevel::Int8.bits_per_param(), 8);
    assert_eq!(QuantizationLevel::Int4.bits_per_param(), 4);
}

#[test]
fn quantization_memory_reduction_all_levels() {
    assert!((QuantizationLevel::None.memory_reduction_factor() - 1.0).abs() < f64::EPSILON);
    assert!((QuantizationLevel::Float16.memory_reduction_factor() - 2.0).abs() < f64::EPSILON);
    assert!((QuantizationLevel::BFloat16.memory_reduction_factor() - 2.0).abs() < f64::EPSILON);
    assert!((QuantizationLevel::Int8.memory_reduction_factor() - 4.0).abs() < f64::EPSILON);
    assert!((QuantizationLevel::Int4.memory_reduction_factor() - 8.0).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// supports_quantization — per device family
// ---------------------------------------------------------------------------

#[test]
fn gpu_supports_all_quantization() {
    let gpu = AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024);
    for q in [
        QuantizationLevel::None,
        QuantizationLevel::Float16,
        QuantizationLevel::BFloat16,
        QuantizationLevel::Int8,
        QuantizationLevel::Int4,
    ] {
        assert!(gpu.supports_quantization(&q), "GPU should support {:?}", q);
    }
}

#[test]
fn cpu_supports_all_quantization() {
    let cpu = AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024);
    for q in [
        QuantizationLevel::None,
        QuantizationLevel::Float16,
        QuantizationLevel::BFloat16,
        QuantizationLevel::Int8,
        QuantizationLevel::Int4,
    ] {
        assert!(cpu.supports_quantization(&q), "CPU should support {:?}", q);
    }
}

#[test]
fn tpu_supports_bf16_not_fp16() {
    let tpu = AcceleratorProfile::tpu(0, 1, TpuVersion::V5e);
    assert!(tpu.supports_quantization(&QuantizationLevel::None));
    assert!(tpu.supports_quantization(&QuantizationLevel::BFloat16));
    assert!(tpu.supports_quantization(&QuantizationLevel::Int8));
    assert!(!tpu.supports_quantization(&QuantizationLevel::Float16));
    assert!(!tpu.supports_quantization(&QuantizationLevel::Int4));
}

#[test]
fn npu_only_int_quantization() {
    for accel in [
        AcceleratorType::IntelNpu,
        AcceleratorType::AppleNpu,
        AcceleratorType::AmdXdnaNpu { device_id: 0 },
    ] {
        let p = AcceleratorProfile {
            accelerator: accel.clone(),
            available: true,
            memory_bytes: 2 * 1024 * 1024 * 1024,
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
        };
        assert!(
            !p.supports_quantization(&QuantizationLevel::None),
            "{:?} should not support FP32",
            accel
        );
        assert!(
            !p.supports_quantization(&QuantizationLevel::Float16),
            "{:?} should not support FP16",
            accel
        );
        assert!(p.supports_quantization(&QuantizationLevel::Int8));
        assert!(p.supports_quantization(&QuantizationLevel::Int4));
    }
}

#[test]
fn gaudi_supports_bf16_fp16_int8_not_int4() {
    let gaudi = AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi3);
    assert!(gaudi.supports_quantization(&QuantizationLevel::None));
    assert!(gaudi.supports_quantization(&QuantizationLevel::BFloat16));
    assert!(gaudi.supports_quantization(&QuantizationLevel::Float16));
    assert!(gaudi.supports_quantization(&QuantizationLevel::Int8));
    assert!(!gaudi.supports_quantization(&QuantizationLevel::Int4));
}

#[test]
fn neuron_supports_bf16_fp16_int8_not_int4() {
    let neuron = AcceleratorProfile {
        accelerator: AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Trainium,
            core_count: 2,
        },
        available: true,
        memory_bytes: 32 * 1024 * 1024 * 1024,
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
    };
    assert!(neuron.supports_quantization(&QuantizationLevel::None));
    assert!(neuron.supports_quantization(&QuantizationLevel::BFloat16));
    assert!(neuron.supports_quantization(&QuantizationLevel::Float16));
    assert!(neuron.supports_quantization(&QuantizationLevel::Int8));
    assert!(!neuron.supports_quantization(&QuantizationLevel::Int4));
}

#[test]
fn qualcomm_supports_fp16_int8_int4_not_fp32() {
    let qc = AcceleratorProfile {
        accelerator: AcceleratorType::QualcommAi100 { device_id: 0 },
        available: true,
        memory_bytes: 32 * 1024 * 1024 * 1024,
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
    };
    assert!(!qc.supports_quantization(&QuantizationLevel::None));
    assert!(qc.supports_quantization(&QuantizationLevel::Float16));
    assert!(qc.supports_quantization(&QuantizationLevel::Int8));
    assert!(qc.supports_quantization(&QuantizationLevel::Int4));
}

// ---------------------------------------------------------------------------
// preferred_quantization
// ---------------------------------------------------------------------------

#[test]
fn preferred_quantization_per_family() {
    assert_eq!(
        AcceleratorProfile::cpu(0).preferred_quantization(),
        QuantizationLevel::Float16
    );
    assert_eq!(
        AcceleratorProfile::cuda(0, 0).preferred_quantization(),
        QuantizationLevel::Float16
    );
    assert_eq!(
        AcceleratorProfile::rocm(0, 0).preferred_quantization(),
        QuantizationLevel::Float16
    );
    assert_eq!(
        AcceleratorProfile::tpu(0, 1, TpuVersion::V5p).preferred_quantization(),
        QuantizationLevel::BFloat16
    );
    assert_eq!(
        AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi2).preferred_quantization(),
        QuantizationLevel::BFloat16
    );

    let npu = AcceleratorProfile {
        accelerator: AcceleratorType::IntelNpu,
        available: true,
        memory_bytes: 0,
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
    };
    assert_eq!(npu.preferred_quantization(), QuantizationLevel::Int8);

    let neuron = AcceleratorProfile {
        accelerator: AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Inferentia,
            core_count: 2,
        },
        available: true,
        memory_bytes: 0,
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
    };
    assert_eq!(neuron.preferred_quantization(), QuantizationLevel::BFloat16);

    let qc = AcceleratorProfile {
        accelerator: AcceleratorType::QualcommAi100 { device_id: 0 },
        available: true,
        memory_bytes: 0,
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
    };
    assert_eq!(qc.preferred_quantization(), QuantizationLevel::Int8);
}

// ---------------------------------------------------------------------------
// HBM constants
// ---------------------------------------------------------------------------

#[test]
fn tpu_hbm_per_chip() {
    assert_eq!(TpuVersion::V4.hbm_per_chip_bytes(), 32 * 1024 * 1024 * 1024);
    assert_eq!(
        TpuVersion::V5e.hbm_per_chip_bytes(),
        16 * 1024 * 1024 * 1024
    );
    assert_eq!(
        TpuVersion::V5p.hbm_per_chip_bytes(),
        95 * 1024 * 1024 * 1024
    );
}

#[test]
fn gaudi_hbm() {
    assert_eq!(GaudiGeneration::Gaudi2.hbm_bytes(), 96 * 1024 * 1024 * 1024);
    assert_eq!(
        GaudiGeneration::Gaudi3.hbm_bytes(),
        128 * 1024 * 1024 * 1024
    );
}

#[test]
fn neuron_hbm_per_core() {
    assert_eq!(
        NeuronChipType::Inferentia.hbm_per_core_bytes(),
        16 * 1024 * 1024 * 1024
    );
    assert_eq!(
        NeuronChipType::Trainium.hbm_per_core_bytes(),
        32 * 1024 * 1024 * 1024
    );
}
