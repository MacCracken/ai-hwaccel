//! Quantisation level and device support tests.

use crate::*;

#[test]
fn quantization_bits() {
    assert_eq!(QuantizationLevel::None.bits_per_param(), 32);
    assert_eq!(QuantizationLevel::Float16.bits_per_param(), 16);
    assert_eq!(QuantizationLevel::BFloat16.bits_per_param(), 16);
    assert_eq!(QuantizationLevel::Int8.bits_per_param(), 8);
    assert_eq!(QuantizationLevel::Int4.bits_per_param(), 4);
}

#[test]
fn quantization_memory_reduction() {
    assert!((QuantizationLevel::None.memory_reduction_factor() - 1.0).abs() < f64::EPSILON);
    assert!((QuantizationLevel::Int4.memory_reduction_factor() - 8.0).abs() < f64::EPSILON);
}

#[test]
fn gpu_supports_all_quantization() {
    let gpu = AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: Some("8.6".into()),
        driver_version: None,
    };
    assert!(gpu.supports_quantization(&QuantizationLevel::None));
    assert!(gpu.supports_quantization(&QuantizationLevel::Float16));
    assert!(gpu.supports_quantization(&QuantizationLevel::BFloat16));
    assert!(gpu.supports_quantization(&QuantizationLevel::Int8));
    assert!(gpu.supports_quantization(&QuantizationLevel::Int4));
}

#[test]
fn tpu_supports_bf16_not_fp16() {
    let tpu = AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 1,
            version: TpuVersion::V5e,
        },
        available: true,
        memory_bytes: 16 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    };
    assert!(tpu.supports_quantization(&QuantizationLevel::None));
    assert!(tpu.supports_quantization(&QuantizationLevel::BFloat16));
    assert!(tpu.supports_quantization(&QuantizationLevel::Int8));
    assert!(!tpu.supports_quantization(&QuantizationLevel::Float16));
    assert!(!tpu.supports_quantization(&QuantizationLevel::Int4));
}

#[test]
fn npu_only_int_quantization() {
    let npu = AcceleratorProfile {
        accelerator: AcceleratorType::IntelNpu,
        available: true,
        memory_bytes: 2 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    };
    assert!(!npu.supports_quantization(&QuantizationLevel::None));
    assert!(!npu.supports_quantization(&QuantizationLevel::Float16));
    assert!(npu.supports_quantization(&QuantizationLevel::Int8));
    assert!(npu.supports_quantization(&QuantizationLevel::Int4));
}

#[test]
fn preferred_quantization_per_family() {
    let gpu = AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 0,
        compute_capability: None,
        driver_version: None,
    };
    assert_eq!(gpu.preferred_quantization(), QuantizationLevel::Float16);

    let tpu = AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 1,
            version: TpuVersion::V5p,
        },
        available: true,
        memory_bytes: 0,
        compute_capability: None,
        driver_version: None,
    };
    assert_eq!(tpu.preferred_quantization(), QuantizationLevel::BFloat16);

    let npu = AcceleratorProfile {
        accelerator: AcceleratorType::IntelNpu,
        available: true,
        memory_bytes: 0,
        compute_capability: None,
        driver_version: None,
    };
    assert_eq!(npu.preferred_quantization(), QuantizationLevel::Int8);
}

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
