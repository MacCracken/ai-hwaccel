//! Display formatting tests.

use crate::*;

#[test]
fn display_accelerator_types() {
    assert_eq!(AcceleratorType::Cpu.to_string(), "CPU");
    assert_eq!(
        AcceleratorType::CudaGpu { device_id: 3 }.to_string(),
        "CUDA GPU (device 3)"
    );
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
        AcceleratorType::IntelOneApi { device_id: 0 }.to_string(),
        "Intel oneAPI GPU (device 0)"
    );
}

#[test]
fn display_tpu_version() {
    assert_eq!(TpuVersion::V4.to_string(), "v4");
    assert_eq!(TpuVersion::V5e.to_string(), "v5e");
    assert_eq!(TpuVersion::V5p.to_string(), "v5p");
}

#[test]
fn display_quantization() {
    assert_eq!(QuantizationLevel::None.to_string(), "FP32");
    assert_eq!(QuantizationLevel::Float16.to_string(), "FP16");
    assert_eq!(QuantizationLevel::BFloat16.to_string(), "BF16");
    assert_eq!(QuantizationLevel::Int8.to_string(), "INT8");
    assert_eq!(QuantizationLevel::Int4.to_string(), "INT4");
}

#[test]
fn display_requirement() {
    assert_eq!(AcceleratorRequirement::None.to_string(), "none");
    assert_eq!(AcceleratorRequirement::Gpu.to_string(), "gpu");
    assert_eq!(
        AcceleratorRequirement::Tpu { min_chips: 4 }.to_string(),
        "tpu(4+ chips)"
    );
    assert_eq!(AcceleratorRequirement::GpuOrTpu.to_string(), "gpu-or-tpu");
    assert_eq!(AcceleratorRequirement::Gaudi.to_string(), "gaudi");
}
