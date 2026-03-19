//! AcceleratorType classification and family tests.

use crate::*;

#[test]
fn type_classification_gpu() {
    assert!(AcceleratorType::CudaGpu { device_id: 0 }.is_gpu());
    assert!(AcceleratorType::RocmGpu { device_id: 0 }.is_gpu());
    assert!(AcceleratorType::MetalGpu.is_gpu());
    assert!(
        AcceleratorType::VulkanGpu {
            device_id: 0,
            device_name: "test".into()
        }
        .is_gpu()
    );
    assert!(AcceleratorType::IntelOneApi { device_id: 0 }.is_gpu());
    assert!(!AcceleratorType::Cpu.is_gpu());
    assert!(!AcceleratorType::IntelNpu.is_gpu());
    assert!(
        !AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 1,
            version: TpuVersion::V5e
        }
        .is_gpu()
    );
}

#[test]
fn type_classification_npu() {
    assert!(AcceleratorType::IntelNpu.is_npu());
    assert!(AcceleratorType::AppleNpu.is_npu());
    assert!(AcceleratorType::AmdXdnaNpu { device_id: 0 }.is_npu());
    assert!(!AcceleratorType::CudaGpu { device_id: 0 }.is_npu());
}

#[test]
fn type_classification_tpu() {
    assert!(
        AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 4,
            version: TpuVersion::V5p
        }
        .is_tpu()
    );
    assert!(!AcceleratorType::CudaGpu { device_id: 0 }.is_tpu());
    assert!(!AcceleratorType::Cpu.is_tpu());
}

#[test]
fn type_classification_ai_asic() {
    assert!(
        AcceleratorType::Gaudi {
            device_id: 0,
            generation: GaudiGeneration::Gaudi2
        }
        .is_ai_asic()
    );
    assert!(
        AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Trainium,
            core_count: 2
        }
        .is_ai_asic()
    );
    assert!(AcceleratorType::QualcommAi100 { device_id: 0 }.is_ai_asic());
    assert!(!AcceleratorType::CudaGpu { device_id: 0 }.is_ai_asic());
}

#[test]
fn type_family() {
    assert_eq!(AcceleratorType::Cpu.family(), AcceleratorFamily::Cpu);
    assert_eq!(
        AcceleratorType::CudaGpu { device_id: 0 }.family(),
        AcceleratorFamily::Gpu
    );
    assert_eq!(AcceleratorType::IntelNpu.family(), AcceleratorFamily::Npu);
    assert_eq!(
        AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 1,
            version: TpuVersion::V4
        }
        .family(),
        AcceleratorFamily::Tpu
    );
    assert_eq!(
        AcceleratorType::Gaudi {
            device_id: 0,
            generation: GaudiGeneration::Gaudi3
        }
        .family(),
        AcceleratorFamily::AiAsic
    );
    assert_eq!(
        AcceleratorType::IntelOneApi { device_id: 0 }.family(),
        AcceleratorFamily::Gpu
    );
}

#[test]
fn throughput_ordering() {
    let cpu = AcceleratorType::Cpu.throughput_multiplier();
    let npu = AcceleratorType::IntelNpu.throughput_multiplier();
    let cuda = AcceleratorType::CudaGpu { device_id: 0 }.throughput_multiplier();
    let tpu_v5p = AcceleratorType::Tpu {
        device_id: 0,
        chip_count: 1,
        version: TpuVersion::V5p,
    }
    .throughput_multiplier();

    assert!(cpu < npu);
    assert!(npu < cuda);
    assert!(cuda < tpu_v5p);
}

#[test]
fn training_multiplier_inference_only_is_zero() {
    assert_eq!(AcceleratorType::IntelNpu.training_multiplier(), 0.0);
    assert_eq!(AcceleratorType::AppleNpu.training_multiplier(), 0.0);
    assert_eq!(
        AcceleratorType::QualcommAi100 { device_id: 0 }.training_multiplier(),
        0.0
    );
    assert_eq!(
        AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Inferentia,
            core_count: 2
        }
        .training_multiplier(),
        0.0
    );
}

#[test]
fn training_multiplier_positive_for_trainable_devices() {
    assert!(AcceleratorType::CudaGpu { device_id: 0 }.supports_training());
    assert!(AcceleratorType::RocmGpu { device_id: 0 }.supports_training());
    assert!(
        AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 1,
            version: TpuVersion::V5p
        }
        .supports_training()
    );
    assert!(
        AcceleratorType::Gaudi {
            device_id: 0,
            generation: GaudiGeneration::Gaudi3
        }
        .supports_training()
    );
    assert!(
        AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Trainium,
            core_count: 2
        }
        .supports_training()
    );
}
