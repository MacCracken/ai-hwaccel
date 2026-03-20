//! AcceleratorRequirement tests.

use crate::*;

#[test]
fn requirement_satisfied_by_all_types() {
    let cuda = AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024);
    let tpu = AcceleratorProfile::tpu(0, 4, TpuVersion::V5p);
    let cpu = AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024);
    let gaudi = AcceleratorProfile::gaudi(0, GaudiGeneration::Gaudi3);
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
    let npu = AcceleratorProfile {
        accelerator: AcceleratorType::IntelNpu,
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

    // None satisfied by anything
    assert!(AcceleratorRequirement::None.satisfied_by(&cuda));
    assert!(AcceleratorRequirement::None.satisfied_by(&cpu));
    assert!(AcceleratorRequirement::None.satisfied_by(&gaudi));
    assert!(AcceleratorRequirement::None.satisfied_by(&neuron));

    // GPU requirement
    assert!(AcceleratorRequirement::Gpu.satisfied_by(&cuda));
    assert!(!AcceleratorRequirement::Gpu.satisfied_by(&tpu));
    assert!(!AcceleratorRequirement::Gpu.satisfied_by(&cpu));
    assert!(!AcceleratorRequirement::Gpu.satisfied_by(&gaudi));
    assert!(!AcceleratorRequirement::Gpu.satisfied_by(&npu));

    // TPU requirement
    assert!(AcceleratorRequirement::Tpu { min_chips: 2 }.satisfied_by(&tpu));
    assert!(!AcceleratorRequirement::Tpu { min_chips: 8 }.satisfied_by(&tpu));
    assert!(!AcceleratorRequirement::Tpu { min_chips: 1 }.satisfied_by(&cuda));

    // Gaudi requirement
    assert!(AcceleratorRequirement::Gaudi.satisfied_by(&gaudi));
    assert!(!AcceleratorRequirement::Gaudi.satisfied_by(&cuda));
    assert!(!AcceleratorRequirement::Gaudi.satisfied_by(&tpu));
    assert!(!AcceleratorRequirement::Gaudi.satisfied_by(&cpu));

    // AwsNeuron requirement
    assert!(AcceleratorRequirement::AwsNeuron.satisfied_by(&neuron));
    assert!(!AcceleratorRequirement::AwsNeuron.satisfied_by(&cuda));
    assert!(!AcceleratorRequirement::AwsNeuron.satisfied_by(&gaudi));

    // GpuOrTpu
    assert!(AcceleratorRequirement::GpuOrTpu.satisfied_by(&cuda));
    assert!(AcceleratorRequirement::GpuOrTpu.satisfied_by(&tpu));
    assert!(!AcceleratorRequirement::GpuOrTpu.satisfied_by(&cpu));
    assert!(!AcceleratorRequirement::GpuOrTpu.satisfied_by(&gaudi));

    // AnyAccelerator
    assert!(AcceleratorRequirement::AnyAccelerator.satisfied_by(&cuda));
    assert!(AcceleratorRequirement::AnyAccelerator.satisfied_by(&tpu));
    assert!(AcceleratorRequirement::AnyAccelerator.satisfied_by(&gaudi));
    assert!(AcceleratorRequirement::AnyAccelerator.satisfied_by(&neuron));
    assert!(AcceleratorRequirement::AnyAccelerator.satisfied_by(&npu));
    assert!(!AcceleratorRequirement::AnyAccelerator.satisfied_by(&cpu));
}

#[test]
fn requirement_unavailable_device_never_satisfies() {
    let mut unavailable = AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024);
    unavailable.available = false;
    assert!(!AcceleratorRequirement::Gpu.satisfied_by(&unavailable));
    assert!(!AcceleratorRequirement::AnyAccelerator.satisfied_by(&unavailable));
    assert!(!AcceleratorRequirement::GpuOrTpu.satisfied_by(&unavailable));
    // None still requires available
    assert!(!AcceleratorRequirement::None.satisfied_by(&unavailable));
}

#[test]
fn requirement_default_is_none() {
    let req: AcceleratorRequirement = Default::default();
    assert_eq!(req, AcceleratorRequirement::None);
}
