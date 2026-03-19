//! Serde roundtrip tests.

use crate::*;

#[test]
fn serde_accelerator_type_roundtrip() {
    let types = vec![
        AcceleratorType::Cpu,
        AcceleratorType::CudaGpu { device_id: 0 },
        AcceleratorType::RocmGpu { device_id: 1 },
        AcceleratorType::MetalGpu,
        AcceleratorType::IntelNpu,
        AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 4,
            version: TpuVersion::V5p,
        },
        AcceleratorType::Gaudi {
            device_id: 0,
            generation: GaudiGeneration::Gaudi3,
        },
        AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Trainium,
            core_count: 2,
        },
        AcceleratorType::QualcommAi100 { device_id: 0 },
        AcceleratorType::IntelOneApi { device_id: 0 },
    ];
    for t in &types {
        let json = serde_json::to_string(t).unwrap();
        let back: AcceleratorType = serde_json::from_str(&json).unwrap();
        assert_eq!(*t, back);
    }
}

#[test]
fn serde_registry_roundtrip() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: Some("8.6".into()),
        driver_version: None,
    });
    let json = serde_json::to_string(&reg).unwrap();
    let back: AcceleratorRegistry = serde_json::from_str(&json).unwrap();
    assert_eq!(back.all_profiles().len(), 2);
}
