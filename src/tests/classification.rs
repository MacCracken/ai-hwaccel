//! AcceleratorType classification, family, throughput, and training tests.

use crate::*;

// ---------------------------------------------------------------------------
// is_gpu / is_npu / is_tpu / is_ai_asic — exhaustive
// ---------------------------------------------------------------------------

#[test]
fn type_classification_gpu() {
    let gpus = [
        AcceleratorType::CudaGpu { device_id: 0 },
        AcceleratorType::RocmGpu { device_id: 0 },
        AcceleratorType::MetalGpu,
        AcceleratorType::VulkanGpu {
            device_id: 0,
            device_name: "test".into(),
        },
        AcceleratorType::IntelOneApi { device_id: 0 },
    ];
    for g in &gpus {
        assert!(g.is_gpu(), "{:?} should be GPU", g);
        assert!(!g.is_npu(), "{:?} should not be NPU", g);
        assert!(!g.is_tpu(), "{:?} should not be TPU", g);
        assert!(!g.is_ai_asic(), "{:?} should not be AI ASIC", g);
    }
}

#[test]
fn type_classification_npu() {
    let npus = [
        AcceleratorType::IntelNpu,
        AcceleratorType::AppleNpu,
        AcceleratorType::AmdXdnaNpu { device_id: 0 },
    ];
    for n in &npus {
        assert!(n.is_npu(), "{:?} should be NPU", n);
        assert!(!n.is_gpu(), "{:?} should not be GPU", n);
        assert!(!n.is_tpu(), "{:?} should not be TPU", n);
        assert!(!n.is_ai_asic(), "{:?} should not be AI ASIC", n);
    }
}

#[test]
fn type_classification_tpu() {
    let tpu = AcceleratorType::Tpu {
        device_id: 0,
        chip_count: 4,
        version: TpuVersion::V5p,
    };
    assert!(tpu.is_tpu());
    assert!(!tpu.is_gpu());
    assert!(!tpu.is_npu());
    assert!(!tpu.is_ai_asic());
}

#[test]
fn type_classification_ai_asic() {
    let asics: Vec<AcceleratorType> = vec![
        AcceleratorType::Gaudi {
            device_id: 0,
            generation: GaudiGeneration::Gaudi2,
        },
        AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Trainium,
            core_count: 2,
        },
        AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Inferentia,
            core_count: 2,
        },
        AcceleratorType::QualcommAi100 { device_id: 0 },
    ];
    for a in &asics {
        assert!(a.is_ai_asic(), "{:?} should be AI ASIC", a);
        assert!(!a.is_gpu(), "{:?} should not be GPU", a);
        assert!(!a.is_tpu(), "{:?} should not be TPU", a);
        assert!(!a.is_npu(), "{:?} should not be NPU", a);
    }
}

#[test]
fn cpu_is_nothing() {
    let cpu = AcceleratorType::Cpu;
    assert!(!cpu.is_gpu());
    assert!(!cpu.is_npu());
    assert!(!cpu.is_tpu());
    assert!(!cpu.is_ai_asic());
}

// ---------------------------------------------------------------------------
// family()
// ---------------------------------------------------------------------------

#[test]
fn type_family_all_variants() {
    assert_eq!(AcceleratorType::Cpu.family(), AcceleratorFamily::Cpu);
    assert_eq!(
        AcceleratorType::CudaGpu { device_id: 0 }.family(),
        AcceleratorFamily::Gpu
    );
    assert_eq!(
        AcceleratorType::RocmGpu { device_id: 0 }.family(),
        AcceleratorFamily::Gpu
    );
    assert_eq!(AcceleratorType::MetalGpu.family(), AcceleratorFamily::Gpu);
    assert_eq!(
        AcceleratorType::VulkanGpu {
            device_id: 0,
            device_name: "x".into()
        }
        .family(),
        AcceleratorFamily::Gpu
    );
    assert_eq!(
        AcceleratorType::IntelOneApi { device_id: 0 }.family(),
        AcceleratorFamily::Gpu
    );
    assert_eq!(AcceleratorType::IntelNpu.family(), AcceleratorFamily::Npu);
    assert_eq!(AcceleratorType::AppleNpu.family(), AcceleratorFamily::Npu);
    assert_eq!(
        AcceleratorType::AmdXdnaNpu { device_id: 0 }.family(),
        AcceleratorFamily::Npu
    );
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
        AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Trainium,
            core_count: 2
        }
        .family(),
        AcceleratorFamily::AiAsic
    );
    assert_eq!(
        AcceleratorType::QualcommAi100 { device_id: 0 }.family(),
        AcceleratorFamily::AiAsic
    );
}

// ---------------------------------------------------------------------------
// throughput_multiplier — ordering and positivity
// ---------------------------------------------------------------------------

#[test]
fn throughput_ordering() {
    let cpu = AcceleratorType::Cpu.throughput_multiplier();
    let npu = AcceleratorType::IntelNpu.throughput_multiplier();
    let vulkan = AcceleratorType::VulkanGpu {
        device_id: 0,
        device_name: "x".into(),
    }
    .throughput_multiplier();
    let cuda = AcceleratorType::CudaGpu { device_id: 0 }.throughput_multiplier();
    let tpu_v5p = AcceleratorType::Tpu {
        device_id: 0,
        chip_count: 1,
        version: TpuVersion::V5p,
    }
    .throughput_multiplier();

    assert!(cpu < npu);
    assert!(npu < vulkan);
    assert!(vulkan < cuda);
    assert!(cuda < tpu_v5p);
}

#[test]
fn throughput_all_positive() {
    let types: Vec<AcceleratorType> = vec![
        AcceleratorType::Cpu,
        AcceleratorType::CudaGpu { device_id: 0 },
        AcceleratorType::RocmGpu { device_id: 0 },
        AcceleratorType::MetalGpu,
        AcceleratorType::VulkanGpu {
            device_id: 0,
            device_name: "x".into(),
        },
        AcceleratorType::IntelNpu,
        AcceleratorType::AppleNpu,
        AcceleratorType::AmdXdnaNpu { device_id: 0 },
        AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 1,
            version: TpuVersion::V4,
        },
        AcceleratorType::Gaudi {
            device_id: 0,
            generation: GaudiGeneration::Gaudi2,
        },
        AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Inferentia,
            core_count: 2,
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
        assert!(
            t.throughput_multiplier() > 0.0,
            "{:?} should have positive throughput",
            t
        );
    }
}

// ---------------------------------------------------------------------------
// training_multiplier
// ---------------------------------------------------------------------------

#[test]
fn training_multiplier_inference_only_is_zero() {
    let inference_only = [
        AcceleratorType::IntelNpu,
        AcceleratorType::AppleNpu,
        AcceleratorType::AmdXdnaNpu { device_id: 0 },
        AcceleratorType::QualcommAi100 { device_id: 0 },
        AcceleratorType::AwsNeuron {
            device_id: 0,
            chip_type: NeuronChipType::Inferentia,
            core_count: 2,
        },
    ];
    for t in &inference_only {
        assert_eq!(
            t.training_multiplier(),
            0.0,
            "{:?} should have zero training multiplier",
            t
        );
        assert!(!t.supports_training());
    }
}

#[test]
fn training_multiplier_positive_for_trainable_devices() {
    let trainable = [
        AcceleratorType::Cpu,
        AcceleratorType::CudaGpu { device_id: 0 },
        AcceleratorType::RocmGpu { device_id: 0 },
        AcceleratorType::MetalGpu,
        AcceleratorType::VulkanGpu {
            device_id: 0,
            device_name: "x".into(),
        },
        AcceleratorType::IntelOneApi { device_id: 0 },
        AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 1,
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
    ];
    for t in &trainable {
        assert!(
            t.training_multiplier() > 0.0,
            "{:?} should have positive training multiplier",
            t
        );
        assert!(t.supports_training());
    }
}

// ---------------------------------------------------------------------------
// Neuron throughput scales with core count
// ---------------------------------------------------------------------------

#[test]
fn neuron_throughput_scales_with_cores() {
    let two_cores = AcceleratorType::AwsNeuron {
        device_id: 0,
        chip_type: NeuronChipType::Trainium,
        core_count: 2,
    };
    let four_cores = AcceleratorType::AwsNeuron {
        device_id: 0,
        chip_type: NeuronChipType::Trainium,
        core_count: 4,
    };
    assert!(four_cores.throughput_multiplier() > two_cores.throughput_multiplier());
}

// ---------------------------------------------------------------------------
// Rank ordering (used by best_available)
// ---------------------------------------------------------------------------

#[test]
fn rank_tpu_v5p_highest() {
    let tpu_v5p = AcceleratorType::Tpu {
        device_id: 0,
        chip_count: 1,
        version: TpuVersion::V5p,
    };
    let cuda = AcceleratorType::CudaGpu { device_id: 0 };
    let cpu = AcceleratorType::Cpu;
    assert!(tpu_v5p.rank() > cuda.rank());
    assert!(cuda.rank() > cpu.rank());
}

#[test]
fn rank_cpu_is_lowest() {
    let types: Vec<AcceleratorType> = vec![
        AcceleratorType::CudaGpu { device_id: 0 },
        AcceleratorType::RocmGpu { device_id: 0 },
        AcceleratorType::MetalGpu,
        AcceleratorType::IntelNpu,
        AcceleratorType::AppleNpu,
    ];
    let cpu_rank = AcceleratorType::Cpu.rank();
    for t in &types {
        assert!(t.rank() > cpu_rank, "{:?} should rank above CPU", t);
    }
}
