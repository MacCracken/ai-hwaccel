//! Serde roundtrip tests for all serializable types.

use crate::*;

// ---------------------------------------------------------------------------
// AcceleratorType — all variants including VulkanGpu and AmdXdnaNpu
// ---------------------------------------------------------------------------

#[test]
fn serde_accelerator_type_roundtrip() {
    let types = vec![
        AcceleratorType::Cpu,
        AcceleratorType::CudaGpu { device_id: 0 },
        AcceleratorType::RocmGpu { device_id: 1 },
        AcceleratorType::MetalGpu,
        AcceleratorType::VulkanGpu {
            device_id: 0,
            device_name: "Test GPU".into(),
        },
        AcceleratorType::IntelNpu,
        AcceleratorType::AmdXdnaNpu { device_id: 0 },
        AcceleratorType::AppleNpu,
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
        AcceleratorType::AwsNeuron {
            device_id: 1,
            chip_type: NeuronChipType::Inferentia,
            core_count: 4,
        },
        AcceleratorType::QualcommAi100 { device_id: 0 },
        AcceleratorType::IntelOneApi { device_id: 0 },
    ];
    for t in &types {
        let json = serde_json::to_string(t).unwrap();
        let back: AcceleratorType = serde_json::from_str(&json).unwrap();
        assert_eq!(*t, back, "roundtrip failed for {:?}", t);
    }
}

// ---------------------------------------------------------------------------
// AcceleratorFamily
// ---------------------------------------------------------------------------

#[test]
fn serde_accelerator_family_roundtrip() {
    for family in [
        AcceleratorFamily::Cpu,
        AcceleratorFamily::Gpu,
        AcceleratorFamily::Npu,
        AcceleratorFamily::Tpu,
        AcceleratorFamily::AiAsic,
    ] {
        let json = serde_json::to_string(&family).unwrap();
        let back: AcceleratorFamily = serde_json::from_str(&json).unwrap();
        assert_eq!(family, back);
    }
}

// ---------------------------------------------------------------------------
// QuantizationLevel
// ---------------------------------------------------------------------------

#[test]
fn serde_quantization_level_roundtrip() {
    for q in [
        QuantizationLevel::None,
        QuantizationLevel::Float16,
        QuantizationLevel::BFloat16,
        QuantizationLevel::Int8,
        QuantizationLevel::Int4,
    ] {
        let json = serde_json::to_string(&q).unwrap();
        let back: QuantizationLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(q, back);
    }
}

// ---------------------------------------------------------------------------
// AcceleratorProfile
// ---------------------------------------------------------------------------

#[test]
fn serde_accelerator_profile_roundtrip() {
    let profiles = vec![
        AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024),
        AcceleratorProfile::tpu(0, 4, TpuVersion::V5p),
        AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024),
        AcceleratorProfile {
            accelerator: AcceleratorType::CudaGpu { device_id: 0 },
            available: true,
            memory_bytes: 24 * 1024 * 1024 * 1024,
            compute_capability: Some("8.9".into()),
            driver_version: Some("545.29.06".into()),
            memory_bandwidth_gbps: None,
            memory_used_bytes: None,
            memory_free_bytes: None,
            pcie_bandwidth_gbps: None,
            numa_node: None,
        },
    ];
    for p in &profiles {
        let json = serde_json::to_string(p).unwrap();
        let back: AcceleratorProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(*p, back);
    }
}

// ---------------------------------------------------------------------------
// AcceleratorRequirement
// ---------------------------------------------------------------------------

#[test]
fn serde_requirement_roundtrip() {
    let reqs = vec![
        AcceleratorRequirement::None,
        AcceleratorRequirement::Gpu,
        AcceleratorRequirement::Tpu { min_chips: 4 },
        AcceleratorRequirement::Gaudi,
        AcceleratorRequirement::AwsNeuron,
        AcceleratorRequirement::GpuOrTpu,
        AcceleratorRequirement::AnyAccelerator,
    ];
    for r in &reqs {
        let json = serde_json::to_string(r).unwrap();
        let back: AcceleratorRequirement = serde_json::from_str(&json).unwrap();
        assert_eq!(*r, back);
    }
}

// ---------------------------------------------------------------------------
// ShardingStrategy
// ---------------------------------------------------------------------------

#[test]
fn serde_sharding_strategy_roundtrip() {
    let strats = vec![
        ShardingStrategy::None,
        ShardingStrategy::PipelineParallel { num_stages: 4 },
        ShardingStrategy::TensorParallel { num_devices: 8 },
        ShardingStrategy::DataParallel { num_replicas: 2 },
    ];
    for s in &strats {
        let json = serde_json::to_string(s).unwrap();
        let back: ShardingStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(*s, back);
    }
}

// ---------------------------------------------------------------------------
// ModelShard
// ---------------------------------------------------------------------------

#[test]
fn serde_model_shard_roundtrip() {
    let shard = ModelShard {
        shard_id: 0,
        layer_range: (0, 31),
        device: AcceleratorType::CudaGpu { device_id: 0 },
        memory_bytes: 8 * 1024 * 1024 * 1024,
    };
    let json = serde_json::to_string(&shard).unwrap();
    let back: ModelShard = serde_json::from_str(&json).unwrap();
    assert_eq!(shard, back);
}

// ---------------------------------------------------------------------------
// ShardingPlan
// ---------------------------------------------------------------------------

#[test]
fn serde_sharding_plan_roundtrip() {
    let plan = ShardingPlan {
        shards: vec![ModelShard {
            shard_id: 0,
            layer_range: (0, 0),
            device: AcceleratorType::Cpu,
            memory_bytes: 1024,
        }],
        strategy: ShardingStrategy::None,
        total_memory_bytes: 1024,
        estimated_tokens_per_sec: Some(42.5),
    };
    let json = serde_json::to_string(&plan).unwrap();
    let back: ShardingPlan = serde_json::from_str(&json).unwrap();
    assert_eq!(plan, back);
}

// ---------------------------------------------------------------------------
// TrainingMethod
// ---------------------------------------------------------------------------

#[test]
fn serde_training_method_roundtrip() {
    let methods = vec![
        TrainingMethod::FullFineTune,
        TrainingMethod::LoRA,
        TrainingMethod::QLoRA { bits: 4 },
        TrainingMethod::QLoRA { bits: 8 },
        TrainingMethod::Prefix,
        TrainingMethod::DPO,
        TrainingMethod::RLHF,
        TrainingMethod::Distillation,
    ];
    for m in &methods {
        let json = serde_json::to_string(m).unwrap();
        let back: TrainingMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(*m, back);
    }
}

// ---------------------------------------------------------------------------
// DetectionError
// ---------------------------------------------------------------------------

#[test]
fn serde_detection_error_roundtrip() {
    let errors = vec![
        DetectionError::ToolNotFound {
            tool: "nvidia-smi".into(),
        },
        DetectionError::ToolFailed {
            tool: "hl-smi".into(),
            exit_code: Some(1),
            stderr: "not found".into(),
        },
        DetectionError::ToolFailed {
            tool: "x".into(),
            exit_code: None,
            stderr: String::new(),
        },
        DetectionError::ParseError {
            backend: "cuda".into(),
            message: "bad csv".into(),
        },
        DetectionError::SysfsReadError {
            path: "/sys/foo".into(),
            message: "ENOENT".into(),
        },
    ];
    for e in &errors {
        let json = serde_json::to_string(e).unwrap();
        let back: DetectionError = serde_json::from_str(&json).unwrap();
        assert_eq!(*e, back);
    }
}

// ---------------------------------------------------------------------------
// MemoryEstimate
// ---------------------------------------------------------------------------

#[test]
fn serde_memory_estimate_roundtrip() {
    let est = MemoryEstimate {
        model_gb: 13.0,
        optimizer_gb: 26.0,
        activation_gb: 13.0,
        total_gb: 52.0,
    };
    let json = serde_json::to_string(&est).unwrap();
    let back: MemoryEstimate = serde_json::from_str(&json).unwrap();
    assert!((est.total_gb - back.total_gb).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// Hardware sub-types
// ---------------------------------------------------------------------------

#[test]
fn serde_tpu_version_roundtrip() {
    for v in [TpuVersion::V4, TpuVersion::V5e, TpuVersion::V5p] {
        let json = serde_json::to_string(&v).unwrap();
        let back: TpuVersion = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }
}

#[test]
fn serde_gaudi_generation_roundtrip() {
    for g in [GaudiGeneration::Gaudi2, GaudiGeneration::Gaudi3] {
        let json = serde_json::to_string(&g).unwrap();
        let back: GaudiGeneration = serde_json::from_str(&json).unwrap();
        assert_eq!(g, back);
    }
}

#[test]
fn serde_neuron_chip_type_roundtrip() {
    for c in [NeuronChipType::Inferentia, NeuronChipType::Trainium] {
        let json = serde_json::to_string(&c).unwrap();
        let back: NeuronChipType = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}

// ---------------------------------------------------------------------------
// Registry with warnings
// ---------------------------------------------------------------------------

#[test]
fn serde_registry_roundtrip() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: Some("8.6".into()),
        driver_version: None,
            memory_bandwidth_gbps: None,
            memory_used_bytes: None,
            memory_free_bytes: None,
            pcie_bandwidth_gbps: None,
            numa_node: None,
    });
    let json = serde_json::to_string(&reg).unwrap();
    let back: AcceleratorRegistry = serde_json::from_str(&json).unwrap();
    assert_eq!(back.all_profiles().len(), 2);
}

#[test]
fn serde_registry_warnings_omitted_when_empty() {
    let reg =
        AcceleratorRegistry::from_profiles(vec![AcceleratorProfile::cpu(16 * 1024 * 1024 * 1024)]);
    let json = serde_json::to_string(&reg).unwrap();
    assert!(!json.contains("warnings"));
}
