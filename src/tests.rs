//! Comprehensive test suite for ai-hwaccel.

use crate::*;

// ---------------------------------------------------------------------------
// AcceleratorType classification
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// TpuVersion HBM
// ---------------------------------------------------------------------------

#[test]
fn tpu_hbm_per_chip() {
    assert_eq!(TpuVersion::V4.hbm_per_chip_bytes(), 32 * 1024 * 1024 * 1024);
    assert_eq!(TpuVersion::V5e.hbm_per_chip_bytes(), 16 * 1024 * 1024 * 1024);
    assert_eq!(TpuVersion::V5p.hbm_per_chip_bytes(), 95 * 1024 * 1024 * 1024);
}

// ---------------------------------------------------------------------------
// Gaudi HBM
// ---------------------------------------------------------------------------

#[test]
fn gaudi_hbm() {
    assert_eq!(GaudiGeneration::Gaudi2.hbm_bytes(), 96 * 1024 * 1024 * 1024);
    assert_eq!(GaudiGeneration::Gaudi3.hbm_bytes(), 128 * 1024 * 1024 * 1024);
}

// ---------------------------------------------------------------------------
// Throughput ordering
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Quantisation support
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

// ---------------------------------------------------------------------------
// AcceleratorRequirement
// ---------------------------------------------------------------------------

#[test]
fn requirement_display() {
    assert_eq!(AcceleratorRequirement::None.to_string(), "none");
    assert_eq!(AcceleratorRequirement::Gpu.to_string(), "gpu");
    assert_eq!(
        AcceleratorRequirement::Tpu { min_chips: 4 }.to_string(),
        "tpu(4+ chips)"
    );
    assert_eq!(AcceleratorRequirement::GpuOrTpu.to_string(), "gpu-or-tpu");
    assert_eq!(AcceleratorRequirement::Gaudi.to_string(), "gaudi");
}

#[test]
fn requirement_satisfied_by() {
    let cuda_profile = AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    };
    let tpu_profile = AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 4,
            version: TpuVersion::V5p,
        },
        available: true,
        memory_bytes: 95 * 4 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    };
    let cpu_profile = AcceleratorProfile {
        accelerator: AcceleratorType::Cpu,
        available: true,
        memory_bytes: 16 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    };

    // None satisfied by anything
    assert!(AcceleratorRequirement::None.satisfied_by(&cuda_profile));
    assert!(AcceleratorRequirement::None.satisfied_by(&cpu_profile));

    // GPU requirement
    assert!(AcceleratorRequirement::Gpu.satisfied_by(&cuda_profile));
    assert!(!AcceleratorRequirement::Gpu.satisfied_by(&tpu_profile));
    assert!(!AcceleratorRequirement::Gpu.satisfied_by(&cpu_profile));

    // TPU requirement
    assert!(
        AcceleratorRequirement::Tpu { min_chips: 2 }.satisfied_by(&tpu_profile)
    );
    assert!(
        !AcceleratorRequirement::Tpu { min_chips: 8 }.satisfied_by(&tpu_profile)
    );
    assert!(
        !AcceleratorRequirement::Tpu { min_chips: 1 }.satisfied_by(&cuda_profile)
    );

    // GpuOrTpu
    assert!(AcceleratorRequirement::GpuOrTpu.satisfied_by(&cuda_profile));
    assert!(AcceleratorRequirement::GpuOrTpu.satisfied_by(&tpu_profile));
    assert!(!AcceleratorRequirement::GpuOrTpu.satisfied_by(&cpu_profile));

    // AnyAccelerator
    assert!(AcceleratorRequirement::AnyAccelerator.satisfied_by(&cuda_profile));
    assert!(AcceleratorRequirement::AnyAccelerator.satisfied_by(&tpu_profile));
    assert!(!AcceleratorRequirement::AnyAccelerator.satisfied_by(&cpu_profile));
}

#[test]
fn requirement_unavailable_device_never_satisfies() {
    let unavailable = AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: false,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    };
    assert!(!AcceleratorRequirement::Gpu.satisfied_by(&unavailable));
    assert!(!AcceleratorRequirement::AnyAccelerator.satisfied_by(&unavailable));
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

#[test]
fn registry_new_has_cpu() {
    let reg = AcceleratorRegistry::new();
    assert_eq!(reg.all_profiles().len(), 1);
    assert_eq!(reg.all_profiles()[0].accelerator, AcceleratorType::Cpu);
}

#[test]
fn registry_best_available_cpu_only() {
    let reg = AcceleratorRegistry::new();
    let best = reg.best_available().unwrap();
    assert_eq!(best.accelerator, AcceleratorType::Cpu);
}

#[test]
fn registry_best_prefers_tpu_v5p_over_cuda() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 4,
            version: TpuVersion::V5p,
        },
        available: true,
        memory_bytes: 380 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    assert!(matches!(
        reg.best_available().unwrap().accelerator,
        AcceleratorType::Tpu { .. }
    ));
}

#[test]
fn registry_best_prefers_cuda_over_tpu_v5e() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 1,
            version: TpuVersion::V5e,
        },
        available: true,
        memory_bytes: 16 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    assert!(matches!(
        reg.best_available().unwrap().accelerator,
        AcceleratorType::CudaGpu { .. }
    ));
}

#[test]
fn registry_total_accelerator_memory() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 8 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 1,
            version: TpuVersion::V4,
        },
        available: true,
        memory_bytes: 32 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    // GPU 8 + TPU 32 = 40 GiB (CPU excluded)
    assert_eq!(reg.total_accelerator_memory(), 40 * 1024 * 1024 * 1024);
}

#[test]
fn registry_by_family() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 8 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::IntelNpu,
        available: true,
        memory_bytes: 2 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    assert_eq!(reg.by_family(AcceleratorFamily::Gpu).len(), 1);
    assert_eq!(reg.by_family(AcceleratorFamily::Npu).len(), 1);
    assert_eq!(reg.by_family(AcceleratorFamily::Tpu).len(), 0);
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

#[test]
fn estimate_memory_fp32() {
    let est = AcceleratorRegistry::estimate_memory(1_000_000_000, &QuantizationLevel::None);
    assert_eq!(est, 4_800_000_000); // 4 bytes * 1B + 20%
}

#[test]
fn estimate_memory_int4() {
    let est = AcceleratorRegistry::estimate_memory(1_000_000_000, &QuantizationLevel::Int4);
    assert_eq!(est, 600_000_000); // 0.5 bytes * 1B + 20%
}

// ---------------------------------------------------------------------------
// suggest_quantization
// ---------------------------------------------------------------------------

#[test]
fn suggest_quantization_tpu_prefers_bf16() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 4,
            version: TpuVersion::V5p,
        },
        available: true,
        memory_bytes: 380 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    assert_eq!(
        reg.suggest_quantization(7_000_000_000),
        QuantizationLevel::BFloat16
    );
}

#[test]
fn suggest_quantization_gpu_prefers_fp16() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    assert_eq!(
        reg.suggest_quantization(7_000_000_000),
        QuantizationLevel::Float16
    );
}

// ---------------------------------------------------------------------------
// Sharding plans
// ---------------------------------------------------------------------------

#[test]
fn plan_sharding_small_model_single_device() {
    let reg = AcceleratorRegistry::new();
    let plan = reg.plan_sharding(1_000_000_000, &QuantizationLevel::Int4);
    assert_eq!(plan.strategy, ShardingStrategy::None);
    assert_eq!(plan.shards.len(), 1);
}

#[test]
fn plan_sharding_tpu_tensor_parallel() {
    let mut reg = AcceleratorRegistry::new();
    // 4 separate TPU chips, each with 95 GB — model too large for any single chip
    for i in 0..4 {
        reg.add_profile(AcceleratorProfile {
            accelerator: AcceleratorType::Tpu {
                device_id: i,
                chip_count: 1,
                version: TpuVersion::V5p,
            },
            available: true,
            memory_bytes: 95 * 1024 * 1024 * 1024,
            compute_capability: None,
            driver_version: None,
        });
    }
    // 70B at BF16 = ~168 GB — doesn't fit on one 95GB chip, but fits on 4
    let plan = reg.plan_sharding(70_000_000_000, &QuantizationLevel::BFloat16);
    assert!(matches!(
        plan.strategy,
        ShardingStrategy::TensorParallel { num_devices: 4 }
    ));
}

#[test]
fn plan_sharding_multi_gpu_pipeline() {
    let mut reg = AcceleratorRegistry::new();
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 8 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    reg.add_profile(AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 1 },
        available: true,
        memory_bytes: 8 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    });
    let plan = reg.plan_sharding(7_000_000_000, &QuantizationLevel::Float16);
    assert!(matches!(
        plan.strategy,
        ShardingStrategy::PipelineParallel { .. }
    ));
    assert_eq!(plan.shards.len(), 2);
}

#[test]
fn plan_sharding_cpu_fallback() {
    let reg = AcceleratorRegistry::new();
    let plan = reg.plan_sharding(70_000_000_000, &QuantizationLevel::None);
    assert_eq!(plan.shards[0].device, AcceleratorType::Cpu);
}

// ---------------------------------------------------------------------------
// Training memory estimation
// ---------------------------------------------------------------------------

#[test]
fn training_memory_tpu_less_optimizer_than_gpu() {
    let gpu = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Gpu);
    let tpu = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Tpu);
    assert!(tpu.optimizer_gb < gpu.optimizer_gb);
}

#[test]
fn training_memory_qlora_less_than_full() {
    let full = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Gpu);
    let qlora = estimate_training_memory(
        7000,
        TrainingMethod::QLoRA { bits: 4 },
        TrainingTarget::Gpu,
    );
    assert!(qlora.total_gb < full.total_gb);
}

#[test]
fn training_memory_gaudi() {
    let est = estimate_training_memory(7000, TrainingMethod::FullFineTune, TrainingTarget::Gaudi);
    assert!(est.total_gb > 30.0);
    assert!(est.total_gb < 60.0);
}

// ---------------------------------------------------------------------------
// TrainingMethod
// ---------------------------------------------------------------------------

#[test]
fn training_method_display() {
    assert_eq!(TrainingMethod::LoRA.to_string(), "lora");
    assert_eq!(TrainingMethod::QLoRA { bits: 4 }.to_string(), "qlora-4bit");
    assert_eq!(TrainingMethod::FullFineTune.to_string(), "full");
    assert_eq!(TrainingMethod::DPO.to_string(), "dpo");
    assert_eq!(TrainingMethod::RLHF.to_string(), "rlhf");
    assert_eq!(TrainingMethod::Distillation.to_string(), "distillation");
}

#[test]
fn training_method_preferred_accelerator() {
    assert_eq!(
        TrainingMethod::LoRA.preferred_accelerator(),
        AcceleratorRequirement::Gpu
    );
    assert_eq!(
        TrainingMethod::QLoRA { bits: 4 }.preferred_accelerator(),
        AcceleratorRequirement::Gpu
    );
    assert_eq!(
        TrainingMethod::FullFineTune.preferred_accelerator(),
        AcceleratorRequirement::GpuOrTpu
    );
    assert_eq!(
        TrainingMethod::DPO.preferred_accelerator(),
        AcceleratorRequirement::GpuOrTpu
    );
}

// ---------------------------------------------------------------------------
// Serde roundtrip
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// ModelShard
// ---------------------------------------------------------------------------

#[test]
fn model_shard_num_layers() {
    let shard = ModelShard {
        shard_id: 0,
        layer_range: (0, 31),
        device: AcceleratorType::Cpu,
        memory_bytes: 1024,
    };
    assert_eq!(shard.num_layers(), 32);
    assert!(shard.is_valid());
}

#[test]
fn model_shard_invalid_range() {
    let shard = ModelShard {
        shard_id: 0,
        layer_range: (10, 5),
        device: AcceleratorType::Cpu,
        memory_bytes: 0,
    };
    assert_eq!(shard.num_layers(), 0);
    assert!(!shard.is_valid());
}

// ---------------------------------------------------------------------------
// detect() smoke test
// ---------------------------------------------------------------------------

#[test]
fn detect_returns_at_least_cpu() {
    let reg = AcceleratorRegistry::detect();
    assert!(!reg.all_profiles().is_empty());
    assert!(reg
        .all_profiles()
        .iter()
        .any(|p| matches!(p.accelerator, AcceleratorType::Cpu)));
}
