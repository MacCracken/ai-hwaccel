//! Tests for new backend types added in 0.20: Cerebras, Graphcore, Groq,
//! Samsung NPU, MediaTek APU.

use crate::*;

// ---------------------------------------------------------------------------
// Classification: family, is_*, throughput, training, rank
// ---------------------------------------------------------------------------

#[test]
fn new_ai_asics_classification() {
    let asics = [
        AcceleratorType::CerebrasWse { device_id: 0 },
        AcceleratorType::GraphcoreIpu { device_id: 0 },
        AcceleratorType::GroqLpu { device_id: 0 },
    ];
    for a in &asics {
        assert!(a.is_ai_asic(), "{:?} should be AI ASIC", a);
        assert!(!a.is_gpu());
        assert!(!a.is_npu());
        assert!(!a.is_tpu());
        assert_eq!(a.family(), AcceleratorFamily::AiAsic);
    }
}

#[test]
fn new_npus_classification() {
    let npus = [
        AcceleratorType::SamsungNpu { device_id: 0 },
        AcceleratorType::MediaTekApu { device_id: 0 },
    ];
    for n in &npus {
        assert!(n.is_npu(), "{:?} should be NPU", n);
        assert!(!n.is_gpu());
        assert!(!n.is_tpu());
        assert!(!n.is_ai_asic());
        assert_eq!(n.family(), AcceleratorFamily::Npu);
    }
}

#[test]
fn new_backends_throughput_positive() {
    let types = [
        AcceleratorType::CerebrasWse { device_id: 0 },
        AcceleratorType::GraphcoreIpu { device_id: 0 },
        AcceleratorType::GroqLpu { device_id: 0 },
        AcceleratorType::SamsungNpu { device_id: 0 },
        AcceleratorType::MediaTekApu { device_id: 0 },
    ];
    for t in &types {
        assert!(
            t.throughput_multiplier() > 0.0,
            "{:?} should have positive throughput",
            t
        );
    }
}

#[test]
fn cerebras_supports_training() {
    let wse = AcceleratorType::CerebrasWse { device_id: 0 };
    assert!(wse.supports_training());
    assert!(wse.training_multiplier() > 0.0);
}

#[test]
fn graphcore_supports_training() {
    let ipu = AcceleratorType::GraphcoreIpu { device_id: 0 };
    assert!(ipu.supports_training());
    assert!(ipu.training_multiplier() > 0.0);
}

#[test]
fn groq_is_inference_only() {
    let lpu = AcceleratorType::GroqLpu { device_id: 0 };
    assert!(!lpu.supports_training());
    assert_eq!(lpu.training_multiplier(), 0.0);
}

#[test]
fn samsung_mediatek_inference_only() {
    let types = [
        AcceleratorType::SamsungNpu { device_id: 0 },
        AcceleratorType::MediaTekApu { device_id: 0 },
    ];
    for t in &types {
        assert!(!t.supports_training(), "{:?} should be inference-only", t);
    }
}

#[test]
fn cerebras_ranks_highest() {
    let wse = AcceleratorType::CerebrasWse { device_id: 0 };
    let cuda = AcceleratorType::CudaGpu { device_id: 0 };
    let tpu = AcceleratorType::Tpu {
        device_id: 0,
        chip_count: 1,
        version: TpuVersion::V5p,
    };
    assert!(wse.rank() > cuda.rank());
    assert!(wse.rank() > tpu.rank());
}

// ---------------------------------------------------------------------------
// Serde roundtrip for new types
// ---------------------------------------------------------------------------

#[test]
fn serde_new_accelerator_types_roundtrip() {
    let types = vec![
        AcceleratorType::CerebrasWse { device_id: 0 },
        AcceleratorType::GraphcoreIpu { device_id: 1 },
        AcceleratorType::GroqLpu { device_id: 2 },
        AcceleratorType::SamsungNpu { device_id: 0 },
        AcceleratorType::MediaTekApu { device_id: 0 },
    ];
    for t in &types {
        let json = serde_json::to_string(t).unwrap();
        let back: AcceleratorType = serde_json::from_str(&json).unwrap();
        assert_eq!(*t, back, "roundtrip failed for {:?}", t);
    }
}

// ---------------------------------------------------------------------------
// Display for new types
// ---------------------------------------------------------------------------

#[test]
fn display_new_backend_types() {
    assert!(
        AcceleratorType::CerebrasWse { device_id: 0 }
            .to_string()
            .contains("Cerebras")
    );
    assert!(
        AcceleratorType::GraphcoreIpu { device_id: 0 }
            .to_string()
            .contains("Graphcore")
    );
    assert!(
        AcceleratorType::GroqLpu { device_id: 0 }
            .to_string()
            .contains("Groq")
    );
    assert!(
        AcceleratorType::SamsungNpu { device_id: 0 }
            .to_string()
            .contains("Samsung")
    );
    assert!(
        AcceleratorType::MediaTekApu { device_id: 0 }
            .to_string()
            .contains("MediaTek")
    );
}

// ---------------------------------------------------------------------------
// Quantization support for new types
// ---------------------------------------------------------------------------

#[test]
fn cerebras_quantization_support() {
    let profile = AcceleratorProfile {
        accelerator: AcceleratorType::CerebrasWse { device_id: 0 },
        available: true,
        memory_bytes: 44 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
        device_name: None,
        memory_bandwidth_gbps: None,
        memory_used_bytes: None,
        memory_free_bytes: None,
        pcie_bandwidth_gbps: None,
        numa_node: None,
        temperature_c: None,
        power_watts: None,
        gpu_utilization_percent: None,
    };
    assert!(profile.supports_quantization(&QuantizationLevel::None));
    assert!(profile.supports_quantization(&QuantizationLevel::BFloat16));
    assert!(profile.supports_quantization(&QuantizationLevel::Float16));
    assert!(profile.supports_quantization(&QuantizationLevel::Int8));
    assert!(!profile.supports_quantization(&QuantizationLevel::Int4));
    assert_eq!(
        profile.preferred_quantization(),
        QuantizationLevel::BFloat16
    );
}

#[test]
fn groq_quantization_support() {
    let profile = AcceleratorProfile {
        accelerator: AcceleratorType::GroqLpu { device_id: 0 },
        available: true,
        memory_bytes: 230 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
        device_name: None,
        memory_bandwidth_gbps: None,
        memory_used_bytes: None,
        memory_free_bytes: None,
        pcie_bandwidth_gbps: None,
        numa_node: None,
        temperature_c: None,
        power_watts: None,
        gpu_utilization_percent: None,
    };
    assert!(!profile.supports_quantization(&QuantizationLevel::None)); // No FP32
    assert!(profile.supports_quantization(&QuantizationLevel::Float16));
    assert!(profile.supports_quantization(&QuantizationLevel::Int8));
    assert!(profile.supports_quantization(&QuantizationLevel::Int4));
    assert_eq!(profile.preferred_quantization(), QuantizationLevel::Int8);
}
