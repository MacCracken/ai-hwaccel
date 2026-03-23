//! Tests for named constants and math correctness.
//!
//! Verifies that the named constants in `units.rs` produce identical results
//! to the original hardcoded values, and that unit conversions are correct.

use crate::units::*;
use crate::*;

// ---------------------------------------------------------------------------
// Constant value sanity checks
// ---------------------------------------------------------------------------

#[test]
fn bytes_per_gib_is_2_pow_30() {
    assert_eq!(BYTES_PER_GIB as u64, 1 << 30);
    assert_eq!(BYTES_PER_GIB_U64, 1 << 30);
}

#[test]
fn bytes_per_gb_is_1e9() {
    assert_eq!(BYTES_PER_GB, 1e9);
}

#[test]
fn bits_per_byte_is_8() {
    assert_eq!(BITS_PER_BYTE, 8.0);
}

#[test]
fn ddr_multiplier_is_2() {
    assert_eq!(DDR_MULTIPLIER, 2.0);
}

#[test]
fn pcie_encoding_ratios() {
    // Gen3+: 128b/130b ≈ 0.9846
    assert!((PCIE_GEN3_PLUS_ENCODING - 128.0 / 130.0).abs() < f64::EPSILON);
    // Gen1/2: 8b/10b = 0.8
    assert!((PCIE_GEN1_GEN2_ENCODING - 0.8).abs() < f64::EPSILON);
}

#[test]
fn pcie_gen3_speed_threshold() {
    assert_eq!(PCIE_GEN3_SPEED_GTS, 8.0);
}

#[test]
fn fp16_bytes_per_param_is_2() {
    assert_eq!(FP16_BYTES_PER_PARAM, 2.0);
}

#[test]
fn fp32_bits_is_32() {
    assert_eq!(FP32_BITS, 32.0);
}

#[test]
fn params_per_million_is_1e6() {
    assert_eq!(PARAMS_PER_MILLION, 1e6);
}

#[test]
fn activation_overhead_divisor_gives_20_percent() {
    // raw + raw/5 = raw * 1.2
    let raw = 1000u64;
    let with_overhead = raw + raw / ACTIVATION_OVERHEAD_DIVISOR;
    assert_eq!(with_overhead, 1200);
}

#[test]
fn tokens_per_sec_base_is_1e9() {
    assert_eq!(TOKENS_PER_SEC_BASE, 1e9);
}

#[test]
fn params_per_layer_estimate_is_250m() {
    assert_eq!(PARAMS_PER_LAYER_ESTIMATE, 250_000_000);
}

// ---------------------------------------------------------------------------
// Sharding planner constants
// ---------------------------------------------------------------------------

#[test]
fn nvswitch_bonus_at_least_non_nvswitch_max() {
    const { assert!(NVSWITCH_TP_BONUS >= 1.0 + MAX_NON_NVSWITCH_TP_BONUS) };
}

#[test]
fn pipeline_efficiency_ordering() {
    const { assert!(PP_HIGH_BW_EFFICIENCY > PP_PCIE_ONLY_EFFICIENCY) };
    const { assert!(PP_HIGH_BW_EFFICIENCY <= 1.0) };
    const { assert!(PP_PCIE_ONLY_EFFICIENCY > 0.0) };
}

// ---------------------------------------------------------------------------
// Math equivalence: verify constants produce same results as original hardcoded values
// ---------------------------------------------------------------------------

#[test]
fn memory_bandwidth_formula_matches_hardcoded() {
    // H100: 2619 MHz * 5120-bit * 2 / 8 / 1000
    let hardcoded = 2619.0 * 5120.0 * 2.0 / 8.0 / 1000.0;
    let with_constants =
        2619.0 * 5120.0 * DDR_MULTIPLIER / BITS_PER_BYTE / MHZ_PER_GHZ;
    assert!(
        (hardcoded - with_constants).abs() < f64::EPSILON,
        "hardcoded={}, constants={}",
        hardcoded,
        with_constants
    );
}

#[test]
fn pcie_bandwidth_formula_matches_hardcoded() {
    // Gen4 x16: 16 GT/s * 16 * (128/130) / 8
    let hardcoded = 16.0 * 16.0 * (128.0 / 130.0) / 8.0;
    let with_constants = 16.0 * 16.0 * PCIE_GEN3_PLUS_ENCODING / BITS_PER_BYTE;
    assert!(
        (hardcoded - with_constants).abs() < f64::EPSILON,
        "hardcoded={}, constants={}",
        hardcoded,
        with_constants
    );
}

#[test]
fn training_base_gb_formula_matches_hardcoded() {
    let model_params_millions = 7000u64;
    let hardcoded = (model_params_millions as f64 * 1_000_000.0 * 2.0) / 1_073_741_824.0;
    let with_constants = (model_params_millions as f64 * PARAMS_PER_MILLION
        * FP16_BYTES_PER_PARAM)
        / BYTES_PER_GIB;
    assert!(
        (hardcoded - with_constants).abs() < f64::EPSILON,
        "hardcoded={}, constants={}",
        hardcoded,
        with_constants
    );
}

#[test]
fn estimate_memory_with_constants_matches_original() {
    // Original: model_params * bits / 8 + (model_params * bits / 8) / 5
    let params = 7_000_000_000u64;
    let bits = 16u64; // FP16
    let original_raw = params * bits / 8;
    let original = original_raw + original_raw / 5;

    let constant_raw = params * bits / BITS_PER_BYTE as u64;
    let constant = constant_raw + constant_raw / ACTIVATION_OVERHEAD_DIVISOR;

    assert_eq!(original, constant);
}

#[test]
fn ingestion_time_formula_matches_hardcoded() {
    let dataset_bytes = 100_000_000_000u64; // 100 GB
    let bandwidth_gbps = 3.5;

    let hardcoded = dataset_bytes as f64 / (bandwidth_gbps * 1_000_000_000.0);
    let with_constants = dataset_bytes as f64 / (bandwidth_gbps * BYTES_PER_GB);
    assert!(
        (hardcoded - with_constants).abs() < f64::EPSILON,
        "hardcoded={}, constants={}",
        hardcoded,
        with_constants
    );
}

#[test]
fn ib_rate_conversion_matches_hardcoded() {
    let gbits = 400.0;
    let hardcoded = gbits / 8.0;
    let with_constants = gbits / GBITS_PER_GBYTE;
    assert!(
        (hardcoded - with_constants).abs() < f64::EPSILON,
        "hardcoded={}, constants={}",
        hardcoded,
        with_constants
    );
}

// ---------------------------------------------------------------------------
// Layer estimation with constant
// ---------------------------------------------------------------------------

#[test]
fn layer_estimation_70b() {
    let params = 70_000_000_000u64;
    let layers = (params / PARAMS_PER_LAYER_ESTIMATE).max(1) as u32;
    assert_eq!(layers, 280);
}

#[test]
fn layer_estimation_7b() {
    let params = 7_000_000_000u64;
    let layers = (params / PARAMS_PER_LAYER_ESTIMATE).max(1) as u32;
    assert_eq!(layers, 28);
}

#[test]
fn layer_estimation_small_model_clamps_to_1() {
    let params = 100_000_000u64; // 100M, less than 250M per layer
    let layers = (params / PARAMS_PER_LAYER_ESTIMATE).max(1) as u32;
    assert_eq!(layers, 1);
}

// ---------------------------------------------------------------------------
// End-to-end: verify plan_sharding and estimate_training_memory still produce
// the same results (no accidental value changes from the refactor).
// ---------------------------------------------------------------------------

#[test]
fn estimate_memory_regression_7b_fp16() {
    let mem = AcceleratorRegistry::estimate_memory(7_000_000_000, &QuantizationLevel::Float16);
    // 7B * 16 bits / 8 = 14 GB raw, +20% = 16.8 GB
    let expected = 7_000_000_000u64 * 16 / 8;
    let expected = expected + expected / 5;
    assert_eq!(mem, expected);
}

#[test]
fn estimate_memory_regression_70b_int4() {
    let mem = AcceleratorRegistry::estimate_memory(70_000_000_000, &QuantizationLevel::Int4);
    let expected = 70_000_000_000u64 * 4 / 8;
    let expected = expected + expected / 5;
    assert_eq!(mem, expected);
}

#[test]
fn training_memory_regression_7b_lora_gpu() {
    let est = estimate_training_memory(7000, TrainingMethod::LoRA, TrainingTarget::Gpu);
    let base_gb = (7000.0 * PARAMS_PER_MILLION * FP16_BYTES_PER_PARAM) / BYTES_PER_GIB;
    let expected_model = base_gb;
    let expected_optimizer = base_gb * 0.1;
    let expected_activation = base_gb * 0.1;
    assert!((est.model_gb - expected_model).abs() < 0.001);
    assert!((est.optimizer_gb - expected_optimizer).abs() < 0.001);
    assert!((est.activation_gb - expected_activation).abs() < 0.001);
}
