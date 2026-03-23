//! Named constants for unit conversions and hardware math.
//!
//! Replaces magic numbers throughout the codebase with self-documenting
//! constants. All values are compile-time constants.

// ---------------------------------------------------------------------------
// Byte / bit conversions
// ---------------------------------------------------------------------------

/// Bits per byte (8).
pub const BITS_PER_BYTE: f64 = 8.0;

/// Bytes per gibibyte (1 GiB = 1,073,741,824 bytes).
pub const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// Bytes per gibibyte as `u64`.
pub const BYTES_PER_GIB_U64: u64 = 1_073_741_824;

/// Bytes per gigabyte (1 GB = 1,000,000,000 bytes, SI decimal).
pub const BYTES_PER_GB: f64 = 1_000_000_000.0;

/// Megahertz-to-gigahertz divisor.
pub const MHZ_PER_GHZ: f64 = 1000.0;

// ---------------------------------------------------------------------------
// DDR memory
// ---------------------------------------------------------------------------

/// DDR (Double Data Rate) multiplier — memory transfers twice per clock cycle.
pub const DDR_MULTIPLIER: f64 = 2.0;

// ---------------------------------------------------------------------------
// PCIe encoding overhead
// ---------------------------------------------------------------------------

/// PCIe Gen3+ uses 128b/130b encoding (~98.5% efficiency).
pub const PCIE_GEN3_PLUS_ENCODING: f64 = 128.0 / 130.0;

/// PCIe Gen1/Gen2 uses 8b/10b encoding (80% efficiency).
pub const PCIE_GEN1_GEN2_ENCODING: f64 = 8.0 / 10.0;

/// PCIe Gen3+ speed threshold in GT/s.
pub const PCIE_GEN3_SPEED_GTS: f64 = 8.0;

// ---------------------------------------------------------------------------
// Model parameter conversions
// ---------------------------------------------------------------------------

/// Parameters per million (for converting `model_params_millions` inputs).
pub const PARAMS_PER_MILLION: f64 = 1_000_000.0;

/// Bytes per parameter in FP16/BF16 (2 bytes = 16 bits).
pub const FP16_BYTES_PER_PARAM: f64 = 2.0;

/// FP32 bits per parameter (used as baseline for memory reduction factor).
pub const FP32_BITS: f64 = 32.0;

// ---------------------------------------------------------------------------
// Memory estimation heuristics
// ---------------------------------------------------------------------------

/// Activation / KV-cache overhead factor (20% of raw model size).
///
/// Applied as: `raw + raw / ACTIVATION_OVERHEAD_DIVISOR`.
pub const ACTIVATION_OVERHEAD_DIVISOR: u64 = 5;

/// Estimated parameters per transformer layer (~250M).
///
/// Used by the pipeline-parallel planner to estimate layer count from
/// total parameter count.
pub const PARAMS_PER_LAYER_ESTIMATE: u64 = 250_000_000;

/// Base tokens/sec numerator: 1 billion (1B params → 1 tok/s baseline).
pub const TOKENS_PER_SEC_BASE: f64 = 1_000_000_000.0;

// ---------------------------------------------------------------------------
// Sharding planner thresholds
// ---------------------------------------------------------------------------

/// NVSwitch interconnect bonus for tensor-parallel throughput.
pub const NVSWITCH_TP_BONUS: f64 = 1.8;

/// Maximum interconnect bonus when NVSwitch is absent.
pub const MAX_NON_NVSWITCH_TP_BONUS: f64 = 0.8;

/// Divisor for scaling interconnect bandwidth into a bonus factor.
pub const TP_INTERCONNECT_BW_DIVISOR: f64 = 200.0;

/// Pipeline-parallel efficiency with high-bandwidth interconnect (~15% overhead).
pub const PP_HIGH_BW_EFFICIENCY: f64 = 0.85;

/// Pipeline-parallel efficiency with PCIe-only (~35% overhead).
pub const PP_PCIE_ONLY_EFFICIENCY: f64 = 0.65;

/// Minimum interconnect bandwidth (GB/s) to consider tensor-parallel.
pub const TP_MIN_INTERCONNECT_BW: f64 = 100.0;

/// Maximum GPU count for tensor-parallel without NVSwitch.
pub const TP_MAX_DEVICES_WITHOUT_NVSWITCH: usize = 8;

/// TPU tensor-parallel ICI bonus multiplier.
pub const TPU_TP_ICI_BONUS: f64 = 2.0;

/// Gigabit-to-gigabyte divisor (for IB rate conversion: Gb/s → GB/s).
pub const GBITS_PER_GBYTE: f64 = 8.0;
