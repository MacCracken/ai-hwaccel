//! Parsing functions exposed for fuzz testing.
//!
//! Not part of the public API — gated behind the `fuzz` feature.

/// Fuzz the CUDA `nvidia-smi` CSV parser.
pub fn fuzz_cuda_parser(input: &str) {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::cuda::parse_cuda_output(input, &mut profiles, &mut warnings);
}

/// Fuzz the Gaudi `hl-smi` CSV parser.
pub fn fuzz_gaudi_parser(input: &str) {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::gaudi::parse_gaudi_output(input, &mut profiles, &mut warnings);
}

/// Fuzz the Vulkan summary parser.
pub fn fuzz_vulkan_summary_parser(input: &str) {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::vulkan::parse_vulkan_output(input, None, &mut profiles, &mut warnings);
}

/// Fuzz the Vulkan full output parser.
pub fn fuzz_vulkan_full_parser(input: &str) {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::vulkan::parse_vulkan_output("", Some(input), &mut profiles, &mut warnings);
}

/// Fuzz the Intel oneAPI `xpu-smi` CSV parser.
pub fn fuzz_intel_oneapi_parser(input: &str) {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::intel_oneapi::parse_xpu_smi_output(input, &mut profiles, &mut warnings);
}

/// Fuzz the AWS Neuron `neuron-ls` JSON parser.
pub fn fuzz_neuron_parser(input: &str) {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::neuron::parse_neuron_output(input, &mut profiles, &mut warnings);
}

/// Fuzz the Apple `system_profiler` parser.
pub fn fuzz_apple_parser(input: &str) {
    let mut profiles = Vec::new();
    let mut warnings = Vec::new();
    crate::detect::apple::parse_system_profiler_output(input, &mut profiles, &mut warnings);
}

/// Fuzz the NVLink output parser.
pub fn fuzz_nvlink_parser(input: &str) {
    let mut interconnects = Vec::new();
    crate::detect::interconnect::parse_nvlink_output(input, &mut interconnects);
}

/// Fuzz the NVSwitch topology parser.
pub fn fuzz_nvswitch_topo_parser(input: &str) {
    let mut interconnects = Vec::new();
    crate::detect::interconnect::parse_nvswitch_topo(input, &mut interconnects);
}

/// Fuzz the XGMI topology parser.
pub fn fuzz_xgmi_topo_parser(input: &str) {
    let mut interconnects = Vec::new();
    crate::detect::interconnect::parse_xgmi_topo(input, &mut interconnects);
}

/// Fuzz the nvidia-smi bandwidth parser.
pub fn fuzz_nvidia_bandwidth_parser(input: &str) {
    let _ = crate::detect::bandwidth::parse_nvidia_bandwidth_output(input);
}

/// Fuzz the InfiniBand rate parser.
pub fn fuzz_ib_rate_parser(input: &str) {
    let _ = crate::detect::interconnect::parse_ib_rate(input);
}

/// Fuzz the PCIe link speed parser.
pub fn fuzz_pcie_link_speed_parser(input: &str) {
    let _ = crate::detect::pcie::parse_link_speed(input);
}

/// Fuzz the DPM clock parser.
pub fn fuzz_dpm_clock_parser(input: &str) {
    let _ = crate::detect::bandwidth::parse_max_dpm_clock(input);
}

/// Fuzz the AcceleratorRegistry JSON deserializer.
///
/// This is security-critical: malicious JSON should never crash the parser
/// or cause unbounded memory allocation.
pub fn fuzz_registry_from_json(input: &str) {
    let _ = crate::registry::AcceleratorRegistry::from_json(input);
}

/// Fuzz the Cerebras CLI memory parser.
pub fn fuzz_cerebras_parser(input: &str) {
    let _ = crate::detect::cerebras::parse_memory_from_cli(input);
}

/// Fuzz the Graphcore gc-info memory parser.
pub fn fuzz_graphcore_parser(input: &str) {
    let _ = crate::detect::graphcore::parse_memory_from_gcinfo(input);
}

/// Fuzz the model format detector with raw bytes.
pub fn fuzz_model_format_detector(input: &[u8]) {
    let _ = crate::model_format::detect_format_from_bytes(input);
}
