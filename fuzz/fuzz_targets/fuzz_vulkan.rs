#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        ai_hwaccel::fuzz_helpers::fuzz_vulkan_summary_parser(s);
        ai_hwaccel::fuzz_helpers::fuzz_vulkan_full_parser(s);
    }
});
