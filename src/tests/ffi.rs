//! Tests for the C FFI module.
//!
//! Exercises all exported `extern "C"` functions and verifies null handling,
//! round-trip correctness, and proper memory management.

use std::ffi::CStr;

use crate::ffi::*;

#[test]
fn detect_returns_non_null() {
    let ptr = ai_hwaccel_detect();
    assert!(!ptr.is_null());
    unsafe { ai_hwaccel_free(ptr) };
}

#[test]
fn device_count_at_least_one() {
    let ptr = ai_hwaccel_detect();
    let count = unsafe { ai_hwaccel_device_count(ptr) };
    assert!(count >= 1, "should always have at least CPU, got {}", count);
    unsafe { ai_hwaccel_free(ptr) };
}

#[test]
fn device_count_null_returns_zero() {
    let count = unsafe { ai_hwaccel_device_count(std::ptr::null()) };
    assert_eq!(count, 0);
}

#[test]
fn has_accelerator_null_returns_false() {
    let result = unsafe { ai_hwaccel_has_accelerator(std::ptr::null()) };
    assert!(!result);
}

#[test]
fn accelerator_memory_null_returns_zero() {
    let mem = unsafe { ai_hwaccel_accelerator_memory(std::ptr::null()) };
    assert_eq!(mem, 0);
}

#[test]
fn json_returns_valid_json() {
    let ptr = ai_hwaccel_detect();
    let json_ptr = unsafe { ai_hwaccel_json(ptr) };
    assert!(!json_ptr.is_null());

    let json_str = unsafe { CStr::from_ptr(json_ptr) }.to_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(json_str).unwrap();
    assert!(parsed.is_object());
    assert!(parsed.get("profiles").is_some());

    unsafe { ai_hwaccel_free_string(json_ptr) };
    unsafe { ai_hwaccel_free(ptr) };
}

#[test]
fn json_null_returns_null() {
    let json_ptr = unsafe { ai_hwaccel_json(std::ptr::null()) };
    assert!(json_ptr.is_null());
}

#[test]
fn free_null_is_safe() {
    // Should not crash.
    unsafe { ai_hwaccel_free(std::ptr::null_mut()) };
}

#[test]
fn free_string_null_is_safe() {
    // Should not crash.
    unsafe { ai_hwaccel_free_string(std::ptr::null_mut()) };
}

#[test]
fn full_lifecycle() {
    // detect → query → json → free_string → free
    let ptr = ai_hwaccel_detect();
    assert!(!ptr.is_null());

    let count = unsafe { ai_hwaccel_device_count(ptr) };
    assert!(count >= 1);

    let _has = unsafe { ai_hwaccel_has_accelerator(ptr) };
    let _mem = unsafe { ai_hwaccel_accelerator_memory(ptr) };

    let json_ptr = unsafe { ai_hwaccel_json(ptr) };
    assert!(!json_ptr.is_null());
    unsafe { ai_hwaccel_free_string(json_ptr) };

    unsafe { ai_hwaccel_free(ptr) };
}

#[test]
fn json_roundtrip_matches_registry() {
    let ptr = ai_hwaccel_detect();
    let json_ptr = unsafe { ai_hwaccel_json(ptr) };
    let json_str = unsafe { CStr::from_ptr(json_ptr) }.to_str().unwrap();

    // Deserialize back and check device count matches.
    let reg = crate::AcceleratorRegistry::from_json(json_str).unwrap();
    let ffi_count = unsafe { ai_hwaccel_device_count(ptr) };
    assert_eq!(reg.all_profiles().len() as u32, ffi_count);

    unsafe { ai_hwaccel_free_string(json_ptr) };
    unsafe { ai_hwaccel_free(ptr) };
}
