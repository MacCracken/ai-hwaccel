//! C-compatible FFI for `ai-hwaccel`.
//!
//! Exposes a minimal C API for detection and querying from non-Rust languages.
//! All functions are `extern "C"` with `#[unsafe(no_mangle)]`.
//!
//! # Memory management
//!
//! - `ai_hwaccel_detect()` returns an opaque pointer. Free it with
//!   `ai_hwaccel_free()`.
//! - `ai_hwaccel_json()` returns a C string. Free it with
//!   `ai_hwaccel_free_string()`.

use std::ffi::CString;
use std::os::raw::c_char;

use crate::registry::AcceleratorRegistry;

/// Opaque handle to an `AcceleratorRegistry`.
pub type HwAccelRegistry = AcceleratorRegistry;

/// Detect all hardware accelerators. Returns an opaque pointer.
///
/// Caller must free with `ai_hwaccel_free()`.
#[unsafe(no_mangle)]
pub extern "C" fn ai_hwaccel_detect() -> *mut HwAccelRegistry {
    let registry = AcceleratorRegistry::detect();
    Box::into_raw(Box::new(registry))
}

/// Free a registry returned by `ai_hwaccel_detect()`.
///
/// # Safety
///
/// `ptr` must be a pointer returned by `ai_hwaccel_detect()`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_hwaccel_free(ptr: *mut HwAccelRegistry) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Number of detected device profiles (including CPU).
///
/// # Safety
///
/// `ptr` must be a valid pointer from `ai_hwaccel_detect()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_hwaccel_device_count(ptr: *const HwAccelRegistry) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    let reg = unsafe { &*ptr };
    reg.all_profiles().len() as u32
}

/// Whether any non-CPU accelerator is available.
///
/// # Safety
///
/// `ptr` must be a valid pointer from `ai_hwaccel_detect()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_hwaccel_has_accelerator(ptr: *const HwAccelRegistry) -> bool {
    if ptr.is_null() {
        return false;
    }
    let reg = unsafe { &*ptr };
    reg.has_accelerator()
}

/// Total accelerator memory in bytes (excluding CPU).
///
/// # Safety
///
/// `ptr` must be a valid pointer from `ai_hwaccel_detect()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_hwaccel_accelerator_memory(ptr: *const HwAccelRegistry) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    let reg = unsafe { &*ptr };
    reg.total_accelerator_memory()
}

/// Serialize the registry to JSON. Returns a null-terminated C string.
///
/// Caller must free with `ai_hwaccel_free_string()`.
///
/// # Safety
///
/// `ptr` must be a valid pointer from `ai_hwaccel_detect()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_hwaccel_json(ptr: *const HwAccelRegistry) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    let reg = unsafe { &*ptr };
    match serde_json::to_string(reg) {
        Ok(json) => match CString::new(json) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a string returned by `ai_hwaccel_json()`.
///
/// # Safety
///
/// `ptr` must be a pointer returned by `ai_hwaccel_json()`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_hwaccel_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}
