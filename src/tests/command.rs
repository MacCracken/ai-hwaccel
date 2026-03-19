//! Tests for the safe command runner and input validation helpers.

use crate::detect::command::{validate_device_id, validate_memory_mb};
use crate::error::DetectionError;

// ---------------------------------------------------------------------------
// validate_device_id
// ---------------------------------------------------------------------------

#[test]
fn valid_device_id() {
    assert_eq!(validate_device_id("0", "test").unwrap(), 0);
    assert_eq!(validate_device_id("7", "test").unwrap(), 7);
    assert_eq!(validate_device_id("1024", "test").unwrap(), 1024);
}

#[test]
fn device_id_non_numeric() {
    let err = validate_device_id("abc", "test").unwrap_err();
    assert!(matches!(err, DetectionError::ParseError { .. }));
}

#[test]
fn device_id_exceeds_max() {
    let err = validate_device_id("1025", "test").unwrap_err();
    assert!(matches!(err, DetectionError::ParseError { .. }));
    assert!(err.to_string().contains("exceeds maximum"));
}

#[test]
fn device_id_negative() {
    let err = validate_device_id("-1", "test").unwrap_err();
    assert!(matches!(err, DetectionError::ParseError { .. }));
}

// ---------------------------------------------------------------------------
// validate_memory_mb
// ---------------------------------------------------------------------------

#[test]
fn valid_memory_mb() {
    assert_eq!(validate_memory_mb("0", "test").unwrap(), 0);
    assert_eq!(validate_memory_mb("8192", "test").unwrap(), 8192);
    assert_eq!(validate_memory_mb("81920", "test").unwrap(), 81920);
}

#[test]
fn memory_mb_non_numeric() {
    let err = validate_memory_mb("NaN", "test").unwrap_err();
    assert!(matches!(err, DetectionError::ParseError { .. }));
}

#[test]
fn memory_mb_exceeds_sanity_limit() {
    // 16 TiB + 1 MB = over limit
    let over = (16 * 1024 * 1024 + 1).to_string();
    let err = validate_memory_mb(&over, "test").unwrap_err();
    assert!(matches!(err, DetectionError::ParseError { .. }));
    assert!(err.to_string().contains("sanity limit"));
}

#[test]
fn memory_mb_at_limit_is_ok() {
    let at_limit = (16 * 1024 * 1024).to_string();
    assert!(validate_memory_mb(&at_limit, "test").is_ok());
}

// ---------------------------------------------------------------------------
// run_tool — tool not found
// ---------------------------------------------------------------------------

#[test]
fn run_tool_not_found() {
    use crate::detect::command::{DEFAULT_TIMEOUT, run_tool};
    let err = run_tool("nonexistent-tool-xyz-12345", &[], DEFAULT_TIMEOUT).unwrap_err();
    assert!(matches!(err, DetectionError::ToolNotFound { .. }));
}

// ---------------------------------------------------------------------------
// run_tool — tool that exits successfully
// ---------------------------------------------------------------------------

#[test]
fn run_tool_echo() {
    use crate::detect::command::{DEFAULT_TIMEOUT, run_tool};
    // `true` is a tool that always exits 0 with no output.
    // It should be on every Unix system.
    if let Ok(output) = run_tool("true", &[], DEFAULT_TIMEOUT) {
        assert!(output.stdout.is_empty() || output.stdout.trim().is_empty());
    }
    // If `true` is not found (unlikely), just skip.
}

// ---------------------------------------------------------------------------
// run_tool — tool that exits with failure
// ---------------------------------------------------------------------------

#[test]
fn run_tool_failure() {
    use crate::detect::command::{DEFAULT_TIMEOUT, run_tool};
    let result = run_tool("false", &[], DEFAULT_TIMEOUT);
    assert!(result.is_err());
    if let Err(DetectionError::ToolFailed { exit_code, .. }) = result {
        assert_eq!(exit_code, Some(1));
    }
}
