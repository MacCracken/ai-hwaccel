//! Detection error types.
//!
//! Detection is best-effort: errors are collected as warnings rather than
//! aborting. Callers can inspect them to understand why a backend was skipped.

use std::fmt;

use serde::{Deserialize, Serialize};

/// An error encountered during hardware detection.
///
/// These are non-fatal — detection continues even when individual backends
/// fail. The registry collects them as [`crate::AcceleratorRegistry::warnings`].
///
/// # Examples
///
/// ```rust
/// use ai_hwaccel::DetectionError;
///
/// let err = DetectionError::ToolNotFound { tool: "nvidia-smi".into() };
/// assert!(err.to_string().contains("nvidia-smi"));
///
/// let err = DetectionError::Timeout { tool: "hl-smi".into(), timeout_secs: 5.0 };
/// assert!(err.to_string().contains("timed out"));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DetectionError {
    /// A required CLI tool was not found on `$PATH`.
    ToolNotFound { tool: String },

    /// A CLI tool was found but exited with a non-zero status.
    ToolFailed {
        tool: String,
        exit_code: Option<i32>,
        stderr: String,
    },

    /// A CLI tool did not exit within the allowed timeout.
    ///
    /// The process was killed. This is distinct from [`ToolFailed`](Self::ToolFailed)
    /// to allow callers to implement retry logic for transient slowness.
    Timeout { tool: String, timeout_secs: f64 },

    /// Output from a CLI tool or sysfs file could not be parsed.
    ParseError { backend: String, message: String },

    /// A sysfs or procfs path could not be read.
    SysfsReadError { path: String, message: String },
}

impl fmt::Display for DetectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ToolNotFound { tool } => {
                write!(f, "{}: tool not found on $PATH", tool)
            }
            Self::ToolFailed {
                tool,
                exit_code,
                stderr,
            } => {
                write!(
                    f,
                    "{}: exited with code {} — {}",
                    tool,
                    exit_code
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "signal".into()),
                    stderr.lines().next().unwrap_or("(no output)")
                )
            }
            Self::Timeout { tool, timeout_secs } => {
                write!(f, "{}: timed out after {:.1}s", tool, timeout_secs)
            }
            Self::ParseError { backend, message } => {
                write!(f, "{}: parse error — {}", backend, message)
            }
            Self::SysfsReadError { path, message } => {
                write!(f, "sysfs {}: {}", path, message)
            }
        }
    }
}

impl std::error::Error for DetectionError {}
