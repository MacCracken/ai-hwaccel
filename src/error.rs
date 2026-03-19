//! Detection error types.
//!
//! Detection is best-effort: errors are collected as warnings rather than
//! aborting. Callers can inspect them to understand why a backend was skipped.

use std::fmt;

use serde::{Deserialize, Serialize};

/// An error encountered during hardware detection.
///
/// These are non-fatal — detection continues even when individual backends
/// fail. The registry collects them as [`AcceleratorRegistry::warnings`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DetectionError {
    /// A required CLI tool was not found on `$PATH`.
    ToolNotFound {
        tool: String,
    },

    /// A CLI tool was found but exited with a non-zero status.
    ToolFailed {
        tool: String,
        exit_code: Option<i32>,
        stderr: String,
    },

    /// Output from a CLI tool or sysfs file could not be parsed.
    ParseError {
        backend: String,
        message: String,
    },

    /// A sysfs or procfs path could not be read.
    SysfsReadError {
        path: String,
        message: String,
    },
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
