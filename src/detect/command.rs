//! Safe subprocess execution with absolute path resolution, timeouts, and
//! output size limits.
//!
//! Every CLI-based detector should use [`run_tool`] instead of
//! `std::process::Command` directly.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::error::DetectionError;

/// Default timeout for subprocess execution.
pub(crate) const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum bytes to read from subprocess stdout (1 MiB).
pub(crate) const MAX_STDOUT_BYTES: usize = 1024 * 1024;

/// Maximum bytes to read from subprocess stderr (4 KiB).
const MAX_STDERR_BYTES: usize = 4096;

/// Successful output from a tool invocation.
#[derive(Debug)]
pub(crate) struct ToolOutput {
    pub stdout: String,
}

/// Run an external tool with security and robustness guarantees:
///
/// 1. **Absolute path resolution** — the tool is resolved via `$PATH` and
///    invoked by absolute path to prevent `$PATH` hijacking.
/// 2. **Timeout** — the process is killed if it doesn't exit within `timeout`.
/// 3. **Output size limit** — stdout is capped at [`MAX_STDOUT_BYTES`].
pub(crate) fn run_tool(
    tool: &str,
    args: &[&str],
    timeout: Duration,
) -> Result<ToolOutput, DetectionError> {
    // 1. Resolve absolute path.
    let abs_path = which(tool).ok_or_else(|| DetectionError::ToolNotFound { tool: tool.into() })?;

    // 2. Spawn with piped stdout/stderr.
    let mut child = Command::new(&abs_path)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| DetectionError::ToolNotFound { tool: tool.into() })?;

    // Take pipes before the wait loop.
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    // 3. Read stdout in background thread with size limit.
    let stdout_handle = std::thread::spawn(move || read_limited(stdout_pipe, MAX_STDOUT_BYTES));
    let stderr_handle = std::thread::spawn(move || read_limited(stderr_pipe, MAX_STDERR_BYTES));

    // 4. Wait with timeout.
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(DetectionError::ToolFailed {
                    tool: tool.into(),
                    exit_code: None,
                    stderr: format!("timed out after {:.1}s", timeout.as_secs_f64()),
                });
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => {
                return Err(DetectionError::ToolFailed {
                    tool: tool.into(),
                    exit_code: None,
                    stderr: e.to_string(),
                });
            }
        }
    };

    let stdout_bytes = stdout_handle.join().unwrap_or_default();
    let stderr_bytes = stderr_handle.join().unwrap_or_default();

    if !status.success() {
        let stderr_str = String::from_utf8_lossy(&stderr_bytes).to_string();
        return Err(DetectionError::ToolFailed {
            tool: tool.into(),
            exit_code: status.code(),
            stderr: stderr_str,
        });
    }

    Ok(ToolOutput {
        stdout: String::from_utf8_lossy(&stdout_bytes).to_string(),
    })
}

/// Read up to `limit` bytes from an optional pipe.
fn read_limited(pipe: Option<impl Read>, limit: usize) -> Vec<u8> {
    let Some(mut reader) = pipe else {
        return Vec::new();
    };
    let mut buf = vec![0u8; 8192];
    let mut out = Vec::new();
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let remaining = limit.saturating_sub(out.len());
        if remaining == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n.min(remaining)]);
    }
    out
}

/// Resolve an executable name to its absolute path via `$PATH`.
fn which(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var("PATH").ok()?;
    let sep = if cfg!(windows) { ';' } else { ':' };
    for dir in path_var.split(sep) {
        let candidate = Path::new(dir).join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Input validation helpers
// ---------------------------------------------------------------------------

/// Validate a device ID is within a sane range.
pub(crate) fn validate_device_id(raw: &str, backend: &str) -> Result<u32, DetectionError> {
    let id: u32 = raw.parse().map_err(|e| DetectionError::ParseError {
        backend: backend.into(),
        message: format!("invalid device id '{}': {}", raw, e),
    })?;
    if id > 1024 {
        return Err(DetectionError::ParseError {
            backend: backend.into(),
            message: format!("device id {} exceeds maximum (1024)", id),
        });
    }
    Ok(id)
}

/// Validate a memory value in MB is within a sane range (0–16 TiB).
pub(crate) fn validate_memory_mb(raw: &str, backend: &str) -> Result<u64, DetectionError> {
    let mb: u64 = raw.parse().map_err(|e| DetectionError::ParseError {
        backend: backend.into(),
        message: format!("invalid memory '{}': {}", raw, e),
    })?;
    if mb > 16 * 1024 * 1024 {
        // 16 TiB is absurdly high — likely a parse error
        return Err(DetectionError::ParseError {
            backend: backend.into(),
            message: format!("memory {} MB exceeds sanity limit (16 TiB)", mb),
        });
    }
    Ok(mb)
}
