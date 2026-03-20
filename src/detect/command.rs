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

/// Environment variables stripped from child processes for security.
///
/// These variables can be used to inject shared libraries into processes.
/// Since we invoke CLI tools for hardware detection (running as the current
/// user), stripping these prevents a compromised environment from injecting
/// code into detection subprocesses.
const SANITIZED_ENV_VARS: &[&str] = &[
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
];

/// Successful output from a tool invocation.
#[derive(Debug)]
pub(crate) struct ToolOutput {
    pub stdout: String,
}

/// Run an external tool with security and robustness guarantees:
///
/// 1. **Absolute path resolution** — the tool is resolved via `$PATH` and
///    invoked by absolute path to prevent `$PATH` hijacking.
/// 2. **Environment sanitization** — `LD_PRELOAD`, `LD_LIBRARY_PATH`,
///    `DYLD_INSERT_LIBRARIES`, and `DYLD_LIBRARY_PATH` are stripped from
///    the child process environment to prevent library injection.
/// 3. **Timeout** — the process is killed if it doesn't exit within `timeout`.
///    Returns [`DetectionError::Timeout`] (not `ToolFailed`) so callers can
///    distinguish slow tools from broken ones.
/// 4. **Output size limit** — stdout is capped at [`MAX_STDOUT_BYTES`].
///
/// # Security
///
/// There is an inherent TOCTOU (time-of-check-time-of-use) gap between
/// `which()` resolving the tool path and `Command::new()` executing it.
/// An attacker with write access to the resolved path could replace the
/// binary between these two operations. This is an accepted risk — it is
/// equivalent to how shells resolve and execute commands, and mitigating it
/// would require `fexecve(2)` which is not portable. The combination of
/// absolute path resolution, environment sanitization, and timeouts limits
/// the attack surface.
pub(crate) fn run_tool(
    tool: &str,
    args: &[&str],
    timeout: Duration,
) -> Result<ToolOutput, DetectionError> {
    // 1. Resolve absolute path.
    let abs_path = which(tool).ok_or_else(|| DetectionError::ToolNotFound { tool: tool.into() })?;

    // 2. Spawn with piped stdout/stderr and sanitized environment.
    let mut cmd = Command::new(&abs_path);
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
    for var in SANITIZED_ENV_VARS {
        cmd.env_remove(var);
    }
    let mut child = cmd
        .spawn()
        .map_err(|_| DetectionError::ToolNotFound { tool: tool.into() })?;

    // 3. Wait with timeout (poll loop at 10ms intervals).
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                // Brief wait for process to exit after kill (max 100ms).
                for _ in 0..10 {
                    if let Ok(Some(_)) = child.try_wait() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                return Err(DetectionError::Timeout {
                    tool: tool.into(),
                    timeout_secs: timeout.as_secs_f64(),
                });
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(e) => {
                return Err(DetectionError::ToolFailed {
                    tool: tool.into(),
                    exit_code: None,
                    stderr: e.to_string(),
                });
            }
        }
    }

    // 4. Process exited — read pipes with size limits. No threads needed
    //    since the child has already exited and the pipes are buffered.
    let stdout_bytes = read_limited(child.stdout.take(), MAX_STDOUT_BYTES);
    let stderr_bytes = read_limited(child.stderr.take(), MAX_STDERR_BYTES);

    let status = child.wait().map_err(|e| DetectionError::ToolFailed {
        tool: tool.into(),
        exit_code: None,
        stderr: e.to_string(),
    })?;

    if !status.success() {
        return Err(DetectionError::ToolFailed {
            tool: tool.into(),
            exit_code: status.code(),
            stderr: String::from_utf8_lossy(&stderr_bytes).into_owned(),
        });
    }

    Ok(ToolOutput {
        stdout: String::from_utf8_lossy(&stdout_bytes).into_owned(),
    })
}

/// Read up to `limit` bytes from an optional pipe.
fn read_limited(pipe: Option<impl Read>, limit: usize) -> Vec<u8> {
    let Some(mut reader) = pipe else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(limit.min(8192));
    let mut buf = [0u8; 8192];
    loop {
        let remaining = limit.saturating_sub(out.len());
        if remaining == 0 {
            break;
        }
        let to_read = buf.len().min(remaining);
        match reader.read(&mut buf[..to_read]) {
            Ok(0) | Err(_) => break,
            Ok(n) => out.extend_from_slice(&buf[..n]),
        }
    }
    out
}

/// Resolve an executable name to its absolute path via `$PATH`.
///
/// On Windows, if `name` has no extension, tries appending `.exe`, `.cmd`,
/// and `.bat` (matching standard `PATHEXT` behavior).
///
/// # Security
///
/// See [`run_tool`] for discussion of the TOCTOU gap between resolution
/// and execution.
fn which(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var("PATH").ok()?;
    let sep = if cfg!(windows) { ';' } else { ':' };

    // On Windows, try common executable extensions if the name has none.
    let extensions: &[&str] = if cfg!(windows) && !name.contains('.') {
        &["", ".exe", ".cmd", ".bat"]
    } else {
        &[""]
    };

    for dir in path_var.split(sep) {
        for ext in extensions {
            let mut candidate = Path::new(dir).join(name);
            if !ext.is_empty() {
                let mut with_ext = candidate.as_os_str().to_os_string();
                with_ext.push(ext);
                candidate = PathBuf::from(with_ext);
            }
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Async variant (requires `async-detect` feature)
// ---------------------------------------------------------------------------

/// Async version of [`run_tool`] using `tokio::process::Command`.
///
/// Uses `tokio::time::timeout` instead of a poll loop, and
/// `tokio::process::Command` for non-blocking subprocess I/O.
/// Same security guarantees as `run_tool` (path resolution, env sanitization,
/// output limits, timeout).
#[cfg(feature = "async-detect")]
pub(crate) async fn run_tool_async(
    tool: &str,
    args: &[&str],
    timeout: Duration,
) -> Result<ToolOutput, DetectionError> {
    let abs_path = which(tool).ok_or_else(|| DetectionError::ToolNotFound { tool: tool.into() })?;

    let mut cmd = tokio::process::Command::new(&abs_path);
    cmd.args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    for var in SANITIZED_ENV_VARS {
        cmd.env_remove(var);
    }
    let child = cmd
        .spawn()
        .map_err(|_| DetectionError::ToolNotFound { tool: tool.into() })?;

    // Wait with timeout.
    let result: Result<std::process::Output, _> =
        match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(r) => r,
            Err(_elapsed) => {
                return Err(DetectionError::Timeout {
                    tool: tool.into(),
                    timeout_secs: timeout.as_secs_f64(),
                });
            }
        };
    let output = result.map_err(|e| DetectionError::ToolFailed {
        tool: tool.into(),
        exit_code: None,
        stderr: e.to_string(),
    })?;

    if !output.status.success() {
        let stderr = &output.stderr[..output.stderr.len().min(MAX_STDERR_BYTES)];
        return Err(DetectionError::ToolFailed {
            tool: tool.into(),
            exit_code: output.status.code(),
            stderr: String::from_utf8_lossy(stderr).into_owned(),
        });
    }

    let stdout = &output.stdout[..output.stdout.len().min(MAX_STDOUT_BYTES)];
    Ok(ToolOutput {
        stdout: String::from_utf8_lossy(stdout).into_owned(),
    })
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
        return Err(DetectionError::ParseError {
            backend: backend.into(),
            message: format!("memory {} MB exceeds sanity limit (16 TiB)", mb),
        });
    }
    Ok(mb)
}
