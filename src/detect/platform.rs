//! Platform abstraction for filesystem and command execution.
//!
//! The [`PlatformProbe`] trait abstracts the system-level operations used by
//! hardware detection backends: reading files, running commands, scanning
//! device nodes, and querying system memory.
//!
//! ## Implementations
//!
//! - [`LivePlatform`] — delegates to the real OS (sysfs, `/proc`, `$PATH` tools).
//!   This is what `detect()` uses.
//! - Future: `MockPlatform` for unit testing without real hardware.
//! - Future: `MacOsPlatform` for `sysctl`/`system_profiler` on macOS.
//! - Future: `WindowsPlatform` for WMI/DXGI on Windows.

use std::path::Path;
use std::time::Duration;

use crate::error::DetectionError;

/// Output from a successful command execution.
#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

/// Platform abstraction for hardware detection.
///
/// Each method corresponds to a system-level operation used by one or more
/// detection backends. Implementing this trait for a new platform enables
/// hardware detection without modifying backend logic.
pub trait PlatformProbe: Send + Sync {
    /// Read a file as a string, capped at `max_bytes`.
    ///
    /// Returns `None` if the file doesn't exist or can't be read.
    fn read_file(&self, path: &Path, max_bytes: usize) -> Option<String>;

    /// Read a u64 from a file (e.g. sysfs pseudo-file with a single number).
    fn read_u64(&self, path: &Path) -> Option<u64> {
        self.read_file(path, 64).and_then(|s| s.trim().parse().ok())
    }

    /// Run an external command with arguments and a timeout.
    fn run_command(
        &self,
        tool: &str,
        args: &[&str],
        timeout: Duration,
    ) -> Result<CommandOutput, DetectionError>;

    /// Check whether a path exists on the filesystem.
    fn path_exists(&self, path: &Path) -> bool;

    /// List device IDs from `/dev/<prefix>*` (e.g. `/dev/neuron0`, `/dev/neuron1`).
    ///
    /// Returns sorted device IDs parsed from the suffix.
    fn list_dev_devices(&self, prefix: &str) -> Vec<u32>;

    /// Check whether any `/dev/<prefix>*` device exists.
    fn has_dev_device(&self, prefix: &str) -> bool {
        !self.list_dev_devices(prefix).is_empty()
    }

    /// Total system memory in bytes.
    fn system_memory_bytes(&self) -> u64;

    /// Read the target of a symbolic link.
    fn read_link(&self, path: &Path) -> Option<std::path::PathBuf>;

    /// List entries in a directory.
    fn read_dir_names(&self, path: &Path) -> Vec<String>;
}

/// Live platform: delegates to real OS operations.
///
/// Uses the existing helper functions in `detect/mod.rs` and `detect/command.rs`.
pub struct LivePlatform;

impl PlatformProbe for LivePlatform {
    fn read_file(&self, path: &Path, max_bytes: usize) -> Option<String> {
        super::read_sysfs_string(path, max_bytes)
    }

    fn run_command(
        &self,
        tool: &str,
        args: &[&str],
        timeout: Duration,
    ) -> Result<CommandOutput, DetectionError> {
        super::command::run_tool(tool, args, timeout).map(|o| CommandOutput {
            stdout: o.stdout,
            stderr: String::new(),
            exit_code: Some(0),
        })
    }

    fn path_exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn list_dev_devices(&self, prefix: &str) -> Vec<u32> {
        super::iter_dev_devices(prefix).collect()
    }

    fn system_memory_bytes(&self) -> u64 {
        super::detect_cpu_memory()
    }

    fn read_link(&self, path: &Path) -> Option<std::path::PathBuf> {
        std::fs::read_link(path).ok()
    }

    fn read_dir_names(&self, path: &Path) -> Vec<String> {
        std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect()
    }
}

/// Mock platform for testing without real hardware.
///
/// Stores virtual filesystem entries and command outputs that backends
/// can query during detection.
#[cfg(test)]
pub struct MockPlatform {
    files: std::collections::HashMap<std::path::PathBuf, String>,
    commands: std::collections::HashMap<String, Result<CommandOutput, DetectionError>>,
    dev_devices: std::collections::HashMap<String, Vec<u32>>,
    memory_bytes: u64,
    links: std::collections::HashMap<std::path::PathBuf, std::path::PathBuf>,
    dirs: std::collections::HashMap<std::path::PathBuf, Vec<String>>,
}

#[cfg(test)]
impl Default for MockPlatform {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl MockPlatform {
    /// Create a new empty mock platform.
    pub fn new() -> Self {
        Self {
            files: std::collections::HashMap::new(),
            commands: std::collections::HashMap::new(),
            dev_devices: std::collections::HashMap::new(),
            memory_bytes: 16 * 1024 * 1024 * 1024,
            links: std::collections::HashMap::new(),
            dirs: std::collections::HashMap::new(),
        }
    }

    /// Add a virtual file with content.
    pub fn with_file(mut self, path: impl Into<std::path::PathBuf>, content: &str) -> Self {
        self.files.insert(path.into(), content.to_string());
        self
    }

    /// Add a successful command output.
    pub fn with_command(mut self, tool: &str, stdout: &str) -> Self {
        self.commands.insert(
            tool.to_string(),
            Ok(CommandOutput {
                stdout: stdout.to_string(),
                stderr: String::new(),
                exit_code: Some(0),
            }),
        );
        self
    }

    /// Set system memory.
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.memory_bytes = bytes;
        self
    }

    /// Add virtual /dev/ devices.
    pub fn with_dev_devices(mut self, prefix: &str, ids: Vec<u32>) -> Self {
        self.dev_devices.insert(prefix.to_string(), ids);
        self
    }

    /// Add a symlink.
    pub fn with_link(
        mut self,
        path: impl Into<std::path::PathBuf>,
        target: impl Into<std::path::PathBuf>,
    ) -> Self {
        self.links.insert(path.into(), target.into());
        self
    }

    /// Add directory entries.
    pub fn with_dir(mut self, path: impl Into<std::path::PathBuf>, entries: Vec<String>) -> Self {
        self.dirs.insert(path.into(), entries);
        self
    }
}

#[cfg(test)]
impl PlatformProbe for MockPlatform {
    fn read_file(&self, path: &Path, _max_bytes: usize) -> Option<String> {
        self.files.get(path).cloned()
    }

    fn run_command(
        &self,
        tool: &str,
        _args: &[&str],
        _timeout: Duration,
    ) -> Result<CommandOutput, DetectionError> {
        self.commands
            .get(tool)
            .cloned()
            .unwrap_or(Err(DetectionError::ToolNotFound {
                tool: tool.to_string(),
            }))
    }

    fn path_exists(&self, path: &Path) -> bool {
        self.files.contains_key(path) || self.dirs.contains_key(path)
    }

    fn list_dev_devices(&self, prefix: &str) -> Vec<u32> {
        self.dev_devices.get(prefix).cloned().unwrap_or_default()
    }

    fn system_memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    fn read_link(&self, path: &Path) -> Option<std::path::PathBuf> {
        self.links.get(path).cloned()
    }

    fn read_dir_names(&self, path: &Path) -> Vec<String> {
        self.dirs.get(path).cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_platform_system_memory() {
        let platform = LivePlatform;
        let mem = platform.system_memory_bytes();
        assert!(mem > 0, "system memory should be positive");
    }

    #[test]
    fn live_platform_path_exists() {
        let platform = LivePlatform;
        assert!(platform.path_exists(Path::new("/proc")));
        assert!(!platform.path_exists(Path::new("/nonexistent_path_abc123")));
    }

    #[test]
    fn live_platform_read_file() {
        let platform = LivePlatform;
        let content = platform.read_file(Path::new("/proc/meminfo"), 64 * 1024);
        assert!(content.is_some());
        assert!(content.unwrap().contains("MemTotal"));
    }

    #[test]
    fn mock_platform_basic() {
        let mock = MockPlatform::new()
            .with_file("/sys/class/misc/intel_npu", "")
            .with_memory(64 * 1024 * 1024 * 1024)
            .with_dev_devices("neuron", vec![0, 1, 2])
            .with_command("nvidia-smi", "0, 81920, 0, 81920, 9.0, 550\n");

        assert!(mock.path_exists(Path::new("/sys/class/misc/intel_npu")));
        assert!(!mock.path_exists(Path::new("/nonexistent")));
        assert_eq!(mock.system_memory_bytes(), 64 * 1024 * 1024 * 1024);
        assert_eq!(mock.list_dev_devices("neuron"), vec![0, 1, 2]);
        assert!(mock.has_dev_device("neuron"));
        assert!(!mock.has_dev_device("tpu"));

        let cmd = mock.run_command("nvidia-smi", &[], Duration::from_secs(5));
        assert!(cmd.is_ok());
        assert!(cmd.unwrap().stdout.contains("81920"));

        let missing = mock.run_command("missing-tool", &[], Duration::from_secs(5));
        assert!(missing.is_err());
    }

    #[test]
    fn mock_platform_read_u64() {
        let mock = MockPlatform::new().with_file(
            "/sys/class/drm/card0/device/mem_info_vram_total",
            "17179869184\n",
        );
        assert_eq!(
            mock.read_u64(Path::new("/sys/class/drm/card0/device/mem_info_vram_total")),
            Some(17179869184)
        );
    }
}
