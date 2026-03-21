//! Hardware detection: probes sysfs, /dev, and PATH tools to discover accelerators.

#[cfg(feature = "amd-xdna")]
pub(crate) mod amd_xdna;
#[cfg(feature = "apple")]
pub(crate) mod apple;
pub(crate) mod bandwidth;
#[cfg(feature = "cerebras")]
pub(crate) mod cerebras;
pub(crate) mod command;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
pub(crate) mod disk;
pub(crate) mod environment;
#[cfg(feature = "gaudi")]
pub(crate) mod gaudi;
#[cfg(feature = "graphcore")]
pub(crate) mod graphcore;
#[cfg(feature = "groq")]
pub(crate) mod groq;
#[cfg(feature = "intel-npu")]
pub(crate) mod intel_npu;
#[cfg(feature = "intel-oneapi")]
pub(crate) mod intel_oneapi;
pub(crate) mod interconnect;
#[cfg(feature = "mediatek-apu")]
pub(crate) mod mediatek_apu;
#[cfg(feature = "aws-neuron")]
pub(crate) mod neuron;
pub(crate) mod numa;
pub(crate) mod pcie;
#[cfg(feature = "qualcomm")]
pub(crate) mod qualcomm;
#[cfg(feature = "rocm")]
pub(crate) mod rocm;
#[cfg(feature = "samsung-npu")]
pub(crate) mod samsung_npu;
#[cfg(feature = "tpu")]
pub(crate) mod tpu;
#[cfg(feature = "vulkan")]
pub(crate) mod vulkan;

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use tracing::debug;

use crate::error::DetectionError;
use crate::hardware::AcceleratorType;
use crate::profile::AcceleratorProfile;
use crate::registry::{AcceleratorRegistry, Backend, DetectBuilder};
use crate::system_io::SystemIo;

/// Per-backend detection result.
type DetectResult = (Vec<AcceleratorProfile>, Vec<DetectionError>);

/// Per-backend detection result with timing.
type TimedDetectResult = (Vec<AcceleratorProfile>, Vec<DetectionError>, Duration);

/// Detection results with per-backend timing information.
#[derive(Debug, Clone)]
pub struct TimedDetection {
    /// The registry with all detected hardware.
    pub registry: AcceleratorRegistry,
    /// Per-backend detection duration.
    pub timings: HashMap<String, Duration>,
    /// Total wall-clock detection time.
    pub total: Duration,
}

impl AcceleratorRegistry {
    /// Probes the system for all available accelerators.
    ///
    /// Detection is best-effort: missing tools or sysfs entries simply mean
    /// the corresponding accelerator is not registered. Non-fatal issues are
    /// collected in [`AcceleratorRegistry::warnings`].
    ///
    /// All backends run **in parallel** via [`std::thread::scope`] for
    /// lower wall-clock latency on systems with multiple CLI tools.
    ///
    /// Backends can be disabled at compile time via cargo features
    /// (e.g. `default-features = false, features = ["cuda", "tpu"]`).
    pub fn detect() -> Self {
        detect_with_builder(DetectBuilder::new())
    }

    /// Like [`detect`](Self::detect), but also returns per-backend timing.
    ///
    /// Useful for diagnosing slow backends. The `timings` map contains
    /// backend names (e.g. `"cuda"`, `"vulkan"`) and how long each took.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ai_hwaccel::AcceleratorRegistry;
    ///
    /// let result = AcceleratorRegistry::detect_with_timing();
    /// for (backend, duration) in &result.timings {
    ///     println!("{}: {:.1}ms", backend, duration.as_secs_f64() * 1000.0);
    /// }
    /// ```
    pub fn detect_with_timing() -> TimedDetection {
        detect_with_builder_timed(DetectBuilder::new())
    }
}

/// Run detection with a builder's backend selection.
///
/// When 2+ backends are enabled, runs them in parallel via `std::thread::scope`.
/// When 0-1 are enabled, runs sequentially to avoid thread spawn overhead.
pub(crate) fn detect_with_builder(builder: DetectBuilder) -> AcceleratorRegistry {
    // Pre-allocate for typical system: 1 CPU + up to 8 accelerators.
    let mut all_profiles = Vec::with_capacity(8);
    all_profiles.push(cpu_profile());
    let mut all_warnings: Vec<DetectionError> = Vec::new();

    let use_threads = builder.enabled_count() >= 2;

    macro_rules! run_backend {
        ($feature:literal, $backend:expr, $detect_fn:expr) => {
            #[cfg(feature = $feature)]
            if builder.backend_enabled($backend) {
                $detect_fn(&mut all_profiles, &mut all_warnings);
            }
        };
    }

    macro_rules! spawn_backend {
        ($feature:literal, $backend:expr, $detect_fn:expr, $handles:expr, $s:expr) => {
            #[cfg(feature = $feature)]
            if builder.backend_enabled($backend) {
                $handles.push($s.spawn(|| {
                    let mut p = Vec::new();
                    let mut w = Vec::new();
                    $detect_fn(&mut p, &mut w);
                    (p, w)
                }));
            }
        };
    }

    if use_threads {
        std::thread::scope(|s| {
            let mut handles: Vec<std::thread::ScopedJoinHandle<'_, DetectResult>> = Vec::new();

            spawn_backend!("cuda", Backend::Cuda, cuda::detect_cuda, handles, s);
            spawn_backend!("rocm", Backend::Rocm, rocm::detect_rocm, handles, s);
            spawn_backend!(
                "apple",
                Backend::Apple,
                apple::detect_metal_and_ane,
                handles,
                s
            );
            spawn_backend!("vulkan", Backend::Vulkan, vulkan::detect_vulkan, handles, s);
            spawn_backend!(
                "intel-npu",
                Backend::IntelNpu,
                intel_npu::detect_intel_npu,
                handles,
                s
            );
            spawn_backend!(
                "amd-xdna",
                Backend::AmdXdna,
                amd_xdna::detect_amd_xdna,
                handles,
                s
            );
            spawn_backend!("tpu", Backend::Tpu, tpu::detect_tpu, handles, s);
            spawn_backend!("gaudi", Backend::Gaudi, gaudi::detect_gaudi, handles, s);
            spawn_backend!(
                "aws-neuron",
                Backend::AwsNeuron,
                neuron::detect_aws_neuron,
                handles,
                s
            );
            spawn_backend!(
                "intel-oneapi",
                Backend::IntelOneApi,
                intel_oneapi::detect_intel_oneapi,
                handles,
                s
            );
            spawn_backend!(
                "qualcomm",
                Backend::Qualcomm,
                qualcomm::detect_qualcomm_ai100,
                handles,
                s
            );
            spawn_backend!(
                "cerebras",
                Backend::Cerebras,
                cerebras::detect_cerebras_wse,
                handles,
                s
            );
            spawn_backend!(
                "graphcore",
                Backend::Graphcore,
                graphcore::detect_graphcore_ipu,
                handles,
                s
            );
            spawn_backend!("groq", Backend::Groq, groq::detect_groq_lpu, handles, s);
            spawn_backend!(
                "samsung-npu",
                Backend::SamsungNpu,
                samsung_npu::detect_samsung_npu,
                handles,
                s
            );
            spawn_backend!(
                "mediatek-apu",
                Backend::MediaTekApu,
                mediatek_apu::detect_mediatek_apu,
                handles,
                s
            );

            for handle in handles {
                if let Ok((profiles, warnings)) = handle.join() {
                    all_profiles.extend(profiles);
                    all_warnings.extend(warnings);
                }
            }
        });
    } else {
        run_backend!("cuda", Backend::Cuda, cuda::detect_cuda);
        run_backend!("rocm", Backend::Rocm, rocm::detect_rocm);
        run_backend!("apple", Backend::Apple, apple::detect_metal_and_ane);
        run_backend!("vulkan", Backend::Vulkan, vulkan::detect_vulkan);
        run_backend!("intel-npu", Backend::IntelNpu, intel_npu::detect_intel_npu);
        run_backend!("amd-xdna", Backend::AmdXdna, amd_xdna::detect_amd_xdna);
        run_backend!("tpu", Backend::Tpu, tpu::detect_tpu);
        run_backend!("gaudi", Backend::Gaudi, gaudi::detect_gaudi);
        run_backend!("aws-neuron", Backend::AwsNeuron, neuron::detect_aws_neuron);
        run_backend!(
            "intel-oneapi",
            Backend::IntelOneApi,
            intel_oneapi::detect_intel_oneapi
        );
        run_backend!(
            "qualcomm",
            Backend::Qualcomm,
            qualcomm::detect_qualcomm_ai100
        );
        run_backend!("cerebras", Backend::Cerebras, cerebras::detect_cerebras_wse);
        run_backend!(
            "graphcore",
            Backend::Graphcore,
            graphcore::detect_graphcore_ipu
        );
        run_backend!("groq", Backend::Groq, groq::detect_groq_lpu);
        run_backend!(
            "samsung-npu",
            Backend::SamsungNpu,
            samsung_npu::detect_samsung_npu
        );
        run_backend!(
            "mediatek-apu",
            Backend::MediaTekApu,
            mediatek_apu::detect_mediatek_apu
        );
    }

    // Post-pass: if vulkaninfo found no Vulkan devices, try sysfs fallback.
    #[cfg(feature = "vulkan")]
    {
        let has_vulkan = all_profiles
            .iter()
            .any(|p| matches!(p.accelerator, AcceleratorType::VulkanGpu { .. }));
        let has_dedicated = all_profiles.iter().any(|p| {
            matches!(
                p.accelerator,
                AcceleratorType::CudaGpu { .. } | AcceleratorType::RocmGpu { .. }
            )
        });
        if !has_vulkan && !has_dedicated && builder.backend_enabled(Backend::Vulkan) {
            vulkan::detect_vulkan_sysfs(&mut all_profiles, &mut all_warnings);
        }
    }

    // Post-pass: remove Vulkan GPUs if a dedicated CUDA or ROCm GPU was found.
    let has_dedicated = all_profiles.iter().any(|p| {
        matches!(
            p.accelerator,
            AcceleratorType::CudaGpu { .. } | AcceleratorType::RocmGpu { .. }
        )
    });
    if has_dedicated {
        all_profiles.retain(|p| !matches!(p.accelerator, AcceleratorType::VulkanGpu { .. }));
    }

    // Post-pass: enrich profiles with memory bandwidth, PCIe, and NUMA.
    // Compute PCI address lists once, shared between PCIe and NUMA passes.
    bandwidth::enrich_bandwidth(&mut all_profiles, &mut all_warnings);
    let nvidia_pci = list_driver_pci_addrs("nvidia");
    let amdgpu_pci = list_driver_pci_addrs("amdgpu");
    pcie::enrich_pcie(&mut all_profiles, &nvidia_pci, &amdgpu_pci);
    numa::enrich_numa(&mut all_profiles, &nvidia_pci, &amdgpu_pci);

    // Detect system-level I/O: interconnects and storage.
    let system_interconnects = interconnect::detect_interconnects(&mut all_warnings);
    let system_storage = disk::detect_storage();
    let system_environment = environment::detect_environment();
    let system_io = SystemIo {
        interconnects: system_interconnects,
        storage: system_storage,
        environment: Some(system_environment),
    };

    debug!(
        count = all_profiles.len(),
        warnings = all_warnings.len(),
        interconnects = system_io.interconnects.len(),
        storage_devices = system_io.storage.len(),
        "accelerator detection complete"
    );
    AcceleratorRegistry {
        schema_version: crate::registry::SCHEMA_VERSION,
        profiles: all_profiles,
        warnings: all_warnings,
        system_io,
    }
}

/// Run detection with timing information per backend.
pub(crate) fn detect_with_builder_timed(builder: DetectBuilder) -> TimedDetection {
    let wall_start = Instant::now();
    let mut all_profiles = Vec::with_capacity(8);
    all_profiles.push(cpu_profile());
    let mut all_warnings: Vec<DetectionError> = Vec::new();
    let mut timings: HashMap<String, Duration> = HashMap::new();

    macro_rules! run_backend_timed {
        ($feature:literal, $backend:expr, $name:literal, $detect_fn:expr) => {
            #[cfg(feature = $feature)]
            if builder.backend_enabled($backend) {
                let start = Instant::now();
                $detect_fn(&mut all_profiles, &mut all_warnings);
                timings.insert($name.into(), start.elapsed());
            }
        };
    }

    macro_rules! spawn_backend_timed {
        ($feature:literal, $backend:expr, $name:literal, $detect_fn:expr, $handles:expr, $s:expr) => {
            #[cfg(feature = $feature)]
            if builder.backend_enabled($backend) {
                $handles.push(($name, $s.spawn(|| {
                    let start = Instant::now();
                    let mut p = Vec::new();
                    let mut w = Vec::new();
                    $detect_fn(&mut p, &mut w);
                    (p, w, start.elapsed())
                })));
            }
        };
    }

    let use_threads = builder.enabled_count() >= 2;

    if use_threads {
        std::thread::scope(|s| {
            let mut handles: Vec<(&str, std::thread::ScopedJoinHandle<'_, TimedDetectResult>)> =
                Vec::new();

            spawn_backend_timed!("cuda", Backend::Cuda, "cuda", cuda::detect_cuda, handles, s);
            spawn_backend_timed!("rocm", Backend::Rocm, "rocm", rocm::detect_rocm, handles, s);
            spawn_backend_timed!("apple", Backend::Apple, "apple", apple::detect_metal_and_ane, handles, s);
            spawn_backend_timed!("vulkan", Backend::Vulkan, "vulkan", vulkan::detect_vulkan, handles, s);
            spawn_backend_timed!("intel-npu", Backend::IntelNpu, "intel_npu", intel_npu::detect_intel_npu, handles, s);
            spawn_backend_timed!("amd-xdna", Backend::AmdXdna, "amd_xdna", amd_xdna::detect_amd_xdna, handles, s);
            spawn_backend_timed!("tpu", Backend::Tpu, "tpu", tpu::detect_tpu, handles, s);
            spawn_backend_timed!("gaudi", Backend::Gaudi, "gaudi", gaudi::detect_gaudi, handles, s);
            spawn_backend_timed!("aws-neuron", Backend::AwsNeuron, "aws_neuron", neuron::detect_aws_neuron, handles, s);
            spawn_backend_timed!("intel-oneapi", Backend::IntelOneApi, "intel_oneapi", intel_oneapi::detect_intel_oneapi, handles, s);
            spawn_backend_timed!("qualcomm", Backend::Qualcomm, "qualcomm", qualcomm::detect_qualcomm_ai100, handles, s);
            spawn_backend_timed!("cerebras", Backend::Cerebras, "cerebras", cerebras::detect_cerebras_wse, handles, s);
            spawn_backend_timed!("graphcore", Backend::Graphcore, "graphcore", graphcore::detect_graphcore_ipu, handles, s);
            spawn_backend_timed!("groq", Backend::Groq, "groq", groq::detect_groq_lpu, handles, s);
            spawn_backend_timed!("samsung-npu", Backend::SamsungNpu, "samsung_npu", samsung_npu::detect_samsung_npu, handles, s);
            spawn_backend_timed!("mediatek-apu", Backend::MediaTekApu, "mediatek_apu", mediatek_apu::detect_mediatek_apu, handles, s);

            for (name, handle) in handles {
                if let Ok((profiles, warnings, duration)) = handle.join() {
                    all_profiles.extend(profiles);
                    all_warnings.extend(warnings);
                    timings.insert(name.into(), duration);
                }
            }
        });
    } else {
        run_backend_timed!("cuda", Backend::Cuda, "cuda", cuda::detect_cuda);
        run_backend_timed!("rocm", Backend::Rocm, "rocm", rocm::detect_rocm);
        run_backend_timed!("apple", Backend::Apple, "apple", apple::detect_metal_and_ane);
        run_backend_timed!("vulkan", Backend::Vulkan, "vulkan", vulkan::detect_vulkan);
        run_backend_timed!("intel-npu", Backend::IntelNpu, "intel_npu", intel_npu::detect_intel_npu);
        run_backend_timed!("amd-xdna", Backend::AmdXdna, "amd_xdna", amd_xdna::detect_amd_xdna);
        run_backend_timed!("tpu", Backend::Tpu, "tpu", tpu::detect_tpu);
        run_backend_timed!("gaudi", Backend::Gaudi, "gaudi", gaudi::detect_gaudi);
        run_backend_timed!("aws-neuron", Backend::AwsNeuron, "aws_neuron", neuron::detect_aws_neuron);
        run_backend_timed!("intel-oneapi", Backend::IntelOneApi, "intel_oneapi", intel_oneapi::detect_intel_oneapi);
        run_backend_timed!("qualcomm", Backend::Qualcomm, "qualcomm", qualcomm::detect_qualcomm_ai100);
        run_backend_timed!("cerebras", Backend::Cerebras, "cerebras", cerebras::detect_cerebras_wse);
        run_backend_timed!("graphcore", Backend::Graphcore, "graphcore", graphcore::detect_graphcore_ipu);
        run_backend_timed!("groq", Backend::Groq, "groq", groq::detect_groq_lpu);
        run_backend_timed!("samsung-npu", Backend::SamsungNpu, "samsung_npu", samsung_npu::detect_samsung_npu);
        run_backend_timed!("mediatek-apu", Backend::MediaTekApu, "mediatek_apu", mediatek_apu::detect_mediatek_apu);
    }

    // Post-pass: sysfs Vulkan fallback (same as detect_with_builder).
    #[cfg(feature = "vulkan")]
    {
        let has_vulkan = all_profiles
            .iter()
            .any(|p| matches!(p.accelerator, AcceleratorType::VulkanGpu { .. }));
        let has_dedicated = all_profiles.iter().any(|p| {
            matches!(
                p.accelerator,
                AcceleratorType::CudaGpu { .. } | AcceleratorType::RocmGpu { .. }
            )
        });
        if !has_vulkan && !has_dedicated && builder.backend_enabled(Backend::Vulkan) {
            let start = Instant::now();
            vulkan::detect_vulkan_sysfs(&mut all_profiles, &mut all_warnings);
            timings.insert("vulkan_sysfs".into(), start.elapsed());
        }
    }

    // Post-pass: remove Vulkan GPUs if a dedicated CUDA or ROCm GPU was found.
    let has_dedicated = all_profiles.iter().any(|p| {
        matches!(
            p.accelerator,
            AcceleratorType::CudaGpu { .. } | AcceleratorType::RocmGpu { .. }
        )
    });
    if has_dedicated {
        all_profiles.retain(|p| !matches!(p.accelerator, AcceleratorType::VulkanGpu { .. }));
    }

    let enrich_start = Instant::now();
    bandwidth::enrich_bandwidth(&mut all_profiles, &mut all_warnings);
    let nvidia_pci = list_driver_pci_addrs("nvidia");
    let amdgpu_pci = list_driver_pci_addrs("amdgpu");
    pcie::enrich_pcie(&mut all_profiles, &nvidia_pci, &amdgpu_pci);
    numa::enrich_numa(&mut all_profiles, &nvidia_pci, &amdgpu_pci);
    timings.insert("_enrich".into(), enrich_start.elapsed());

    let sysio_start = Instant::now();
    let system_interconnects = interconnect::detect_interconnects(&mut all_warnings);
    let system_storage = disk::detect_storage();
    let system_environment = environment::detect_environment();
    let system_io = SystemIo {
        interconnects: system_interconnects,
        storage: system_storage,
        environment: Some(system_environment),
    };
    timings.insert("_system_io".into(), sysio_start.elapsed());

    let registry = AcceleratorRegistry {
        schema_version: crate::registry::SCHEMA_VERSION,
        profiles: all_profiles,
        warnings: all_warnings,
        system_io,
    };

    TimedDetection {
        registry,
        timings,
        total: wall_start.elapsed(),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// List PCI addresses bound to a given driver (sorted).
pub(super) fn list_driver_pci_addrs(driver: &str) -> Vec<String> {
    let driver_path = format!("/sys/bus/pci/drivers/{}", driver);
    let dir = Path::new(&driver_path);
    if !dir.exists() {
        return Vec::new();
    }
    let mut addrs: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            // PCI addresses look like "0000:01:00.0"
            if name.contains(':')
                && name.contains('.')
                && name
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() || c == ':' || c == '.')
            {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    addrs.sort();
    addrs
}

/// Build a default CPU profile with detected system memory.
pub(crate) fn cpu_profile() -> AcceleratorProfile {
    AcceleratorProfile {
        accelerator: AcceleratorType::Cpu,
        available: true,
        memory_bytes: detect_cpu_memory(),
        compute_capability: None,
        driver_version: None,
        memory_bandwidth_gbps: None,
        memory_used_bytes: None,
        memory_free_bytes: None,
        pcie_bandwidth_gbps: None,
        numa_node: None,
        temperature_c: None,
        power_watts: None,
        gpu_utilization_percent: None,
    }
}

/// System memory from /proc/meminfo (fallback: 16 GiB).
fn detect_cpu_memory() -> u64 {
    if let Some(info) = read_sysfs_string(std::path::Path::new("/proc/meminfo"), 64 * 1024) {
        for line in info.lines() {
            if line.starts_with("MemTotal:")
                && let Some(kb_str) = line.split_whitespace().nth(1)
                && let Ok(kb) = kb_str.parse::<u64>()
            {
                return kb.saturating_mul(1024);
            }
        }
    }
    // macOS fallback via safe command runner (absolute path, timeout).
    if let Ok(output) = command::run_tool("sysctl", &["-n", "hw.memsize"], command::DEFAULT_TIMEOUT)
        && let Ok(bytes) = output.stdout.trim().parse::<u64>()
    {
        return bytes;
    }
    debug!("could not read system memory, defaulting to 16 GiB");
    16 * 1024 * 1024 * 1024
}

/// Read a u64 from a sysfs file, capped at 64 bytes.
pub(super) fn read_sysfs_u64(path: &Path) -> Option<u64> {
    read_sysfs_string(path, 64).and_then(|s| s.trim().parse().ok())
}

/// Read a string from a sysfs file, capped at `max_bytes` to prevent DoS.
///
/// Sysfs pseudo-files report `st_size = 4096` regardless of actual content,
/// so we can't use metadata for size checking. Instead, we read up to
/// `max_bytes` and discard if truncated.
///
/// Uses a stack buffer for small reads (≤ 512 bytes) to avoid heap allocation
/// in the common case.
pub(super) fn read_sysfs_string(path: &Path, max_bytes: usize) -> Option<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;

    // Stack buffer for common small reads, heap for larger ones.
    const STACK_SIZE: usize = 512;
    if max_bytes < STACK_SIZE {
        let mut buf = [0u8; STACK_SIZE];
        let n = file.read(&mut buf[..max_bytes + 1]).ok()?;
        if n > max_bytes {
            return None;
        }
        return String::from_utf8(buf[..n].to_vec()).ok();
    }

    let mut buf = vec![0u8; max_bytes + 1];
    let n = file.read(&mut buf).ok()?;
    if n > max_bytes {
        return None;
    }
    String::from_utf8(buf[..n].to_vec()).ok()
}

// ---------------------------------------------------------------------------
// True async detection (requires `async-detect` feature)
// ---------------------------------------------------------------------------

/// Async detection orchestrator using `tokio::process::Command`.
///
/// CLI backends run as concurrent tokio tasks with true async subprocess I/O.
/// Sysfs-only backends run in a single `spawn_blocking` task since they are
/// fast filesystem reads. Post-passes (bandwidth, PCIe, NUMA) run after all
/// backends complete.
#[cfg(feature = "async-detect")]
pub(crate) async fn detect_with_builder_async(builder: DetectBuilder) -> AcceleratorRegistry {
    let mut all_profiles = vec![cpu_profile()];
    let mut all_warnings: Vec<DetectionError> = Vec::new();

    debug!(
        backends = builder.enabled_count(),
        "starting async detection"
    );

    // Spawn async CLI backends as concurrent tokio tasks.
    let mut handles: Vec<tokio::task::JoinHandle<DetectResult>> = Vec::new();

    macro_rules! spawn_async_backend {
        ($feature:literal, $backend:expr, $detect_fn:path) => {
            #[cfg(feature = $feature)]
            if builder.backend_enabled($backend) {
                handles.push(tokio::spawn($detect_fn()));
            }
        };
    }

    spawn_async_backend!("cuda", Backend::Cuda, cuda::detect_cuda_async);
    spawn_async_backend!("vulkan", Backend::Vulkan, vulkan::detect_vulkan_async);
    spawn_async_backend!("gaudi", Backend::Gaudi, gaudi::detect_gaudi_async);
    spawn_async_backend!(
        "aws-neuron",
        Backend::AwsNeuron,
        neuron::detect_aws_neuron_async
    );
    spawn_async_backend!("apple", Backend::Apple, apple::detect_metal_and_ane_async);
    spawn_async_backend!(
        "intel-oneapi",
        Backend::IntelOneApi,
        intel_oneapi::detect_intel_oneapi_async
    );

    // Sysfs-only backends run in a single blocking task.
    let sysfs_builder = builder.clone();
    let sysfs_handle = tokio::task::spawn_blocking(move || {
        let mut profiles = Vec::new();
        let mut warnings: Vec<DetectionError> = Vec::new();

        macro_rules! run_sysfs {
            ($feature:literal, $backend:expr, $detect_fn:expr) => {
                #[cfg(feature = $feature)]
                if sysfs_builder.backend_enabled($backend) {
                    $detect_fn(&mut profiles, &mut warnings);
                }
            };
        }

        run_sysfs!("rocm", Backend::Rocm, rocm::detect_rocm);
        run_sysfs!("intel-npu", Backend::IntelNpu, intel_npu::detect_intel_npu);
        run_sysfs!("amd-xdna", Backend::AmdXdna, amd_xdna::detect_amd_xdna);
        run_sysfs!("tpu", Backend::Tpu, tpu::detect_tpu);
        run_sysfs!(
            "qualcomm",
            Backend::Qualcomm,
            qualcomm::detect_qualcomm_ai100
        );
        run_sysfs!("cerebras", Backend::Cerebras, cerebras::detect_cerebras_wse);
        run_sysfs!(
            "graphcore",
            Backend::Graphcore,
            graphcore::detect_graphcore_ipu
        );
        run_sysfs!("groq", Backend::Groq, groq::detect_groq_lpu);
        run_sysfs!(
            "samsung-npu",
            Backend::SamsungNpu,
            samsung_npu::detect_samsung_npu
        );
        run_sysfs!(
            "mediatek-apu",
            Backend::MediaTekApu,
            mediatek_apu::detect_mediatek_apu
        );

        (profiles, warnings)
    });

    // Collect async CLI results.
    for handle in handles {
        if let Ok((profiles, warnings)) = handle.await {
            all_profiles.extend(profiles);
            all_warnings.extend(warnings);
        }
    }

    // Collect sysfs results.
    if let Ok((profiles, warnings)) = sysfs_handle.await {
        all_profiles.extend(profiles);
        all_warnings.extend(warnings);
    }

    // Post-pass: sysfs Vulkan fallback.
    #[cfg(feature = "vulkan")]
    {
        let has_vulkan = all_profiles
            .iter()
            .any(|p| matches!(p.accelerator, AcceleratorType::VulkanGpu { .. }));
        let has_dedicated = all_profiles.iter().any(|p| {
            matches!(
                p.accelerator,
                AcceleratorType::CudaGpu { .. } | AcceleratorType::RocmGpu { .. }
            )
        });
        if !has_vulkan && !has_dedicated && builder.backend_enabled(Backend::Vulkan) {
            vulkan::detect_vulkan_sysfs(&mut all_profiles, &mut all_warnings);
        }
    }

    // Post-pass: remove Vulkan GPUs if a dedicated CUDA or ROCm GPU was found.
    let has_dedicated = all_profiles.iter().any(|p| {
        matches!(
            p.accelerator,
            AcceleratorType::CudaGpu { .. } | AcceleratorType::RocmGpu { .. }
        )
    });
    if has_dedicated {
        all_profiles.retain(|p| !matches!(p.accelerator, AcceleratorType::VulkanGpu { .. }));
    }

    // Post-pass: enrich with bandwidth (async), PCIe, NUMA.
    bandwidth::enrich_bandwidth_async(&mut all_profiles, &mut all_warnings).await;
    let nvidia_pci = list_driver_pci_addrs("nvidia");
    let amdgpu_pci = list_driver_pci_addrs("amdgpu");
    pcie::enrich_pcie(&mut all_profiles, &nvidia_pci, &amdgpu_pci);
    numa::enrich_numa(&mut all_profiles, &nvidia_pci, &amdgpu_pci);

    // System I/O: async interconnects + blocking storage.
    let (system_interconnects, ic_warnings) = interconnect::detect_interconnects_async().await;
    all_warnings.extend(ic_warnings);

    let system_storage = tokio::task::spawn_blocking(disk::detect_storage)
        .await
        .unwrap_or_default();

    let system_environment = environment::detect_environment();
    let system_io = SystemIo {
        interconnects: system_interconnects,
        storage: system_storage,
        environment: Some(system_environment),
    };

    debug!(
        count = all_profiles.len(),
        warnings = all_warnings.len(),
        interconnects = system_io.interconnects.len(),
        storage_devices = system_io.storage.len(),
        "async accelerator detection complete"
    );
    AcceleratorRegistry {
        schema_version: crate::registry::SCHEMA_VERSION,
        profiles: all_profiles,
        warnings: all_warnings,
        system_io,
    }
}
