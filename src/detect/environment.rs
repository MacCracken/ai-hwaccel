//! Runtime environment detection: Docker, Kubernetes, cloud instance metadata.
//!
//! Detects whether the system is running inside a container or VM, and
//! extracts cloud instance metadata when available.

use tracing::debug;

use crate::system_io::RuntimeEnvironment;

/// Detect the runtime environment.
pub(crate) fn detect_environment() -> RuntimeEnvironment {
    let container = detect_container();
    let cloud = detect_cloud_instance();

    debug!(
        is_docker = container.is_docker,
        is_kubernetes = container.is_kubernetes,
        cloud_provider = ?cloud.as_ref().map(|c| &c.provider),
        "environment detection complete"
    );

    let kubernetes_gpu = if container.is_kubernetes {
        detect_kubernetes_gpu()
    } else {
        None
    };

    RuntimeEnvironment {
        is_docker: container.is_docker,
        is_kubernetes: container.is_kubernetes,
        kubernetes_namespace: container.k8s_namespace,
        cloud_instance: cloud,
        kubernetes_gpu,
    }
}

struct ContainerInfo {
    is_docker: bool,
    is_kubernetes: bool,
    k8s_namespace: Option<String>,
}

fn detect_container() -> ContainerInfo {
    let is_docker = std::path::Path::new("/.dockerenv").exists()
        || std::fs::read_to_string("/proc/1/cgroup")
            .map(|s| s.contains("docker") || s.contains("containerd"))
            .unwrap_or(false);

    let k8s_dir = std::path::Path::new("/var/run/secrets/kubernetes.io");
    let is_kubernetes = k8s_dir.exists();
    let k8s_namespace = if is_kubernetes {
        std::fs::read_to_string("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    };

    ContainerInfo {
        is_docker,
        is_kubernetes,
        k8s_namespace,
    }
}

fn detect_cloud_instance() -> Option<crate::system_io::CloudInstanceMeta> {
    // Read DMI files once and pass to all detectors to avoid redundant I/O.
    let dmi = DmiInfo::read();

    if let Some(inst) = detect_aws(&dmi) {
        return Some(inst);
    }
    if let Some(inst) = detect_gce(&dmi) {
        return Some(inst);
    }
    detect_azure(&dmi)
}

/// Cached DMI information read once from sysfs.
struct DmiInfo {
    sys_vendor: String,
    bios_vendor: String,
    product_name: String,
    board_asset_tag: String,
    chassis_asset_tag: String,
}

impl DmiInfo {
    fn read() -> Self {
        let read = |path: &str| {
            std::fs::read_to_string(path)
                .unwrap_or_default()
                .trim()
                .to_string()
        };
        Self {
            sys_vendor: read("/sys/class/dmi/id/sys_vendor"),
            bios_vendor: read("/sys/class/dmi/id/bios_vendor"),
            product_name: read("/sys/class/dmi/id/product_name"),
            board_asset_tag: read("/sys/class/dmi/id/board_asset_tag"),
            chassis_asset_tag: read("/sys/class/dmi/id/chassis_asset_tag"),
        }
    }
}

/// Detect AWS instance type via DMI (no HTTP needed).
fn detect_aws(dmi: &DmiInfo) -> Option<crate::system_io::CloudInstanceMeta> {
    let is_aws = dmi.bios_vendor.contains("Amazon")
        || dmi.sys_vendor.contains("Amazon")
        || dmi.sys_vendor.contains("Xen");

    if !is_aws {
        return None;
    }

    let instance_type = if !dmi.product_name.is_empty() && dmi.product_name.contains('.') {
        Some(dmi.product_name.clone())
    } else {
        None
    };

    let instance_id = if dmi.board_asset_tag.starts_with("i-") {
        Some(dmi.board_asset_tag.clone())
    } else {
        None
    };

    let region = std::env::var("AWS_DEFAULT_REGION")
        .ok()
        .or_else(|| std::env::var("AWS_REGION").ok());

    Some(crate::system_io::CloudInstanceMeta {
        provider: "aws".into(),
        instance_type,
        instance_id,
        region,
        zone: None,
    })
}

/// Detect GCE via DMI.
fn detect_gce(dmi: &DmiInfo) -> Option<crate::system_io::CloudInstanceMeta> {
    if !dmi.product_name.contains("Google") && !dmi.sys_vendor.contains("Google") {
        return None;
    }

    Some(crate::system_io::CloudInstanceMeta {
        provider: "gcp".into(),
        instance_type: None,
        instance_id: None,
        region: None,
        zone: None,
    })
}

/// Detect Azure via DMI.
fn detect_azure(dmi: &DmiInfo) -> Option<crate::system_io::CloudInstanceMeta> {
    if !dmi.sys_vendor.contains("Microsoft")
        && !dmi.chassis_asset_tag.contains("7783-7084-3265-9085")
    {
        return None;
    }

    Some(crate::system_io::CloudInstanceMeta {
        provider: "azure".into(),
        instance_type: None,
        instance_id: None,
        region: None,
        zone: None,
    })
}

/// Detect GPU devices allocated via Kubernetes device plugins.
///
/// Checks environment variables set by NVIDIA device plugin, AMD device
/// plugin, and generic Kubernetes GPU scheduling:
///
/// - `NVIDIA_VISIBLE_DEVICES` — set by NVIDIA k8s device plugin
/// - `GPU_DEVICE_ORDINAL` — set by some schedulers
/// - `CUDA_VISIBLE_DEVICES` — set by NVIDIA container runtime
fn detect_kubernetes_gpu() -> Option<crate::system_io::KubernetesGpuInfo> {
    // Try NVIDIA device plugin first (most common).
    if let Ok(val) = std::env::var("NVIDIA_VISIBLE_DEVICES")
        && !val.is_empty()
        && val != "void"
        && val != "none"
    {
        let device_ids = parse_device_list(&val);
        let gpu_count = if val == "all" {
            0 // Unknown count, all GPUs visible.
        } else {
            device_ids.len() as u32
        };
        debug!(
            device_ids = ?device_ids,
            source = "NVIDIA_VISIBLE_DEVICES",
            "Kubernetes GPU devices detected"
        );
        return Some(crate::system_io::KubernetesGpuInfo {
            device_ids,
            gpu_count,
            source: "NVIDIA_VISIBLE_DEVICES".into(),
        });
    }

    // Try CUDA_VISIBLE_DEVICES (container runtime or user-set).
    if let Ok(val) = std::env::var("CUDA_VISIBLE_DEVICES")
        && !val.is_empty()
    {
        let device_ids = parse_device_list(&val);
        let gpu_count = device_ids.len() as u32;
        debug!(
            device_ids = ?device_ids,
            source = "CUDA_VISIBLE_DEVICES",
            "Kubernetes GPU devices detected"
        );
        return Some(crate::system_io::KubernetesGpuInfo {
            device_ids,
            gpu_count,
            source: "CUDA_VISIBLE_DEVICES".into(),
        });
    }

    // Try GPU_DEVICE_ORDINAL (some schedulers).
    if let Ok(val) = std::env::var("GPU_DEVICE_ORDINAL")
        && !val.is_empty()
    {
        let device_ids = parse_device_list(&val);
        let gpu_count = device_ids.len() as u32;
        debug!(
            device_ids = ?device_ids,
            source = "GPU_DEVICE_ORDINAL",
            "Kubernetes GPU devices detected"
        );
        return Some(crate::system_io::KubernetesGpuInfo {
            device_ids,
            gpu_count,
            source: "GPU_DEVICE_ORDINAL".into(),
        });
    }

    None
}

/// Parse a comma-separated device list (e.g. "0,1,2" or "GPU-uuid1,GPU-uuid2").
fn parse_device_list(val: &str) -> Vec<String> {
    if val == "all" {
        return vec!["all".into()];
    }
    val.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}
