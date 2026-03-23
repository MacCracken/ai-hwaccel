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

    RuntimeEnvironment {
        is_docker: container.is_docker,
        is_kubernetes: container.is_kubernetes,
        kubernetes_namespace: container.k8s_namespace,
        cloud_instance: cloud,
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

fn detect_cloud_instance() -> Option<crate::system_io::CloudInstance> {
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
fn detect_aws(dmi: &DmiInfo) -> Option<crate::system_io::CloudInstance> {
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

    Some(crate::system_io::CloudInstance {
        provider: "aws".into(),
        instance_type,
        instance_id,
        region,
        zone: None,
    })
}

/// Detect GCE via DMI.
fn detect_gce(dmi: &DmiInfo) -> Option<crate::system_io::CloudInstance> {
    if !dmi.product_name.contains("Google") && !dmi.sys_vendor.contains("Google") {
        return None;
    }

    Some(crate::system_io::CloudInstance {
        provider: "gcp".into(),
        instance_type: None,
        instance_id: None,
        region: None,
        zone: None,
    })
}

/// Detect Azure via DMI.
fn detect_azure(dmi: &DmiInfo) -> Option<crate::system_io::CloudInstance> {
    if !dmi.sys_vendor.contains("Microsoft")
        && !dmi.chassis_asset_tag.contains("7783-7084-3265-9085")
    {
        return None;
    }

    Some(crate::system_io::CloudInstance {
        provider: "azure".into(),
        instance_type: None,
        instance_id: None,
        region: None,
        zone: None,
    })
}
