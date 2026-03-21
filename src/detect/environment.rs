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
    // Try AWS first (fastest — /sys check, no HTTP).
    if let Some(inst) = detect_aws() {
        return Some(inst);
    }
    // Try GCE.
    if let Some(inst) = detect_gce() {
        return Some(inst);
    }
    // Try Azure.
    if let Some(inst) = detect_azure() {
        return Some(inst);
    }
    None
}

/// Detect AWS instance type via instance identity document.
///
/// AWS exposes instance metadata via IMDSv1 at a well-known sysfs path
/// or via the metadata service. We check DMI first (no HTTP needed).
fn detect_aws() -> Option<crate::system_io::CloudInstance> {
    // Check DMI for AWS hypervisor.
    let bios_vendor = std::fs::read_to_string("/sys/class/dmi/id/bios_vendor")
        .unwrap_or_default();
    let sys_vendor = std::fs::read_to_string("/sys/class/dmi/id/sys_vendor")
        .unwrap_or_default();

    let is_aws = bios_vendor.contains("Amazon")
        || sys_vendor.contains("Amazon")
        || sys_vendor.contains("Xen");

    if !is_aws {
        return None;
    }

    // Try to get instance type from DMI product name (e.g., "p4d.24xlarge").
    let instance_type = std::fs::read_to_string("/sys/class/dmi/id/product_name")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.contains('.'));

    // Try instance-id from DMI.
    let instance_id = std::fs::read_to_string("/sys/class/dmi/id/board_asset_tag")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| s.starts_with("i-"));

    // Region from availability zone env or metadata.
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
fn detect_gce() -> Option<crate::system_io::CloudInstance> {
    let product_name = std::fs::read_to_string("/sys/class/dmi/id/product_name")
        .unwrap_or_default();
    let sys_vendor = std::fs::read_to_string("/sys/class/dmi/id/sys_vendor")
        .unwrap_or_default();

    if !product_name.contains("Google") && !sys_vendor.contains("Google") {
        return None;
    }

    // GCE stores machine type in the product name (e.g., "Google Compute Engine").
    // The actual instance type needs the metadata server, but we can detect the platform.
    Some(crate::system_io::CloudInstance {
        provider: "gcp".into(),
        instance_type: None,
        instance_id: None,
        region: None,
        zone: None,
    })
}

/// Detect Azure via DMI.
fn detect_azure() -> Option<crate::system_io::CloudInstance> {
    let sys_vendor = std::fs::read_to_string("/sys/class/dmi/id/sys_vendor")
        .unwrap_or_default();
    let chassis_asset_tag = std::fs::read_to_string("/sys/class/dmi/id/chassis_asset_tag")
        .unwrap_or_default();

    if !sys_vendor.contains("Microsoft") && !chassis_asset_tag.contains("7783-7084-3265-9085") {
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
