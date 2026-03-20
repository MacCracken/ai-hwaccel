//! System-level I/O detection: network interconnects and disk throughput.
//!
//! These are not per-device but per-system, and are critical for multi-node
//! training planning and data loading bottleneck estimation.

use serde::{Deserialize, Serialize};

/// System-wide I/O topology and throughput information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemIo {
    /// Detected network interconnects (InfiniBand, RoCE, NVLink, etc.).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interconnects: Vec<Interconnect>,
    /// Detected storage devices with estimated throughput.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub storage: Vec<StorageDevice>,
}

impl SystemIo {
    /// Empty system I/O (no interconnects or storage detected).
    pub fn empty() -> Self {
        Self {
            interconnects: Vec::new(),
            storage: Vec::new(),
        }
    }

    /// Whether any high-speed interconnect was detected.
    pub fn has_interconnect(&self) -> bool {
        !self.interconnects.is_empty()
    }

    /// Total interconnect bandwidth in GB/s across all ports.
    pub fn total_interconnect_bandwidth_gbps(&self) -> f64 {
        self.interconnects.iter().map(|i| i.bandwidth_gbps).sum()
    }

    /// Estimate data ingestion time for a given dataset size in bytes,
    /// using the fastest detected storage or network path.
    pub fn estimate_ingestion_secs(&self, dataset_bytes: u64) -> Option<f64> {
        let best_storage_gbps = self
            .storage
            .iter()
            .map(|s| s.bandwidth_gbps)
            .fold(0.0f64, f64::max);

        if best_storage_gbps <= 0.0 {
            return None;
        }

        let bytes_per_sec = best_storage_gbps * 1_000_000_000.0;
        Some(dataset_bytes as f64 / bytes_per_sec)
    }
}

/// A detected network interconnect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interconnect {
    /// Interconnect type.
    pub kind: InterconnectKind,
    /// Port or device name (e.g. "mlx5_0", "nvlink0").
    pub name: String,
    /// Link bandwidth in GB/s.
    pub bandwidth_gbps: f64,
    /// Link state (e.g. "Active", "Up").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
}

/// Type of network interconnect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterconnectKind {
    /// InfiniBand (IB).
    InfiniBand,
    /// RDMA over Converged Ethernet.
    RoCE,
    /// NVIDIA NVLink (inter-GPU).
    NVLink,
    /// NVIDIA NVSwitch (multi-GPU fabric).
    NVSwitch,
    /// AMD Infinity Fabric / XGMI (inter-GPU).
    XgmiInfinityFabric,
    /// Google ICI (TPU interconnect).
    Ici,
}

impl std::fmt::Display for InterconnectKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InfiniBand => write!(f, "InfiniBand"),
            Self::RoCE => write!(f, "RoCE"),
            Self::NVLink => write!(f, "NVLink"),
            Self::NVSwitch => write!(f, "NVSwitch"),
            Self::XgmiInfinityFabric => write!(f, "XGMI"),
            Self::Ici => write!(f, "ICI"),
        }
    }
}

/// A detected storage device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageDevice {
    /// Block device name (e.g. "nvme0n1", "sda").
    pub name: String,
    /// Storage type.
    pub kind: StorageKind,
    /// Estimated sequential read bandwidth in GB/s.
    pub bandwidth_gbps: f64,
}

/// Type of storage device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageKind {
    /// NVMe SSD.
    NVMe,
    /// SATA SSD.
    SataSsd,
    /// Rotational HDD.
    Hdd,
    /// Unknown type.
    Unknown,
}

impl std::fmt::Display for StorageKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NVMe => write!(f, "NVMe"),
            Self::SataSsd => write!(f, "SATA SSD"),
            Self::Hdd => write!(f, "HDD"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}
