//! Tests for SystemIo, power/thermal fields, and from_json.

use crate::*;

// ---------------------------------------------------------------------------
// SystemIo serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn serde_system_io_roundtrip() {
    let sio = SystemIo {
        interconnects: vec![Interconnect {
            kind: InterconnectKind::InfiniBand,
            name: "mlx5_0".into(),
            bandwidth_gbps: 25.0,
            state: Some("Active".into()),
        }],
        storage: vec![StorageDevice {
            name: "nvme0n1".into(),
            kind: StorageKind::NVMe,
            bandwidth_gbps: 3.5,
        }],
        environment: None,
    };
    let json = serde_json::to_string(&sio).unwrap();
    let back: SystemIo = serde_json::from_str(&json).unwrap();
    assert_eq!(sio, back);
}

#[test]
fn serde_system_io_empty_roundtrip() {
    let sio = SystemIo::empty();
    let json = serde_json::to_string(&sio).unwrap();
    let back: SystemIo = serde_json::from_str(&json).unwrap();
    assert_eq!(sio, back);
}

#[test]
fn serde_interconnect_kind_all_variants() {
    let kinds = [
        InterconnectKind::InfiniBand,
        InterconnectKind::RoCE,
        InterconnectKind::NVLink,
        InterconnectKind::NVSwitch,
        InterconnectKind::XgmiInfinityFabric,
        InterconnectKind::Ici,
    ];
    for k in &kinds {
        let json = serde_json::to_string(k).unwrap();
        let back: InterconnectKind = serde_json::from_str(&json).unwrap();
        assert_eq!(*k, back);
    }
}

#[test]
fn serde_storage_kind_all_variants() {
    let kinds = [
        StorageKind::NVMe,
        StorageKind::SataSsd,
        StorageKind::Hdd,
        StorageKind::Unknown,
    ];
    for k in &kinds {
        let json = serde_json::to_string(k).unwrap();
        let back: StorageKind = serde_json::from_str(&json).unwrap();
        assert_eq!(*k, back);
    }
}

// ---------------------------------------------------------------------------
// SystemIo methods
// ---------------------------------------------------------------------------

#[test]
fn system_io_has_interconnect() {
    let empty = SystemIo::empty();
    assert!(!empty.has_interconnect());

    let with_ib = SystemIo {
        interconnects: vec![Interconnect {
            kind: InterconnectKind::InfiniBand,
            name: "mlx5_0".into(),
            bandwidth_gbps: 25.0,
            state: None,
        }],
        storage: vec![],
        environment: None,
    };
    assert!(with_ib.has_interconnect());
}

#[test]
fn system_io_total_bandwidth() {
    let sio = SystemIo {
        interconnects: vec![
            Interconnect {
                kind: InterconnectKind::InfiniBand,
                name: "mlx5_0".into(),
                bandwidth_gbps: 25.0,
                state: None,
            },
            Interconnect {
                kind: InterconnectKind::InfiniBand,
                name: "mlx5_1".into(),
                bandwidth_gbps: 25.0,
                state: None,
            },
        ],
        storage: vec![],
        environment: None,
    };
    assert!((sio.total_interconnect_bandwidth_gbps() - 50.0).abs() < 0.01);
}

#[test]
fn system_io_ingestion_estimate() {
    let sio = SystemIo {
        interconnects: vec![],
        storage: vec![StorageDevice {
            name: "nvme0n1".into(),
            kind: StorageKind::NVMe,
            bandwidth_gbps: 3.5,
        }],
        environment: None,
    };
    let secs = sio
        .estimate_ingestion_secs(100 * 1024 * 1024 * 1024)
        .unwrap();
    // 100 GB / 3.5 GB/s ≈ 28.6 seconds
    assert!(secs > 25.0 && secs < 35.0, "got {}", secs);
}

#[test]
fn system_io_ingestion_no_storage() {
    let sio = SystemIo::empty();
    assert!(sio.estimate_ingestion_secs(100).is_none());
}

// ---------------------------------------------------------------------------
// Power/thermal fields serde
// ---------------------------------------------------------------------------

#[test]
fn serde_profile_with_power_thermal() {
    let mut p = AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024);
    p.temperature_c = Some(72);
    p.power_watts = Some(300.5);
    p.gpu_utilization_percent = Some(95);

    let json = serde_json::to_string(&p).unwrap();
    assert!(json.contains("temperature_c"));
    assert!(json.contains("power_watts"));
    assert!(json.contains("gpu_utilization_percent"));

    let back: AcceleratorProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(back.temperature_c, Some(72));
    assert_eq!(back.power_watts, Some(300.5));
    assert_eq!(back.gpu_utilization_percent, Some(95));
}

#[test]
fn serde_profile_without_power_thermal_omits_fields() {
    let p = AcceleratorProfile::cuda(0, 24 * 1024 * 1024 * 1024);
    let json = serde_json::to_string(&p).unwrap();
    assert!(!json.contains("temperature_c"));
    assert!(!json.contains("power_watts"));
    assert!(!json.contains("gpu_utilization_percent"));
}

#[test]
fn serde_old_json_without_new_fields_deserializes() {
    // Simulate v1 JSON without any 0.20 fields
    let json = r#"{
        "accelerator": {"CudaGpu": {"device_id": 0}},
        "available": true,
        "memory_bytes": 25769803776,
        "compute_capability": "8.6",
        "driver_version": null
    }"#;
    let p: AcceleratorProfile = serde_json::from_str(json).unwrap();
    assert_eq!(p.memory_bytes, 25769803776);
    assert!(p.temperature_c.is_none());
    assert!(p.power_watts.is_none());
    assert!(p.memory_bandwidth_gbps.is_none());
    assert!(p.numa_node.is_none());
}

// ---------------------------------------------------------------------------
// from_json schema version validation
// ---------------------------------------------------------------------------

#[test]
fn from_json_current_version() {
    let reg = AcceleratorRegistry::new();
    let json = serde_json::to_string(&reg).unwrap();
    let back = AcceleratorRegistry::from_json(&json).unwrap();
    assert_eq!(back.schema_version(), SCHEMA_VERSION);
}

#[test]
fn from_json_old_version_ok() {
    let json =
        r#"{"schema_version":1,"profiles":[],"system_io":{"interconnects":[],"storage":[]}}"#;
    let reg = AcceleratorRegistry::from_json(json).unwrap();
    assert_eq!(reg.schema_version(), 1);
}

#[test]
fn from_json_future_version_warns_but_succeeds() {
    let json =
        r#"{"schema_version":999,"profiles":[],"system_io":{"interconnects":[],"storage":[]}}"#;
    // Should succeed (just warns, doesn't reject)
    let reg = AcceleratorRegistry::from_json(json).unwrap();
    assert_eq!(reg.schema_version(), 999);
}

#[test]
fn from_json_invalid_json_errors() {
    let result = AcceleratorRegistry::from_json("not json");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn estimate_memory_zero_params() {
    let mem = AcceleratorRegistry::estimate_memory(0, &QuantizationLevel::Float16);
    assert_eq!(mem, 0);
}

#[test]
fn estimate_memory_huge_params() {
    // 1 trillion params at FP32 should not overflow
    let mem = AcceleratorRegistry::estimate_memory(1_000_000_000_000, &QuantizationLevel::None);
    assert!(mem > 0);
}

#[test]
fn display_interconnect_kind_all() {
    let kinds = [
        (InterconnectKind::InfiniBand, "InfiniBand"),
        (InterconnectKind::RoCE, "RoCE"),
        (InterconnectKind::NVLink, "NVLink"),
        (InterconnectKind::NVSwitch, "NVSwitch"),
        (InterconnectKind::XgmiInfinityFabric, "XGMI"),
        (InterconnectKind::Ici, "ICI"),
    ];
    for (kind, expected) in &kinds {
        assert_eq!(kind.to_string(), *expected);
    }
}

#[test]
fn display_storage_kind_all() {
    let kinds = [
        (StorageKind::NVMe, "NVMe"),
        (StorageKind::SataSsd, "SATA SSD"),
        (StorageKind::Hdd, "HDD"),
        (StorageKind::Unknown, "Unknown"),
    ];
    for (kind, expected) in &kinds {
        assert_eq!(kind.to_string(), *expected);
    }
}
