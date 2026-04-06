//! Tests for container/VM/cloud environment detection.

use crate::*;

#[test]
fn environment_detection_runs() {
    // This just verifies the detection code doesn't panic.
    let registry = AcceleratorRegistry::detect();
    let env = registry.system_io().environment.as_ref();
    // On a dev machine, we expect at least the environment to be detected.
    assert!(env.is_some());
}

#[test]
fn environment_serde_roundtrip() {
    let env = RuntimeEnvironment {
        is_docker: true,
        is_kubernetes: false,
        kubernetes_namespace: None,
        cloud_instance: Some(CloudInstanceMeta {
            provider: "aws".into(),
            instance_type: Some("p4d.24xlarge".into()),
            instance_id: Some("i-1234567890abcdef0".into()),
            region: Some("us-east-1".into()),
            zone: None,
        }),
        kubernetes_gpu: None,
    };
    let json = serde_json::to_string(&env).unwrap();
    let back: RuntimeEnvironment = serde_json::from_str(&json).unwrap();
    assert_eq!(env, back);
}

#[test]
fn environment_serde_minimal() {
    let env = RuntimeEnvironment {
        is_docker: false,
        is_kubernetes: false,
        kubernetes_namespace: None,
        cloud_instance: None,
        kubernetes_gpu: None,
    };
    let json = serde_json::to_string(&env).unwrap();
    // Optional fields should be omitted.
    assert!(!json.contains("kubernetes_namespace"));
    assert!(!json.contains("cloud_instance"));
    let back: RuntimeEnvironment = serde_json::from_str(&json).unwrap();
    assert_eq!(env, back);
}

#[test]
fn system_io_without_environment_deserializes() {
    // Old JSON without environment field should deserialize fine.
    let json = r#"{"interconnects":[],"storage":[]}"#;
    let sio: SystemIo = serde_json::from_str(json).unwrap();
    assert!(sio.environment.is_none());
}
