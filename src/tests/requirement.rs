//! AcceleratorRequirement tests.

use crate::*;

#[test]
fn requirement_satisfied_by() {
    let cuda_profile = AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: true,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    };
    let tpu_profile = AcceleratorProfile {
        accelerator: AcceleratorType::Tpu {
            device_id: 0,
            chip_count: 4,
            version: TpuVersion::V5p,
        },
        available: true,
        memory_bytes: 95 * 4 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    };
    let cpu_profile = AcceleratorProfile {
        accelerator: AcceleratorType::Cpu,
        available: true,
        memory_bytes: 16 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    };

    // None satisfied by anything
    assert!(AcceleratorRequirement::None.satisfied_by(&cuda_profile));
    assert!(AcceleratorRequirement::None.satisfied_by(&cpu_profile));

    // GPU requirement
    assert!(AcceleratorRequirement::Gpu.satisfied_by(&cuda_profile));
    assert!(!AcceleratorRequirement::Gpu.satisfied_by(&tpu_profile));
    assert!(!AcceleratorRequirement::Gpu.satisfied_by(&cpu_profile));

    // TPU requirement
    assert!(AcceleratorRequirement::Tpu { min_chips: 2 }.satisfied_by(&tpu_profile));
    assert!(!AcceleratorRequirement::Tpu { min_chips: 8 }.satisfied_by(&tpu_profile));
    assert!(!AcceleratorRequirement::Tpu { min_chips: 1 }.satisfied_by(&cuda_profile));

    // GpuOrTpu
    assert!(AcceleratorRequirement::GpuOrTpu.satisfied_by(&cuda_profile));
    assert!(AcceleratorRequirement::GpuOrTpu.satisfied_by(&tpu_profile));
    assert!(!AcceleratorRequirement::GpuOrTpu.satisfied_by(&cpu_profile));

    // AnyAccelerator
    assert!(AcceleratorRequirement::AnyAccelerator.satisfied_by(&cuda_profile));
    assert!(AcceleratorRequirement::AnyAccelerator.satisfied_by(&tpu_profile));
    assert!(!AcceleratorRequirement::AnyAccelerator.satisfied_by(&cpu_profile));
}

#[test]
fn requirement_unavailable_device_never_satisfies() {
    let unavailable = AcceleratorProfile {
        accelerator: AcceleratorType::CudaGpu { device_id: 0 },
        available: false,
        memory_bytes: 24 * 1024 * 1024 * 1024,
        compute_capability: None,
        driver_version: None,
    };
    assert!(!AcceleratorRequirement::Gpu.satisfied_by(&unavailable));
    assert!(!AcceleratorRequirement::AnyAccelerator.satisfied_by(&unavailable));
}
