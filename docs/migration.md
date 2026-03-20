# Migration Guide

## v0.19.3 → v0.20.3

### New fields on `AcceleratorProfile`

Five optional fields were added to `AcceleratorProfile`:

- `memory_bandwidth_gbps: Option<f64>` — VRAM bandwidth in GB/s
- `memory_used_bytes: Option<u64>` — current VRAM usage
- `memory_free_bytes: Option<u64>` — current free VRAM
- `pcie_bandwidth_gbps: Option<f64>` — PCIe link bandwidth in GB/s
- `numa_node: Option<u32>` — NUMA node affinity

**If you construct `AcceleratorProfile` structs directly** (not via `::cuda()`,
`::rocm()`, etc.), you must add these fields. Set them to `None`:

```rust
AcceleratorProfile {
    accelerator: AcceleratorType::CudaGpu { device_id: 0 },
    available: true,
    memory_bytes: 24 * 1024 * 1024 * 1024,
    compute_capability: Some("8.6".into()),
    driver_version: None,
    // New in 0.20:
    memory_bandwidth_gbps: None,
    memory_used_bytes: None,
    memory_free_bytes: None,
    pcie_bandwidth_gbps: None,
    numa_node: None,
}
```

The convenience constructors (`AcceleratorProfile::cuda()`, etc.) already
include these fields, so code using them is unaffected.

**JSON compatibility**: Old JSON without these fields deserializes correctly —
the new fields use `#[serde(default)]` and are omitted from output when `None`.

### New `SystemIo` on `AcceleratorRegistry`

The registry now includes system-level I/O information:

```rust
let sio = registry.system_io();
// sio.interconnects — InfiniBand, RoCE, NVLink
// sio.storage — NVMe, SSD, HDD
// sio.estimate_ingestion_secs(bytes) — data loading estimate
```

**JSON compatibility**: Old JSON without `system_io` deserializes correctly
(defaults to empty interconnects and storage).

### Schema version bumped to 2

`SCHEMA_VERSION` changed from `1` to `2`. If you compare or assert on this
constant, update accordingly.

### New `DetectionError::Timeout` variant

A new error variant was added:

```rust
DetectionError::Timeout { tool: String, timeout_secs: f64 }
```

Previously, timeouts were reported as `ToolFailed`. Now they are distinct,
enabling retry logic for slow tools.

**If you match exhaustively on `DetectionError`** (without a `_` wildcard),
you'll need to add a `Timeout` arm. The enum is `#[non_exhaustive]`, so a
`_` catch-all is recommended:

```rust
match err {
    DetectionError::ToolNotFound { .. } => { /* ... */ }
    DetectionError::Timeout { .. } => { /* ... */ }
    _ => { /* handles future variants too */ }
}
```

**Note**: `DetectionError` now implements `PartialEq` but no longer `Eq`
(because `timeout_secs` is `f64`). If you relied on `Eq`, use `PartialEq`
instead.

### Subprocess environment sanitization

`run_tool()` now strips `LD_PRELOAD`, `LD_LIBRARY_PATH`,
`DYLD_INSERT_LIBRARIES`, and `DYLD_LIBRARY_PATH` from child processes.
This is a security hardening change. If a detection tool requires one of
these variables, it will no longer receive it.

### CLI changes

The `--table` output now includes additional columns:
- `Free` — current free VRAM
- `BW` — memory bandwidth
- `PCIe` — PCIe link bandwidth
- `NUMA` — NUMA node

New CLI flags:
- `--columns name,mem,bw` — select specific table columns
- `--tsv` — tab-separated output (machine-readable)
- `--alert mem>90` — alert when VRAM usage exceeds threshold (with `--watch`)
- `--watch` now shows memory usage deltas between refreshes
