# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [semantic versioning](https://semver.org/) as of v0.19.3.

## [1.1.1] - 2026-04-03

### Fixed

- **Division-by-zero in cost headroom** — `recommend_instance` no longer
  produces NaN when an instance has `total_gpu_memory_gb == 0`.
- **Invalid layer range in pipeline sharding** — tiny models (< 250M params)
  with multiple pipeline stages could produce shards where `start > end`;
  layer ranges are now clamped to valid bounds.
- **`DetectionError::ToolFailed` Display allocation** — removed intermediate
  `String` allocation when formatting exit codes.

### Changed

- **`QuantizationLevel::bits_per_param()` and `memory_reduction_factor()` are
  now `const fn`** — enables use in const contexts.
- **Added `#[must_use]` attributes** to `plan_sharding()`, `total_memory()`,
  `has_accelerator()`, `available()`, `by_family()`, `satisfying()`, and
  `ShardingPlan::shards()`.
- **Replaced magic numbers** — `Display` impls in `ShardingPlan` and
  `AcceleratorProfile` now use `units::BYTES_PER_GIB` instead of
  `1024.0 * 1024.0 * 1024.0`.
- **`cost::all_instances()`** now logs `tracing::warn` on malformed embedded
  pricing JSON instead of silently returning empty.
- Updated doc examples to reference `version = "1.1"` (was `"0.19"`).

### Dependencies

- criterion 0.5.1 → 0.8.2 (dev-dependency, major version bump)
- criterion-plot 0.5.0 → 0.8.2
- itertools 0.10.5 → 0.13.0 (transitive)
- Removed transitive deps: is-terminal, hermit-abi

---

## [1.1.0] - 2026-04-03

### Changed

- **License: AGPL-3.0-only → GPL-3.0-only** — updated Cargo.toml, deny.toml,
  CLAUDE.md, and LICENSE file. Removes the network-use copyleft clause.
- **`available()`, `by_family()`, `satisfying()` return `impl Iterator`** —
  zero-alloc queries for callers using `.count()`, `.any()`, `.next()`.
  Callers needing a `Vec` use `.collect()` explicitly. Benchmarked: 3–35x
  faster for non-materializing callers; `.collect()` path unchanged.
- **Detection macro consolidation** — 6 local macros (`run_backend!`,
  `spawn_backend!`, `run_backend_timed!`, `spawn_backend_timed!`,
  `spawn_async_backend!`, `run_sysfs!`) replaced by 3 registration table
  macros (`backend_table!`, `async_cli_backends!`, `sysfs_backends!`) with
  local callback dispatch. Adding a new backend is now a 1-line table entry
  instead of editing 6 locations.
- **Watch mode allocations reduced** — delta tracking uses index+Display
  keys instead of Debug format, avoiding per-tick `format!("{:?}")` allocs.
- **`tracing-subscriber` slimmed** — dropped `env-filter` (regex engine,
  ~347 KB .text) and `json` (tracing-serde) features. `EnvFilter` replaced
  with simple `LevelFilter` match on `RUST_LOG`. `--json-log` flag removed.
- **Release profile optimized** — `lto = true`, `codegen-units = 1`,
  `strip = true`, `panic = "abort"`, `opt-level = "z"`. Binary size:
  2.6 MB → 838 KB (68% reduction).

### Added

- **Dual iterator/collect benchmarks** — `registry_queries` group now
  benchmarks both `_count` (lazy) and `_collect` (materialized) variants
  for `available()`, `by_family()`, and `satisfying()` to transparently
  show the allocation cost difference.

### Fixed

- Cleaned up unused license allowances in `deny.toml` (BSD-2-Clause,
  BSD-3-Clause, ISC, Unicode-DFS-2016 were not in the dependency tree).
- Pruned unnecessary `cargo-vet` exemptions.
- Certified 14 updated dependency versions for `cargo-vet`.

### Dependencies

- indexmap 2.13.0 → 2.13.1
- js-sys 0.3.91 → 0.3.94
- libc 0.2.183 → 0.2.184
- mio 1.1.1 → 1.2.0
- proptest 1.10.0 → 1.11.0
- tokio 1.50.0 → 1.51.0
- tokio-macros 2.6.1 → 2.7.0
- wasm-bindgen 0.2.114 → 0.2.117
- web-sys 0.3.91 → 0.3.94
- zerocopy 0.8.47 → 0.8.48

---

## [1.0.0] - 2026-03-27

### Breaking Changes

- **`AcceleratorType` is now `Copy`** — `device_name: String` moved from
  `VulkanGpu` variant into `AcceleratorProfile::device_name: Option<String>`.
  `VulkanGpu` is now `VulkanGpu { device_id: u32 }`. All `.clone()` calls on
  `AcceleratorType` are eliminated. Callers that matched on
  `VulkanGpu { device_name, .. }` should read `profile.device_name` instead.
- **`ShardingPlan::shards` is now `pub(crate)`** — use `plan.shards()` accessor
  instead of direct field access.
- **`cost::CloudInstance` renamed to `cost::CloudGpuInstance`** — the actual
  type is now `CloudGpuInstance`, removing the `as` alias in re-exports.
- **`system_io::CloudInstance` renamed to `system_io::CloudInstanceMeta`** —
  disambiguates from the cost pricing type.

### Added

#### API

- `TryFrom<u32>` for `QuantizationLevel` — map `32 → None`, `16 → Float16`,
  `8 → Int8`, `4 → Int4`. Returns `Err(bits)` for unsupported values.
- `AcceleratorProfile::device_name: Option<String>` — human-readable device
  name (e.g. "RTX 4090"), populated by CUDA and Vulkan detectors.
- `#[non_exhaustive]` on `ShardingStrategy`, `TrainingMethod`, `TrainingTarget`,
  `InterconnectKind`, `StorageKind`, `CloudProvider`.
- `#[must_use]` on 18 pure public methods across registry, profile,
  quantization, requirement, training, cost, sharding, and plan modules.
- `#[inline]` on 9 additional hot-path getters.
- `Backend::WindowsWmi` variant with `with_windows_wmi()` /
  `without_windows_wmi()` builder methods.

#### Platform & Validation

- **Cloud hardware validation fixtures** — realistic parser tests for
  A100 80GB 8-GPU, H100 80GB SXM, Grace Hopper GH200 (unified memory),
  Gaudi 3 8-device, Neuron trn1.32xlarge/inf2.48xlarge, MI300X 192GB.
  Planning pipeline tests for 8x A100 sharding, TPU v5p 256-chip pod,
  TPU v5e 4-chip, 8x Gaudi3, 8x MI300X.
- **macOS `system_profiler -json` GPU detection** — `parse_displays_json()`
  parses `SPDisplaysDataType -json` for GPU name, vendor, Metal family,
  core count, discrete VRAM. `parse_sysctl_output()` for CPU topology
  (memory, core count, perf/efficiency cores).
- **Windows WMI GPU detection** — new `detect/windows.rs` module behind
  `windows-wmi` feature flag. `parse_wmic_output()` for
  `Win32_VideoController` CSV, `parse_powershell_csv()` for
  `Get-CimInstance` fallback. `find_nvidia_smi_windows()` for path resolution.
- **Platform abstraction trait** — `PlatformProbe` trait in
  `detect/platform.rs` abstracting filesystem reads, command execution,
  device enumeration, and system memory. `LivePlatform` + `MockPlatform`.
- **Feature profiles**: `minimal` (CPU-only) and `common`
  (cuda+rocm+apple+vulkan+intel-npu) feature sets.

#### Testing

- **471 tests** (up from 358): cloud hardware fixtures, ASIC quantization
  coverage, macOS/Windows parser tests, platform trait tests, planning
  pipeline tests, `TryFrom<u32>` tests.

### Changed

- `AcceleratorType` derives `Copy` — zero-cost pass-by-value for all 19
  hardware variants.
- `AcceleratorProfile::Display` includes device name when present.
- CLI decomposed: `print_table()` split into `filter_profiles()`,
  `sort_profiles()`, `render_header()`, `render_row()`, `render_footer()`.
  `handle_cost_mode()` and `handle_profile_mode()` extracted.
- `Column` type gains `header()`, `width()`, `is_left_aligned()` methods.
- `parse_csv_line()` shared CSV helper for cuda/gaudi/intel_oneapi parsers.

### Fixed

- Scaffold hardening audit: all public enums now `#[non_exhaustive]`, all pure
  functions `#[must_use]`, all hot-path getters `#[inline]`.

---

## [0.23.3] - 2026-03-23

### Added

#### Benchmark infrastructure

- **Benchmark history tracking**: `scripts/bench-history.sh` captures criterion
  results to `bench-history.csv` with 7-column format (timestamp, commit,
  branch, benchmark, low_ns, estimate_ns, high_ns). Auto-generates
  `benchmarks.md` with 3-point trend tables (baseline → previous → current).
- **95 benchmarks across 16 groups**: detection, parsing, planning, training,
  cost, quantization, registry queries, caching, lazy detection, large-registry
  sharding, and JSON serialization.
- **New bench files**: `benches/training.rs`, `benches/cost.rs`,
  `benches/quantization.rs`, `benches/parsing.rs`, `benches/registry.rs`.
- `make bench` target for running the full benchmark suite.

#### Testing

- **358 tests** (up from ~280): added FFI module tests (11), async detection
  tests (5), parser fixture tests for all backends (Vulkan summary, Apple
  system_profiler, Gaudi multi-device, CUDA edge cases, Neuron JSON, Intel
  oneAPI CSV, Cerebras memory, Graphcore memory), and named-constant
  verification tests.
- **23 test modules** covering all public API surface.

#### API

- `DetectBuilder::with(Backend)` / `without(Backend)` — generic methods for
  enabling/disabling backends. Existing `with_cuda()` etc. are now inline
  wrappers.
- `ShardingPlan::shards()` accessor method.
- `Default` derive for `ShardingStrategy`, `TrainingMethod`, `TrainingTarget`.
- `Display` impl for `MemoryEstimate`.
- `Default` impl for `AcceleratorProfile` — simplifies construction with
  `..Default::default()`.

### Changed

- **`plan_sharding()` decomposed** into `InterconnectInfo::scan()`,
  `build_tpu_tensor_plan()`, `build_gpu_tensor_plan()`, `build_pipeline_plan()`
  helper functions. Main method is now a dispatcher (~40 lines vs 225).
- **`suggest_quantization()` precomputes estimates**: 4 calls instead of up to 9
  redundant `estimate_memory()` invocations.
- **`/dev` device iteration helpers**: `iter_dev_devices()` and
  `has_dev_device()` replace ~70 lines of duplicated `/dev` scanning across 8
  backends (neuron, tpu, groq, cerebras, graphcore, qualcomm, samsung, mediatek).
- **`..Default::default()` in all detector profiles**: 22 profile constructions
  across 15 files simplified, eliminating ~100 lines of explicit `None` fields.
- **Detection modules made public**: `detect::bandwidth`, `detect::interconnect`,
  `detect::pcie`, `detect::cuda`, `detect::gaudi`, `detect::vulkan` — enables
  external benchmarking and testing of parsing functions.

### Performance

- `#[inline]` on 12 hot-path methods: `bits_per_param`, `memory_reduction_factor`,
  `is_gpu/npu/tpu/ai_asic`, `family`, `throughput_multiplier`,
  `training_multiplier`, `supports_training`, `has_interconnect`.
- **Single-pass interconnect scan** in `plan_sharding()` — combined 3 iterator
  passes into 1 `for` loop with `match`.
- **Direct JSON deserialization** in `cost.rs` — eliminated intermediate
  `serde_json::Value` clone.
- **Deferred string allocation** in CUDA parser — `&str` until non-empty check.
- **Filter-before-clone** in environment detection — AWS instance fields.
- **ROCm sysfs filter-before-alloc** — trim-then-check avoids empty String alloc.
- **Disk detection deferred `to_string()`** — skip checks use `&str` reference.

### Fixed

- **Integer overflow in Graphcore parser** (`parse_memory_from_gcinfo`): fuzz
  input with huge MB/GB values caused `u64` multiply overflow. Now uses
  `saturating_mul`. Also fixed in Cerebras and Apple memory parsers.
- **Fuzz CI timeout**: reduced per-target fuzz time from 30s to 15s (11 targets),
  added `timeout-minutes: 15` job limit.
- **Clippy `len_zero`**: `registry.all_profiles().len() >= 1` replaced with
  `!is_empty()` in async detection tests.
- Removed dead `use std::path::Path` import in TPU detector.
- Removed 3 unnecessary `return;` statements in Samsung/MediaTek/Qualcomm
  detectors.

### Exports

- `units` module (named constants for hardware math).

---

## [0.21.3] - 2026-03-23

### Added

#### Detection performance

- **Lazy detection**: `LazyRegistry::new()` defers backend probing until a
  specific accelerator family is queried. Avoids spawning `nvidia-smi` when
  the caller only needs TPU info.
- **vulkaninfo timeout + caching**: 3s subprocess timeout (down from 5s).
  Results cached to `$XDG_CACHE_HOME/ai-hwaccel/vulkan.json` with 60s TTL.
  Falls back to sysfs-only detection on timeout.
- **Sysfs-only Vulkan fallback**: Detects GPUs via
  `/sys/class/drm/card*/device/{vendor,device}` with PCI ID lookup table.
  Covers NVIDIA, AMD, and Intel GPUs without spawning `vulkaninfo`.
- **Detection result disk caching**: `DiskCachedRegistry::new(ttl)` persists
  full registry to `$XDG_CACHE_HOME/ai-hwaccel/registry.json` with atomic
  writes (temp+rename) to prevent multi-process corruption.
- **Per-backend timing**: `AcceleratorRegistry::detect_with_timing()` returns
  `TimedDetection` with per-backend `Duration` map. CLI: `--profile` flag.

#### Planning

- **Topology-aware sharding**: `plan_sharding()` now prefers tensor parallel
  for NVSwitch-connected groups or high-bandwidth NVLink (>100 GB/s). Pipeline
  parallel orders stages by NUMA locality. Throughput estimates account for
  interconnect overhead.
- **Cost-aware planning**: Static pricing table in `data/cloud_pricing.json`
  (18 instances across AWS/GCP/Azure). `cost::recommend_instance()` returns
  cheapest viable cloud instance. CLI: `--cost 70B --quant bf16`.

#### Platform

- **Container/VM detection**: Detects Docker, Kubernetes, and cloud provider
  (AWS/GCE/Azure) via DMI sysfs. Exposed as `SystemIo::environment`.
  No HTTP metadata calls — purely filesystem-based.

#### Python bindings (groundwork)

- **PyO3 module scaffold**: `py/` directory with maturin build wrapping
  `detect()`, `suggest_quantization()`, `plan_sharding()`, `system_io()`,
  `estimate_training_memory()`.
- **Type stubs**: `ai_hwaccel.pyi` for IDE support.
- **Examples**: `basic_detect.py`, `sharding_plan.py`, `training_memory.py`.

### Changed

- **Schema version**: v2 → v3 (new `environment` field in `SystemIo`).
  Old v1/v2 JSON deserializes cleanly with `environment: None`.
- **Pipeline parallel throughput**: Now scales by `num_stages` with
  interconnect overhead factor (15% NVLink, 35% PCIe-only).

### Performance

- **cost.rs OnceLock**: Pricing JSON parsed once per process (was re-parsing
  on every `recommend_instance()` call).
- **CachedRegistry lock scope**: Mutex released before running `detect()` —
  concurrent readers no longer blocked during detection.
- **DMI caching**: Cloud detection reads DMI files once, shares across
  AWS/GCE/Azure detectors (was 6-7 redundant sysfs reads).
- **read_sysfs_string**: Heap path avoids `.to_vec()` double-allocation.
- **list_driver_pci_addrs**: Uses `Path::join()` and byte-level validation.
- **Atomic cache writes**: Disk cache uses temp+rename to prevent corruption.

### Exports

- `LazyRegistry`, `DiskCachedRegistry`, `TimedDetection`
- `CloudGpuInstance`, `CloudProvider`, `InstanceRecommendation` (cost module)
- `RuntimeEnvironment`, `CloudInstance` (system_io)

---

## [0.20.3] - 2026-03-19

### Added

#### System I/O and monitoring

- **VRAM bandwidth probing**: `AcceleratorProfile::memory_bandwidth_gbps`
  calculates theoretical memory throughput from clock speed and bus width.
  NVIDIA via `nvidia-smi --query-gpu=clocks.max.memory` + compute capability
  lookup; AMD via sysfs `pp_dpm_mclk` + PCI device ID lookup. Includes
  fallback tables for known GPU specs.
- **Runtime VRAM usage**: `memory_used_bytes` and `memory_free_bytes` for
  CUDA (via `nvidia-smi`) and ROCm (via sysfs).
- **PCIe link detection**: `pcie_bandwidth_gbps` reads sysfs
  `current_link_width`/`current_link_speed` for CUDA and ROCm GPUs.
- **NUMA topology**: `numa_node` maps GPUs to their NUMA node via sysfs PCI
  device info.
- **Power and thermal monitoring**: `temperature_c`, `power_watts`,
  `gpu_utilization_percent` on `AcceleratorProfile`. CUDA via `nvidia-smi`
  (`temperature.gpu`, `power.draw`, `utilization.gpu`). ROCm via sysfs hwmon
  (`temp1_input`, `power1_average`, `gpu_busy_percent`).
- **Network interconnect detection**: `SystemIo::interconnects` detects
  InfiniBand and RoCE via `/sys/class/infiniband/`, NVLink via `nvidia-smi
  nvlink -s`. Exposes bandwidth and link state.
- **Disk I/O detection**: `SystemIo::storage` probes `/sys/block/*/queue/`
  to classify NVMe, SATA SSD, and HDD with estimated bandwidth.
- **Ingestion estimation**: `SystemIo::estimate_ingestion_secs()` estimates
  data loading time given dataset size and detected storage throughput.
- **New types**: `SystemIo`, `Interconnect`, `InterconnectKind`,
  `StorageDevice`, `StorageKind` — all serializable.

#### Detection improvements

- **AMD ROCm enrichment**: clock speeds (`pp_dpm_sclk`/`pp_dpm_mclk`),
  VBIOS version, GPU temperature, power draw, and utilization from sysfs.
  CXL-attached memory detection for MI300X/MI350.
- **Vulkan full parsing**: compute queue families, queue counts, and subgroup
  sizes from full `vulkaninfo` output (not just `--summary`).
- **NVIDIA Grace Hopper**: detects GH200/GH100 from GPU name, adds 480 GB
  unified LPDDR5X to reported HBM for capacity planning.

#### New backends (untested — written from documentation)

- **Cerebras WSE**: `cerebras_cli system-info` + `/dev/cerebras*` fallback.
- **Graphcore IPU**: `gc-info` JSON parsing + `/dev/ipu*` fallback.
- **Groq LPU**: `/dev/groq*` placeholder (driver not yet public).
- **Samsung NPU**: `/sys/class/misc/samsung_npu` + `/dev/samsung_npu*`.
- **MediaTek APU**: `/sys/class/misc/mtk_apu` + `/dev/mtk_mdla*`.

#### API and CLI

- **Schema v2**: `SCHEMA_VERSION` bumped to 2, formalizing all system I/O
  fields, power/thermal fields, and the `Timeout` error variant.
- **`DetectionError::Timeout`**: new error variant for timed-out tools,
  separate from `ToolFailed`. Enables programmatic retry logic.
- **True async detection**: `detect_async()` now uses
  `tokio::process::Command` for non-blocking subprocess I/O. CLI backends
  run as concurrent tokio tasks, sysfs-only backends in a single
  `spawn_blocking`.
- **`--columns`**: select specific table columns (`--columns name,mem,bw`).
- **`--tsv`**: tab-separated output for machine-readable table data.
- **`--watch` deltas**: memory usage changes shown between refreshes.
- **`--alert`**: threshold alerts during watch mode (`--alert mem>90`).
- **CLI table**: now shows Free VRAM, BW, PCIe, NUMA, plus Interconnects
  and Storage sections.

#### Testing

- **Hardware integration tests**: `tests/hardware_integration.rs` with 17
  tests covering CPU, ROCm, Vulkan, PCIe, bandwidth, storage, interconnects,
  JSON roundtrip, and concurrent detection. Auto-skips when hardware absent.
- **Fuzz testing**: 9 `cargo-fuzz` targets covering all CLI output parsers.
  Found and fixed integer overflow in CUDA memory parser.
- **Load testing**: concurrent 4-thread detection test + benchmark.
- **System I/O benchmarks**: per-backend, serialization, deserialization,
  and query benchmarks in `benches/detect.rs`.

#### Documentation

- **Troubleshooting guide**: `docs/troubleshooting.md`.
- **Performance tuning guide**: `docs/performance.md`.
- **Migration guide**: `docs/migration.md` (v0.19.3 → v0.20.3).
- **Crate-level docs**: expanded with error handling, custom backends, serde
  integration, and system I/O examples.

### Security

- **Subprocess environment sanitization**: `run_tool()` strips `LD_PRELOAD`,
  `LD_LIBRARY_PATH`, `DYLD_INSERT_LIBRARIES`, `DYLD_LIBRARY_PATH` from child
  processes to prevent library injection.
- **Windows `which()` improvements**: tries `.exe`, `.cmd`, `.bat` extensions
  when the tool name has no extension.
- **TOCTOU documentation**: the inherent time-of-check-time-of-use gap
  between path resolution and execution is documented as an accepted risk.

### Fixed

- **Integer overflow in CUDA parser**: `memory.used`/`memory.free` values
  exceeding u64 range on multiply now use `saturating_mul` with range filter.
  Found via fuzz testing.
- **Unbounded CSV field parsing**: CUDA parser now caps CSV splits to 20
  fields to prevent memory exhaustion from malicious `nvidia-smi` output.
- **Path traversal in PCI address handling**: PCI addresses in `pcie.rs`
  and `numa.rs` are now validated (hex+colon+dot only) and paths are
  canonicalized to prevent symlink-based information disclosure.
- **Grace Hopper memory validation**: unified memory is only added when
  reported HBM is in the realistic 80–100 GB range (prevents miscalculation
  from malformed nvidia-smi output).
- **Silent device ID fallback**: TPU and Neuron `/dev` parsers now skip
  malformed device names instead of silently mapping them to device 0.
- **Unbounded Vulkan device name**: `vulkaninfo` device names capped at
  256 characters to prevent memory exhaustion.
- **Defensive CSV bounds**: CUDA parser uses `.get()` for all field access
  instead of direct indexing.
- **Neuron JSON defaults removed**: malformed `neuron-ls` JSON devices are
  now skipped instead of using fabricated defaults (2 cores, 8192 MB).
- **Sysfs read size cap**: all sysfs reads across the codebase now use
  `read_sysfs_string()` with byte limits (64 B for values, 256 B for
  strings, 4 KiB for multi-line files, 64 KiB for /proc/meminfo).
  Handles sysfs pseudo-files correctly (they report 4096 as size regardless
  of content). Applied to: ROCm, TPU, PCIe, NUMA, interconnect, disk,
  bandwidth, neuron, and apple detectors.
- **Subprocess zombie prevention**: `child.kill()` in timeout handler now
  polls `try_wait()` for up to 100ms instead of blocking `wait()` to avoid
  hanging on zombie processes.
- **Cache lock poisoning**: `CachedRegistry` now invalidates cached state
  when the mutex is poisoned instead of continuing with potentially corrupt
  data.
- **Shard memory truncation**: `plan.rs` pipeline sharding now uses
  `div_ceil` instead of truncating division, preventing unallocated bytes.
- **Gaudi/oneAPI CSV caps**: both parsers now use `.take(20)` field limit
  matching CUDA, preventing DoS from malicious CLI output.
- **Intel oneAPI device ID validation**: uses `validate_device_id()` instead
  of `unwrap_or(0)`.
- **Neuron JSON array bounded**: capped at 256 devices to prevent DoS from
  crafted `neuron-ls` output. Device index truncation eliminated.
- **Schema version validation**: new `AcceleratorRegistry::from_json()` warns
  when deserializing registries with newer schema versions.

### Performance

- **Batched nvidia-smi**: CUDA detection and bandwidth probing merged into
  a single subprocess call, eliminating one nvidia-smi invocation per
  detection cycle (~5-10ms saved on NVIDIA systems).
- **Shared PCI address lists**: PCIe and NUMA enrichment now share a single
  `list_driver_pci_addrs()` computation instead of scanning sysfs twice.
- **Single-pass plan_sharding**: TPU and GPU device collection fused into
  one iteration over profiles instead of two separate filter passes.
- **Cached sort keys**: `--table` sort uses `sort_by_cached_key` for O(n)
  string allocations instead of O(n log n).
- **Stack buffer for sysfs reads**: `read_sysfs_string()` uses a 512-byte
  stack buffer for common small reads, avoiding heap allocation.
- **Pre-allocated collections**: profile collection uses `with_capacity(8)`,
  plan_sharding device vectors use `with_capacity(8/16)`.
- **`tracing-subscriber` made optional**: moved behind `cli` feature flag.
  Library users no longer pull 23 transitive crates.
- **`#[inline]` on hot-path queries**: `available()`, `total_memory()`,
  `has_accelerator()`, `by_family()`.
- **Vulkan output cap**: full `vulkaninfo` output parsing capped at 256 KiB.
- **Apple field cap**: `system_profiler` field values capped at 256 chars.

### CI

- **Cross-platform test matrix**: Ubuntu, macOS, and Windows runners for
  unit, integration, and doc tests.
- **Benchmark regression tracking**: `github-action-benchmark` on main with
  120% alert threshold.
- **Fuzz CI**: all 9 fuzz targets run for 30s each on every push/PR.
- **Minimal feature testing**: `--no-default-features` and single-backend
  builds verified in CI.
- **Cross-platform release builds**: Linux AMD64/ARM64, Windows AMD64, macOS
  ARM64 binaries built and published on tag.

## [0.19.3] - 2026-03-19

### Performance

- **Detection 3.5x faster**: eliminated per-subprocess reader threads in the
  command runner. Pipes are now read after the child exits (no deadlock risk
  since output is capped at 1 MiB). Poll interval reduced from 50ms to 10ms.
- **Single-pass `suggest_quantization`**: replaced 5 separate profile scans
  (`best_memory_for` per family) with one loop collecting all family maxima.
  Reduces O(5n) → O(n) on the profile list.
- **Sequential path for ≤1 backend**: `DetectBuilder::none().with_cuda()`
  skips `std::thread::scope` entirely, avoiding thread spawn/join overhead
  for selective single-backend detection.
- **CachedRegistry zero-copy**: `get()` now returns `Arc<AcceleratorRegistry>`
  instead of cloning the entire profile list on every call.
- **Reduced allocations**: `/proc/meminfo` parsing uses `nth()` iterator
  instead of collecting into a `Vec`. `read_limited` pre-allocates with
  `Vec::with_capacity`. `String::from_utf8_lossy().into_owned()` avoids
  double allocation. `#[inline]` on hot accessors (`all_profiles`,
  `warnings`, `estimate_memory`).

### Added

- **Async detection**: `AcceleratorRegistry::detect_async()` and
  `DetectBuilder::detect_async()` behind the `async-detect` cargo feature.
  Uses `tokio::task::spawn_blocking` to avoid blocking the async runtime.
- **CLI `--watch <secs>` mode**: re-detects on interval with screen clear and
  device-count change notifications.
- **CLI `--sort` flag**: sort `--table` output by `mem`, `name`, or `family`.
- **CLI `--family` flag**: filter `--table` output to a specific family
  (`gpu`, `tpu`, `npu`, `asic`, `cpu`).
- **C FFI** (`src/ffi.rs` + `include/ai_hwaccel.h`): `extern "C"` API with
  `ai_hwaccel_detect()`, `ai_hwaccel_device_count()`,
  `ai_hwaccel_has_accelerator()`, `ai_hwaccel_accelerator_memory()`,
  `ai_hwaccel_json()` and corresponding free functions.
- **Framework integration guide**: `docs/guides/framework-integration.md` with
  code examples for `candle`, `burn`, `tch-rs`, `ort`, multi-device sharding,
  and training memory budgeting.
- `tokio` optional dependency (behind `async-detect` feature).
- **Feature flags**: each of the 11 hardware backends is gated behind a cargo
  feature (`cuda`, `rocm`, `apple`, `vulkan`, `intel-npu`, `amd-xdna`, `tpu`,
  `gaudi`, `aws-neuron`, `intel-oneapi`, `qualcomm`). All enabled by default
  via `all-backends`. Disabled backends are not compiled.
- **CLI `--table` / `-t` flag**: human-readable tabular device listing with
  device ID, name, memory, family, and status columns.
- **CLI `--debug` / `-d` flag**: sets `RUST_LOG=debug` for verbose detection
  diagnostics without manually setting the environment variable.
- **Serde schema version**: `AcceleratorRegistry` now serializes a
  `schema_version` field (currently `1`) for forward-compatibility. Accessible
  via `registry.schema_version()` and `SCHEMA_VERSION` constant.
- **Property-based tests**: `proptest` fuzzing for `estimate_memory`,
  `plan_sharding`, `suggest_quantization`, and `estimate_training_memory`
  across random parameter counts and device configurations.
- **Architecture decision records**: `docs/decisions/` with 4 ADRs:
  sysfs-over-vendor-SDKs, calendar versioning, parallel detection, and
  feature flags per backend.
- **Crate-level guide**: expanded `lib.rs` documentation with a 4-step
  walkthrough (detect → query → plan → train) and cargo feature reference
  table.
- **JSON schema**: `docs/schema.json` documenting the serialized registry
  format (JSON Schema draft 2020-12).
- **`CachedRegistry`**: thread-safe detection cache with configurable TTL.
  Avoids redundant CLI tool invocations on repeated `detect()` calls.
- **Mock detection tests**: `tests/mock_detection.rs` with 11 tests using
  `tempfile` to build fake sysfs trees for hardware-independent backend
  testing, plus serde `deny_unknown_fields` rejection tests and schema
  version validation.
- **Windows CI**: added `x86_64-pc-windows-msvc` to the CI test matrix.
- `proptest` and `tempfile` dev-dependencies.
- Test suite expanded to 173 tests (140 unit + 9 integration + 11 mock +
  13 doc-tests).
- **Modular architecture**: refactored 3 monolithic source files into 23
  focused modules with single responsibilities.
  - `types.rs` (714 lines) split into `hardware/` (with `tpu.rs`, `gaudi.rs`,
    `neuron.rs`), `profile.rs`, `quantization.rs`, `requirement.rs`,
    `sharding.rs`, and `training.rs`.
  - `detect.rs` (693 lines) split into `registry.rs` (struct + query methods)
    and `detect/` module with one file per hardware backend.
  - `plan.rs` split into `plan.rs` (sharding logic) and `training.rs`
    (training types and memory estimation).
  - `tests.rs` (849 lines) split into `tests/` module with 10 files by concern.
- **`DetectionError` type** (`src/error.rs`): non-fatal detection errors
  captured as structured warnings (`ToolNotFound`, `ToolFailed`, `ParseError`,
  `SysfsReadError`) and accessible via `AcceleratorRegistry::warnings()`.
- **`DetectBuilder`**: selective backend detection via builder pattern —
  `AcceleratorRegistry::builder().with_cuda().without_vulkan().detect()`.
  Includes `Backend` enum with `ALL` constant.
- **`#[non_exhaustive]`** on `AcceleratorType`, `AcceleratorFamily`,
  `QuantizationLevel`, `AcceleratorRequirement`, and `DetectionError` for
  semver-safe enum extension.
- **Convenience constructors** on `AcceleratorProfile`: `cuda()`, `rocm()`,
  `tpu()`, `gaudi()`, `cpu()` for test and manual-config ergonomics.
- **`Display` for `ShardingPlan`**: human-readable multi-line plan summary
  showing strategy, memory, throughput, and per-shard device assignments.
- **CLI `--pretty` / `-p` flag**: pretty-printed JSON output.
- **CLI warnings**: detection warnings appear in `--summary` JSON output and
  are logged at `warn` level.
- **Structured logging**: CLI binary uses `tracing-subscriber` with `RUST_LOG`
  environment variable support and `--json-log` flag for structured JSON
  output to stderr.
- **Parallel detection**: all backends run concurrently via
  `std::thread::scope`, reducing wall-clock latency on multi-tool systems.
  Vulkan deduplication moved to a post-pass.
- **Safe command runner** (`detect/command.rs`): all CLI-based detectors use
  `run_tool()` which enforces:
  - Absolute path resolution via `which()` to prevent `$PATH` hijacking.
  - 5-second timeout with `child.kill()` on expiry.
  - Output size limits: stdout capped at 1 MiB, stderr at 4 KiB.
- **Input validation**: `validate_device_id()` (0--1024) and
  `validate_memory_mb()` (0--16 TiB) reject out-of-range parsed values from
  CLI tool output.
- **`#[serde(deny_unknown_fields)]`** on `AcceleratorRegistry`,
  `AcceleratorProfile`, `ModelShard`, `ShardingPlan` to reject unexpected
  JSON fields during deserialization.
- **`deny.toml`**: `cargo-deny` configuration for license allowlist, advisory
  checks, and crate source restrictions. New `make deny` target.
- **Threat model**: `docs/development/threat-model.md` documenting attack
  surface, trust assumptions, and mitigations.
- **Integration tests**: `tests/integration.rs` with 9 end-to-end tests
  covering the detect-query-plan pipeline, builder, JSON roundtrip, manual
  registry, training estimation, Display impls, and warnings.
- **Benchmark suite**: `criterion` benchmarks in `benches/` for `detect()`,
  `plan_sharding()`, `suggest_quantization()`, `estimate_memory()`, and
  `estimate_training_memory()`.
- **`examples/` directory**: four runnable examples — `detect.rs`, `plan.rs`,
  `training.rs`, `json_output.rs`.
- **Rustdoc examples**: `# Examples` sections on `AcceleratorRegistry`,
  `AcceleratorProfile`, `QuantizationLevel`, `DetectionError`, and
  `estimate_training_memory()`. All compile as doc-tests.
- **CI improvements**: cross-platform matrix (Linux + macOS), MSRV job
  (Rust 1.89), coverage via `cargo-llvm-cov` + Codecov, `cargo-deny`
  supply-chain checks.
- `tracing-subscriber` dependency (with `env-filter` and `json` features).
- `criterion` dev-dependency for benchmarks.
- `docs/development/roadmap.md` documenting the path to v1.0.
- Test suite expanded from 46 to 149 tests (133 unit + 9 integration +
  7 doc-tests).

### Changed

- **Switched from CalVer to SemVer**: version is now `0.19.3` (pre-1.0). The
  `0.x` series may contain breaking changes between minor versions.
- **NVIDIA detection**: now parses `driver_version` from `nvidia-smi` and
  reports structured `DetectionError` on tool failure or parse errors.
- **Vulkan detection**: parses `vulkaninfo --summary` for real device names,
  memory heap sizes, API version, and driver version instead of registering a
  generic placeholder device.
- **Apple detection**: macOS support via `system_profiler SPHardwareDataType`
  for chip name and unified memory size. ANE memory estimate varies by chip
  generation (M1: 4 GB, M2: 6 GB, M3/M4: 8 GB). Linux Asahi detection
  preserved as fallback.
- **CPU memory detection**: macOS fallback via `sysctl hw.memsize` when
  `/proc/meminfo` is unavailable.
- All detection backends now report structured warnings and use the safe
  command runner for CLI tool invocations.

### Fixed

- **`suggest_quantization` semantic bugs**: no longer returns BF16 for
  Qualcomm/Neuron AI ASICs (only Gaudi). Falls back through FP16→INT8→INT4
  on CPU instead of unconditionally returning FP16 for models that don't fit.
- **CPU memory detection**: macOS `sysctl` fallback now uses the safe command
  runner (`run_tool`) with absolute path resolution and timeout, matching all
  other CLI tool invocations.
- **Integer overflow safety**: all memory multiplications (TPU HBM × chip
  count, Neuron cores × memory, KB→bytes) use `saturating_mul` to prevent
  panics on extreme values.
- **Pipeline parallel layer assignment**: last shard now captures all remaining
  layers instead of potentially leaving a gap when layer count doesn't divide
  evenly across devices.
- **Per-chip memory precision**: TPU tensor-parallel uses ceiling division
  (`div_ceil`) so no bytes are lost to rounding.
- **Cache mutex poisoning**: `CachedRegistry` recovers from poisoned locks
  instead of panicking.
- **UTF-8 safe truncation**: CLI table column truncation uses `chars().take()`
  instead of byte slicing.
- **Windows compatibility**: mock detection tests gate Unix symlinks behind
  `#[cfg(unix)]` so the test file compiles on Windows.
- **Gaudi detection**: malformed CSV lines now produce `ParseError` warnings
  instead of being silently skipped. Device IDs and memory values are
  validated with `validate_device_id` / `validate_memory_mb`.
- `Cargo.toml` license field corrected from `AGPL-3.0` to the SPDX-correct
  `AGPL-3.0-only`.
- Added missing `homepage` and `readme` fields to `Cargo.toml` for crates.io
  compliance.

## [2026.3.19] - 2026-03-19 (CalVer, pre-SemVer switch)

### Added

- Initial public release.
- Hardware detection for 13 accelerator families: NVIDIA CUDA, AMD ROCm,
  Apple Metal, Apple ANE, Intel NPU, AMD XDNA, Google TPU (v4/v5e/v5p),
  Intel Gaudi (2/3), AWS Inferentia, AWS Trainium, Qualcomm Cloud AI 100,
  Vulkan Compute, and CPU fallback.
- `AcceleratorRegistry` with `detect()`, querying, and planning APIs.
- Quantization-aware memory estimation (`FP32`, `FP16`, `BF16`, `INT8`, `INT4`).
- Model sharding planner with tensor-parallel, pipeline-parallel, and
  data-parallel strategies.
- Training memory estimator for full fine-tune, LoRA, QLoRA, DPO, RLHF, and
  distillation methods.
- Serde support for all public types.
- CLI binary with `--summary` and `--version` flags.
- CI pipeline (format, clippy, tests, cargo-audit).
- Release automation with version consistency checks.
- Project documentation: `README.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `CHANGELOG.md`.
- `LICENSE` (AGPL-3.0-only).
- `Makefile` for local development (`check`, `fmt`, `clippy`, `test`, `build`,
  `doc`, `clean`).
- `scripts/version-bump.sh` for calendar versioning.
