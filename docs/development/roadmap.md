# Roadmap

Open work, grouped by the release it is planned for. Correct output and
platform validation come before new features. What already shipped is in
[CHANGELOG.md](../../CHANGELOG.md).

---

## 2.6.x — Platform validation, fixture-first

Implement or lock each detection path against a captured fixture first (no
hardware needed: `tests/fixtures/` or inline strings), then confirm on real
hardware when access happens. Missing hardware never blocks the source-side
work.

- [ ] **Remove the deprecated `disk_cached_*` functions** in 2.6.0.
  `disk_cached_new`, `disk_cached_get` and `disk_cached_invalidate` were
  deprecated in 2.5.1, when they stopped writing a file they never read back;
  they now behave as `cached_registry_new`, `cached_get` and
  `cached_invalidate`.

- [ ] **Fixtures for every parser.** `tests/fixtures/vulkaninfo/` holds real
  captures (Renoir on Linux, UHD 600 on Windows). The other parsers are still
  tested against inline strings in `gpu_parser_test.tcyr` and
  `backend_test.tcyr`. Move those into per-backend fixture files, so a
  contribution is "a capture of `<tool> <args>` on `<hardware>`, added to
  `tests/fixtures/`".
- [ ] **Captures from real hardware:**
  - [ ] NVIDIA H100 / A100 / GH200: `nvidia-smi` CSV (AWS p5 / GCP a3-high),
    including a real `pci.device_id`, the duplicate pass's key. GH200's
    unified memory (`mem_bytes + 480 GiB`) is coded; a fixture locks it in.
  - [ ] AMD MI300X / MI250: `/sys/class/drm/*/device/*`. The MI300X CXL path
    (`mem_info_vis_vram_total`) is coded.
  - [ ] Google TPU v5e / v5p: `/sys/class/accel/*` on a GCE v5 slice.
  - [ ] AWS Neuron trn1 / inf2: `neuron-ls --json` from trn1.32xlarge,
    covering the per-core count math.
  - [ ] Intel Gaudi 3: `hl-smi --query-aip` (AWS DL2), locking in the HL-325
    device-name override.
  - [ ] Intel Data Center GPU Max: HBM vs DDR memory tiers, and
    `xpu-smi discovery --dump 1,2,16,24` from any oneAPI GPU. The parser
    (2.5.3) follows the XPU Manager V1.x user guide's CSV format and maps
    columns by header label; it has not met real output, and V2.x was not
    checked.
  - [ ] `vulkaninfo` (`--summary` and full) from more drivers: Intel ANV on
    Linux; lavapipe alone (it must not be reported); RADV and AMDVLK installed
    together (two Vulkan devices for one AMD GPU); MoltenVK on macOS and
    Honeykrisp on Asahi Linux (the Vulkan view of the Metal GPU that the
    duplicate pass drops).
  - [ ] Asahi Linux: `/proc/device-tree/compatible`, with the Honeykrisp
    capture above.
- [ ] **Detectors no test ever runs.** `backend_test` covers these
  accelerators' profiles, ranks and names, but nothing calls their `detect_*`
  functions. Their parsing sits next to the tool and sysfs reads, so factor out
  pure parsers first, then test them against fixtures (from vendor docs until
  real captures exist):
  - [ ] Cerebras WSE (`cloud_asic.cyr`): `/dev/cerebras*` + sysfs; a sample
    from Cerebras docs or a contributor with access.
  - [ ] Graphcore IPU (`cloud_asic.cyr`): `gc-info` output, which can be
    synthesized from the public SDK documentation.
  - [ ] Groq LPU (`cloud_asic.cyr`): `/dev/groq*` sysfs. The driver is not in
    public distros, but the sysfs format is documented.
  - [ ] Qualcomm Cloud AI 100 (`edge.cyr`).
  - [ ] Samsung NPU (`edge.cyr`): it probes `/sys/class/misc/samsung_npu`.
    Confirm the node on an Exynos device (Galaxy S24+) or from Samsung's
    documentation.
  - [ ] MediaTek APU (`edge.cyr`): it probes `/sys/class/misc/mtk_apu`.
    Confirm the node on a Dimensity device or from the NeuroPilot
    documentation.
- [ ] **An AGNOS GPU detector.** `BACKEND_AGNOS_GPU` / `ACCEL_AGNOS_GPU` are
  declared (name, rank, throughput and training factor) but no detector exists,
  so the backend never reports anything.

---

## 2.7.x — Multi-node and hot-plug

### Multi-node detection

- [ ] **SSH probe** — `registry_detect_remote(hosts)`, merged into a
  cluster-wide registry.
- [ ] **Cluster-aware sharding** — distribute across nodes, weighing IB/RoCE
  bandwidth for pipeline vs data parallelism.

### Hot-plug

- [ ] **`udev` watcher (Linux)** — `registry_watch()` returns a stream of
  device-added / device-removed events.
- [ ] **Dynamic registry updates** — `CachedRegistry` invalidates itself on
  hot-plug events.

---

## 2.8.x — Fleet and scale

Fleet-wide inventory, health monitoring and capacity planning at datacenter
scale.

### Fleet inventory

- [ ] **Fleet registry** — `FleetRegistry` aggregates registries from many
  nodes and tracks hostname, IP, rack and zone.
- [ ] **Discovery** — mDNS/DNS-SD, Consul, Kubernetes node labels.
- [ ] **Inventory persistence** — SQLite or JSON lines, diffed against the
  previous scan.
- [ ] **Fleet CLI** — `ai-hwaccel fleet scan --subnet 10.0.0.0/24`.

### Health monitoring

- [ ] **Continuous health checks** — periodic re-detection and trend tracking.
- [ ] **Alert rules** — configurable thresholds; webhook / Slack / PagerDuty.
- [ ] **Dead-GPU detection** — flag nodes whose accelerators went missing.
- [ ] **ECC error tracking** — `nvidia-smi` ECC, ROCm `ras_features`.

### Capacity planning

- [ ] **Fleet-wide sharding** — recommend nodes and a distribution strategy.
- [ ] **Bin packing** — place several models across the fleet.
- [ ] **Scaling recommendations** — e.g. "405B needs 3 more H100 nodes".

### Observability and export

- [ ] **Prometheus metrics** — per-device gauges and fleet aggregates.
- [ ] **OpenTelemetry spans** around detection.
- [ ] **Grafana dashboard template** — fleet GPU-utilization heatmap.
- [ ] **Structured event log** — JSON lines for ELK / Loki / Datadog.

### Multi-tenancy

- [ ] **Device reservation** — `registry_reserve(device_id, owner)`.
- [ ] **Namespace isolation** — Kubernetes pod-scoped detection.
- [ ] **Quota management** — per-team GPU-hour budgets.

---

## Later (2.x, not scheduled)

- [ ] **Per-backend build flags (ADR-004).** The `-D<BACKEND>` flags ADR-004
  describes were never implemented: nothing in `src/` reads them, so every
  build includes every backend. Implement the `#ifdef` gates (cyrius 6.6.5 and
  later honour `-D` after the operands), or retire the ADR. Builder masks
  already select backends at run time.
- [ ] **Android** — HAL `hwbinder` for the NNAPI accelerator list.
- [ ] **FreeBSD** — DRM sysctl equivalents for GPU detection.
- [ ] **JS/TS bindings.** cyrius 6.7.18's `--target js` still cannot build
  `src/` ("ts lex error before emit"), and detection needs processes and the
  filesystem anyway. A Node package can wrap the CLI and its JSON, exactly as
  the Python package does.
- [ ] **Power budget planning** — recommend a device mix for a power cap.
- [ ] **Thermal throttling prediction** — warn as devices approach their
  limits.
- [ ] **Plugin system** — third-party backends via dynamic loading.
- [ ] **`switch` dispatch for the enum tables** (`accel_name()`,
  `family_name()`, `format_name()`, `_gguf_file_type_name()`,
  `requirement_satisfied()`). No longer blocked: cyrius 6.7.18 accepts an enum
  name as a `case` label, where 6.6.6 gave `expected number, got identifier`.
  (`case 0:` with comments was rejected as too brittle against renumbering.)

---

## Non-goals

- **Runtime execution** — detection and planning only, not inference or
  training.
- **Kernel driver management** — no installing or configuring drivers.
- **Cloud provisioning** — detect what is present, not what could be spun up.
