# Roadmap

Open work, grouped by the release it is planned for. Correct output and
platform validation come before new features. What already shipped is in
[CHANGELOG.md](../../CHANGELOG.md).

---

## 2.4.x — Correct output, and CI that would notice

Fix what ai-hwaccel reports wrong today, and close the CI gaps that let those
bugs ship.

### Detection output

- [ ] **Vendor tools run without a time limit.** A hung tool (`nvidia-smi` on
  a GPU in an error state, for example) blocks detection until it exits. The
  stdlib waits for ever by default, and ai-hwaccel never sets a deadline.
  cyrius 6.6.6 added `proc_set_timeout_ms()`, which kills a child that stays
  idle past the deadline. It is POSIX-only (Windows and AGNOS don't define
  it), so the call needs a target gate. Set a deadline around the tool runs,
  and report a timeout with `warning_timeout`, which exists but nothing emits.
  Until then, the Python package's 30 s limit on each CLI run is the only
  bound.
- [ ] **A read that fills its buffer writes one byte past it.** Several
  readers NUL-terminate with `store8(buf + n, 0)` when `n` can equal the
  buffer size:
  - `run_tool` (1 MiB of tool output);
  - `sysfs_read` and the `/proc/meminfo` read (4 KiB);
  - `load_cloud_instances` (32 KiB);
  - the disk cache (64 KiB).

  The output is also cut off without a warning. 2.3.22 fixed the same defect
  in `load_models`: read at most `size - 1` bytes and warn when the buffer
  fills. Apply that to the others. The tool cap is the easiest to reach:
  `vulkaninfo`'s full output is 60–85 KB per GPU in `tests/fixtures/vulkaninfo/`.
- [ ] **Intel Macs report a phantom Apple Silicon GPU and Neural Engine.** On
  an Intel Mac the sysctl brand check fails, so detection falls back to
  `system_profiler SPHardwareDataType`. `_parse_apple_macos` then emits a
  Metal GPU and a Neural Engine whatever the output says (named "Apple
  Silicon", sized by the Mac's RAM). It should emit nothing without a `Chip:`
  line. The Intel Mac's own GPU is not detected at all. This comes from
  reading the code; it has not been run on `ach`, and no wheel targets Intel
  Macs.
- [ ] **No `pci_id` on oneAPI, Windows or Apple GPU profiles.** The duplicate
  pass matches on `pci_id`, which only the CUDA, ROCm and Vulkan detectors
  set. An Intel Data Center GPU that both `xpu-smi` and `vulkaninfo` report is
  therefore listed twice. DXGI profiles need one too (next items).
- [ ] **Lazy queries should run only the detectors they need.** Every lazy
  family probe goes through `registry_detect_with`, so it runs the post-passes
  too, and `detect_interconnects` spawns `nvidia-smi nvlink -s` whenever no
  NVSwitch sysfs node exists. A TPU-only query therefore still starts
  `nvidia-smi`, the exact spawn `lazy.cyr` exists to avoid, and
  `lazy_into_registry` repeats the post-passes for every family. Run the
  detectors per family and the post-passes once, in `lazy_into_registry`.
  Consider also gating the NVLink probe on a CUDA profile being present.
- [ ] **`registry_detect_with(builder_no_exec())` still spawns.** The mask
  removes the exec backends, but the call passes `allow_exec = 1`: Vulkan runs
  `vulkaninfo`, the interconnect pass runs `nvidia-smi`, and the Apple and
  Windows fallbacks may run `system_profiler` and `wmic`. The comment on
  `builder_no_exec()` and the README now say so and point to
  `registry_detect_with_opts(mask, 0)`. Decide whether a mask without exec
  backends should imply `allow_exec = 0`.
- [ ] **Warnings name tools that could never exist.** On Linux the Apple
  backend's fallback adds `system_profiler` to `warnings` on every run, and a
  missing `nvidia-smi` is listed twice (the CUDA backend and the interconnect
  pass). Run the `system_profiler` fallback on macOS only, and report each
  missing tool once.
- [ ] **Integrated GPUs on Windows report their dedicated carve-out.** An
  integrated (UMA) adapter's `DedicatedVideoMemory` is its boot-time carve-out
  (128 MiB on an Intel UHD 600), not the shared system memory it can also use
  (`SharedSystemMemory`, typically half of RAM). Reporting the shared budget
  needs an integrated-vs-discrete signal, which `DXGI_ADAPTER_DESC1` does not
  carry. `vulkaninfo` has both the signal (`deviceType`) and the size (the
  device-local heap: 4 202 799 104 bytes on `cass`, half its RAM), once tools
  run on Windows at all (next item).
- [ ] **No vendor tool ever runs on Windows.** `which()` (`src/detect/command.cyr`)
  splits `PATH` on `:` and looks for the bare name. Windows separates with `;`,
  has drive letters and needs `.exe`. So every `run_tool` backend reports
  "tool not found" there: `nvidia-smi`, `vulkaninfo` (`cass` has
  `C:\Windows\System32\vulkaninfo.exe`) and the rest. GPUs are still found,
  through DXGI, but with no CUDA details. Fixing it adds CUDA and Vulkan
  profiles next to the DXGI ones. The duplicate pass would drop them once DXGI
  profiles carry a `pci_id` (`DXGI_ADAPTER_DESC1` has `VendorId` and
  `DeviceId`) and count as the survivor for Vulkan. CUDA's profile should then
  win over DXGI's.

### Disk cache

- [ ] **The disk cache writes a file it never reads.** `disk_cached_get`
  writes `registry.json` after each detection, but its read path is a `TODO`:
  on a cold start it loads the file, discards it and detects again. Either
  parse it back with `profile_from_json` (bayan-json is optional, so fall back
  to detection without it), or drop the disk layer. While there:
  - without `HOME` the cache is `/tmp/ai-hwaccel-cache.json`, and the write
    follows symlinks, so another local user can aim it at a file the caller
    can write;
  - the directory is created one level deep, so without an existing
    `~/.cache` nothing is written;
  - on Windows, where `HOME` is usually unset, it takes the `/tmp` path.

### Build

- [ ] **Retire the `CYRIUS_DCE=1` PE workaround** (your call). This was fixed
  upstream in cyrius 6.6.1 (PE declines compaction and NOP-fills dead code) and
  verified on `cass` under 6.6.6. Restoring the flag in `stage_win_cross.sh`
  shrinks the EXE inside the wheel from 77 534 to 43 684 B compressed. See
  [the issue](issues/2026-09-07-cyrius-dce-pe-access-violation.md).

### CI coverage — run what ships, on every target

- [ ] **A `macos-smoke` job.** `wheels.yml`'s `macos` job builds the arm64
  binary but never runs it, which is how the RAM bug fixed in 2.3.27 survived.
  Mirror `windows-smoke`: JSON shape, stderr silent at the default level,
  `--version`, CPU memory against `sysctl hw.memsize`, and a Metal GPU whenever
  `system_profiler` lists one. Run `registry_detect_threaded()` there too (so
  far it has run on Apple Silicon only by hand, on `ecb`). Until then:
  cross-build with the pinned `cycc_aarch64` and `CYRIUS_MACHO_ARM=1`, then
  ad-hoc sign and run on `ecb`.
- [ ] **Make the lint step fail.** A lint warning never fails the build:
  `cyrius lint` exits 0 even when it reports warnings, and CI's loop adds
  `|| true` anyway. `cyrius lint --strict` exits 2 on a warning, and every
  file in `src/` passes it today, so switch the loop to `--strict` and drop
  the `|| true`.
- [ ] **qemu-aarch64 disk-cache test**
  ([issue](issues/archived/2026-09-07-cache-raw-syscalls-wrong-on-aarch64.md)).
- [ ] **A cache TTL test that does not assume Linux**
  ([issue](issues/archived/2026-09-07-monotonic-secs-unguarded-on-macos-windows.md)).
- [ ] **`cyrius capacity --check` as a CI gate.** On 6.6.6 it passes against
  `src/main.cyr` (597 variable slots in use, every table under 85%).
- [ ] **`cyrius vet` over `tests/tcyr/`.** CI vets only `src/main.cyr`, and
  each test unit is its own compilation root.

---

## 2.5.x — Platform validation, fixture-first

Implement or lock each detection path against a captured fixture first (no
hardware needed: `tests/fixtures/` or inline strings), then confirm on real
hardware when access happens. Missing hardware never blocks the source-side
work.

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
  - [ ] Intel Data Center GPU Max: HBM vs DDR memory tiers.
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

## 2.6.x — Multi-node and hot-plug

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

## 2.7.x — Fleet and scale

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
- [ ] **JS/TS bindings.** cyrius 6.6.6 has `--target js`, but it cannot build
  `src/` ("ts lex error before emit"), and detection needs processes and the
  filesystem anyway. A Node package can wrap the CLI and its JSON, exactly as
  the Python package does.
- [ ] **Power budget planning** — recommend a device mix for a power cap.
- [ ] **Thermal throttling prediction** — warn as devices approach their
  limits.
- [ ] **Plugin system** — third-party backends via dynamic loading.

### Blocked outside this repo

- [ ] **`switch` dispatch for the enum tables** (`accel_name()`,
  `family_name()`, `format_name()`, `_gguf_file_type_name()`,
  `requirement_satisfied()`). cyrius still rejects an enum name as a `case`
  label: 6.6.6 gives `expected number, got identifier`. `case 0:` with
  comments was rejected as too brittle against renumbering.

---

## Non-goals

- **Runtime execution** — detection and planning only, not inference or
  training.
- **Kernel driver management** — no installing or configuring drivers.
- **Cloud provisioning** — detect what is present, not what could be spun up.
