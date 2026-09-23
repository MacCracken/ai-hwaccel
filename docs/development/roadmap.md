# Roadmap

Open work, grouped by the release it is planned for. What shipped is
summarized in one table below; the details are in
[CHANGELOG.md](../../CHANGELOG.md).

The numbering inherited the Rust roadmap. The Cyrius port shipped as 2.0.0,
so the Rust plan's "1.3 / 1.4 / 1.5 / 2.0" became 2.2 / 2.3 / 2.4 / 2.5 here.
The 2026-09-23 review moved Multi-node to 2.6 and Fleet to 2.7, so that
correctness and platform validation come first.

---

## Shipped

| Releases | Dates | What shipped |
|---|---|---|
| 2.0.0 | 2026-04-13 | Rewrite from Rust to Cyrius |
| 2.0.1 | 2026-05-10 | Toolchain modernization (cyrius 3.10 → 5.10) |
| 2.1.0 – 2.1.7 | 2026-05-10 → 05-11 | cc5 adoption arc: test reorg, CI tightening, Rust-parity check, `#derive(accessors)` on every heap struct |
| 2.2.0 – 2.2.3 | 2026-05-11 → 05-19 | Test-rename fix, cyrius 5.11.8, Windows backend skeleton, `cycc` rename |
| 2.2.4 – 2.2.6 | 2026-05-19 | Library consumers: `[lib]` + `cyrius distlib` bundle, the no-exec detection contract, mihi follow-ups |
| 2.3.0 | 2026-06-01 | cyrius 6.0.25, JSON serializer hot path, dedup audit |
| 2.3.1 – 2.3.4 | 2026-06-01 | JSON schema v4, Python bindings, cwd-independent data files, Linux wheels |
| 2.3.5 – 2.3.7 | 2026-06-01 → 06-03 | cyrius 6.0.30, macOS arm64 wheel, Windows wheel |
| 2.3.8 – 2.3.12 | 2026-06-05 → 06-15 | Windows DXGI VRAM + structured logging, `windows-smoke` CI gate, `--data-dir`; cyrius 6.0.70 → 6.2.11 |
| 2.3.13 – 2.3.20 | 2026-07-13 → 08-30 | cyrius 6.4.62 → 6.5.x, symbol namespacing (`HWA_ERR_*`, `hw_registry_new`, kavach clashes), profile JSON round-trip, focused bayan dep |
| 2.3.21 – 2.3.24 | 2026-09-07 → 09-22 | cyrius 6.6.x + bayan 1.5.x, the five defects cyrius 6.6.0 surfaced, issue-folder triage |
| 2.3.25 – 2.3.29 | 2026-09-23 | Windows detection without wmic, Windows GPUs from every detection entry point, lazy NPU queries find the Apple Neural Engine, macOS real RAM and Apple Silicon via sysctl (no `system_profiler`; found in no-exec mode), unified memory counted once in totals (schema v6), integrated GPUs through Vulkan sized and shared, no lavapipe, Vulkan in no-exec mode |

---

## 2.4.x — Correct output, and CI that would notice

Fix what ai-hwaccel reports wrong today, and close the CI gaps that let those
bugs ship. The first item changes output (profile counts and memory totals),
which is why the series starts at 2.4.0.

### Detection output

- [ ] **One physical device, one profile.** Several backends can report the
  same GPU. On the Linux dev host, a single AMD Cezanne iGPU (one `lspci` VGA
  entry) comes back twice, as a ROCm GPU and as a Vulkan GPU, 8 GiB each since
  2.3.29 (the Vulkan profile had a 4 GiB estimate before). So `gpu_count` is 2
  and `accelerator_memory_bytes` counts 16 GiB. CUDA + Vulkan on Linux and
  CUDA + DXGI on Windows likely overlap the same way (not verified: no such
  host). The Rust releases dropped every Vulkan GPU whenever any CUDA or ROCm
  GPU was found, which also dropped a separate Intel iGPU. The Cyrius port lost
  even that, and `docs/troubleshooting.md` still promised it until 2.3.29. This
  needs a stable device key, a rule for which backend's profile survives, and
  a rule for which memory figure it keeps. The key can be the PCI bus address:
  full `vulkaninfo` output carries it (`VkPhysicalDevicePCIBusInfoPropertiesEXT`),
  as sysfs does; on Windows it is vendor:device + LUID. Shared-memory profiles
  already count once in the totals (2.3.28, 2.3.29), so only dedicated memory
  is double counted.
- [x] **Unified memory is counted twice in totals.** On Apple Silicon the CPU
  and the Metal GPU profiles describe the same RAM, and `total_memory_bytes`
  summed every profile: on `ecb` (48 GB) 2.3.27 reported 100 GiB. **Fixed in
  2.3.28:** profiles carry `shared_memory_bytes` (schema v6), and totals count
  system RAM once. This covers Apple Silicon, the client NPUs and GH200.
- [x] **Integrated GPUs seen through Vulkan count as memory of their own.**
  Every Vulkan GPU carried a 4 GiB estimate, so an Intel iGPU's system RAM
  counted twice. **Fixed in 2.3.29:** `vulkaninfo`'s `deviceType` and `vendorID`
  are parsed. An integrated GPU other than AMD is all shared, sized from its
  largest device-local heap in the full output. An AMD APU is sized from its
  sysfs carve-out. Software implementations (lavapipe) are no longer reported
  as GPUs, and the sysfs scan runs in no-exec mode.
- [ ] **Lazy queries should run only the detectors they need.** Every lazy
  family probe goes through `registry_detect_with`, so it runs the post-passes
  too, and `detect_interconnects` spawns `nvidia-smi nvlink -s` whenever no
  NVSwitch sysfs node exists. A TPU-only query therefore still starts
  `nvidia-smi`, the exact spawn `lazy.cyr` exists to avoid, and
  `lazy_into_registry` repeats the post-passes for every family. Run the
  detectors per family and the post-passes once, in `lazy_into_registry`.
  Consider also gating the NVLink probe on a CUDA profile being present.
- [ ] **Make the no-exec contract match its documentation.** The comment on
  `builder_no_exec()` offers `registry_detect_with(builder_no_exec())` as a
  spawn-free path, but that path passes `allow_exec = 1`: `detect_interconnects`
  still spawns `nvidia-smi`, and the Windows `wmic` fallback may run. Only
  `registry_detect_no_exec()` is spawn-free. Fix the comment (and any consumer
  guidance), or route the mask variant through `allow_exec = 0`.
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
  profiles next to the DXGI ones, so it waits on the duplicate-device item.

### Build and docs

- [ ] **`docs/schema.json` describes the Rust-era v1 output.** It pins
  `schema_version` to 1 and a tagged-union `accelerator`, with
  `additionalProperties: false`, so no current output (v6) validates against
  it. Regenerate it from `src/json_out.cyr`'s actual shape, or drop it.
- [ ] **ADR-004's `-D<BACKEND>` recipes build every backend.** The recipes are
  in `docs/guides/production.md:16`, `docs/troubleshooting.md:31`,
  `docs/performance.md:57`, `docs/guides/testing.md:38,41` and
  `docs/decisions/004-feature-flags-per-backend.md:23,26`. cyrius 6.6.5 honours
  `-D` after the operands, but nothing in `src/` reads `CUDA` / `ROCM` / `TPU` /
  `NO_BACKENDS` (its only `#ifdef`s are `CYRIUS_TARGET_*`). Implement the gates,
  or correct the docs and ADR-004's status.
- [ ] **Retire the `CYRIUS_DCE=1` PE workaround** (your call). This was fixed
  upstream in cyrius 6.6.1 (PE declines compaction and NOP-fills dead code) and
  verified on `cass` under 6.6.6. Restoring the flag in `stage_win_cross.sh`
  shrinks the EXE inside the wheel from 77 534 to 43 684 B compressed. See
  [the issue](issues/2026-09-07-cyrius-dce-pe-access-violation.md).
- [ ] **Archive the five resolved issue files** (`load_models`, resolved in
  2.3.22, and the four 2026-09-07 portability defects, resolved in 2.3.21) by
  moving them to `issues/archived/`. Their residuals are tracked in this file
  now. The PE DCE issue stays open until the flag is restored.

### CI coverage — run what ships, on every target

- [ ] **A `macos-smoke` job.** `wheels.yml`'s `macos` job builds the arm64
  binary but never runs it, which is how the RAM bug fixed in 2.3.27 survived.
  Mirror `windows-smoke`: JSON shape, stderr silent at the default level,
  `--version`, CPU memory against `sysctl hw.memsize`, and a Metal GPU whenever
  `system_profiler` lists one. Until then, the 2.3.26 path works: cross-build
  with the pinned `cycc_aarch64` and `CYRIUS_MACHO_ARM=1`, then ad-hoc sign and
  run on `ecb`.
- [x] **Threaded detection on real Apple Silicon**
  ([issue](issues/2026-09-07-threaded-detect-vs-single-threaded-sakshi.md)):
  ran on `ecb` in 2.3.27 (cross-built): 3 profiles, and the registry
  serializes. A CI run waits on `macos-smoke`.
- [x] **`AI_HWACCEL_DATA_DIR` on macOS**
  ([issue](issues/2026-09-07-cmd-getenv-proc-only.md), step 3): resolves
  `VERSION` from `/` on `ecb` in 2.3.27, as `--data-dir` does. Windows was
  already verified on `cass`.
- [ ] **qemu-aarch64 disk-cache test**
  ([issue](issues/2026-09-07-cache-raw-syscalls-wrong-on-aarch64.md)).
- [ ] **A cache TTL test that does not assume Linux**
  ([issue](issues/2026-09-07-monotonic-secs-unguarded-on-macos-windows.md)).
- [ ] **`cyrius capacity --check` as a CI gate.** It is no longer blocked: on
  6.6.6 it passes against `src/main.cyr` (597 variable slots in use, every
  table under 85%).
- [ ] **`cyrius vet` over `tests/tcyr/`.** CI vets only `src/main.cyr`, and
  each test unit is its own compilation root.

---

## 2.5.x — Platform validation, fixture-first

Implement or lock each detection path against a captured fixture first (no
hardware needed: `tests/fixtures/` or inline strings), then confirm on real
hardware when access happens. Missing hardware never blocks the source-side
work.

- [ ] **`tests/fixtures/`.** Move the inline tool outputs in
  `gpu_parser_test.tcyr` / `backend_test.tcyr` into per-backend fixture files.
  This sets the contribution pattern: "a capture of `<tool> <args>` on
  `<hardware>`, added to `fixtures/`".
- [ ] **Captures from real hardware:**
  - [ ] NVIDIA H100 / A100 / GH200: `nvidia-smi` CSV (AWS p5 / GCP a3-high).
    GH200's unified memory (`mem_bytes + 480 GiB`) is coded; a fixture locks
    it in.
  - [ ] AMD MI300X / MI250: `/sys/class/drm/*/device/*`. The MI300X CXL path
    (`mem_info_vis_vram_total`) is coded.
  - [ ] Google TPU v5e / v5p: `/sys/class/accel/*` on a GCE v5 slice.
  - [ ] AWS Neuron trn1 / inf2: `neuron-ls --json` from trn1.32xlarge,
    covering the per-core count math.
  - [ ] Intel Gaudi 3: `hl-smi --query-aip` (AWS DL2), locking in the HL-325
    device-name override.
  - [ ] Intel Data Center GPU Max: HBM vs DDR memory tiers.
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
  - [ ] Samsung NPU (`edge.cyr`): `/sys/class/npu` on Exynos (Galaxy S24+);
    a Samsung dev-portal capture.
  - [ ] MediaTek APU (`edge.cyr`): `/sys/class/misc/apusys` on Dimensity;
    NeuroPilot docs.
- [ ] **An AGNOS GPU detector.** `BACKEND_AGNOS_GPU` / `ACCEL_AGNOS_GPU` are
  declared (name, rank, throughput and training factor) but no detector exists,
  so the backend never reports anything.

---

## 2.6.x — Multi-node and hot-plug

*(2.4.0 before the 2026-09-23 review; 1.5.0 in the Rust roadmap)*

### Multi-node detection

- [ ] **SSH probe** — `registry_detect_remote(hosts)`, merged into a
  cluster-wide registry.
- [ ] **Cluster-aware sharding** — distribute across nodes, weighing IB/RoCE
  bandwidth for pipeline vs data parallelism.

### Hot-plug

- [ ] **`udev` watcher (Linux)** — `registry_watch()` returns a stream of
  device-added / device-removed events. cycc's `defer` and `lib/thread.cyr`
  make this simpler than the Rust plan assumed.
- [ ] **Dynamic registry updates** — `CachedRegistry` invalidates itself on
  hot-plug events.

---

## 2.7.x — Fleet and scale

*(2.5.0 before the 2026-09-23 review; 2.0.0 in the Rust roadmap)*

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

- [ ] **Android** — HAL `hwbinder` for the NNAPI accelerator list.
- [ ] **FreeBSD** — DRM sysctl equivalents for GPU detection.
- [ ] **JS/TS bindings, re-scoped.** The old plan waited for a cyrius WASM
  target. cyrius 6.6.6 does have `--target js`, but it cannot build `src/`
  ("ts lex error before emit"), and detection needs processes and the
  filesystem anyway. A Node package can wrap the CLI and its JSON, exactly as
  the Python binding does, with no compiler target at all.
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
- [ ] **mihi-side smoke of the `[lib]` surface** — mihi M3
  (`mihi_gpu_vendor` / `mihi_gpu_model` on a Ryzen 5800H); tracked in mihi's
  roadmap.

---

## Considered and declined

- **`lib/regex.cyr` for parser output** (2.1.3) — the parsers split lines,
  split CSV fields and check single tokens; nothing there needs regex.
- **`lib/chrono.cyr` for the cache TTL** (2.1.2) — a dependency for a
  three-line saving. Revisit if something needs chrono's ISO-8601 / duration
  surface anyway.
- **Multi-return `(value, error)` in `detect/*`** (2.1.3) — detectors push
  0..N profiles and 0..M warnings into out-param vectors; there is no single
  value to return.
- **Adopting `lib/test.cyr`** (2.1.2) — it is a parameterised-test helper, not
  a replacement for `lib/assert.cyr`, which the tests already use.

## Closed in the 2026-09-23 review

Open items that were already done, obsolete or superseded, and why:

- **`cyrius.lock` committed + `cyrius deps --verify`** — done. CI's *Verify
  dep hashes* step compares the lock with `HEAD` (since 2.3.21).
- **`lib/json.cyr` audit** — obsolete. `json.cyr` was folded into bayan in
  cyrius 6.2.11. The serializer is hand-rolled, and parsing has used
  `bayan-json` since 2.3.15.
- **Second "adopt `lib/test.cyr`" entry** — duplicate of the declined item
  above.
- **Windows DXGI adapter enumeration and its cass smoke** (2.2.x) — done:
  DXGI VRAM in 2.3.9, `windows-smoke` in 2.3.11, DXGI enumeration in 2.3.25.
- **Upstream PE environment-read reroute** (2.3.12) — it exists
  (`0xF015`, `GetEnvironmentVariableA`). `cmd_getenv` has used it through the
  stdlib `getenv` since 2.3.21, verified on `cass`.
- **Defer-on-all-paths audit of file handles** — moot. `src/` opens no file
  handles; all file I/O goes through the stdlib's `file_read_all` /
  `file_write_all`.
- **Backfill 2.3.14 – 2.3.20 into the roadmap** — superseded by the Shipped
  table, which covers every release in the CHANGELOG.
- **`cyrius capacity --check`, "stalled on toolchain"** — unblocked; moved to
  2.4.x.

---

## Non-goals

- **Runtime execution** — detection and planning only, not inference or
  training.
- **Kernel driver management** — no installing or configuring drivers.
- **Cloud provisioning** — detect what is present, not what could be spun up.
