# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

The project numbering inherited the Rust roadmap; the Cyrius port shipped
as 2.0.0, so what the original plan called "1.3 / 1.4 / 1.5 / 2.0" has
been re-shelved as 2.2 / 2.3 / 2.4 / 2.5 below. Scope is preserved.

---

## 2.0.1 — Toolchain modernization (SHIPPED, 2026-05-10)

See CHANGELOG. Pure mechanical bump from cyrius 3.10.0 → 5.10.34. Sets
up everything the 2.1.0 adoption arc needs (cc5 IR, manifest interpolation,
`cyrius deps` / `cyrius.lock`, modernized CI, fmt-clean tree).

---

## 2.1.x — cc5 adoption arc (SHIPPED, 2026-05-10 → 2026-05-11)

Closed at 2.1.7 (P(-1) scaffold hardening). Seven slots:
- **2.1.0** — test reorg + CI tighten
- **2.1.1** — Rust parity audit
- **2.1.2** — defer audit, chrono rejected, build/ untrack
- **2.1.3** — `#derive(accessors)` on meta + storage
- **2.1.4** — ic + plan + est
- **2.1.5** — reg + model
- **2.1.6** — profile (the big one)
- **2.1.7** — P(-1) close-out: remaining 8 structs derived (env, sio,
  shard, cloud_inst, rec, cached, disk_cached, lazy). Every heap struct
  in the project is now on the accessor surface. CI gate at 15 entries
  (10 cross-file + 5 field-count bound).

---

## 2.1.0 — cc5 adoption arc

The toolchain bump in 2.0.1 was mechanical only — no source changed. 2.1.0
adopts the cc5-era features where they remove boilerplate, plug a real
gap, or tighten the CI gate. Each item is independent; ship in any order.

### Language features (adopt where they earn their keep)

- [x] **`#derive(accessors)` on the major struct types — DONE in 2.1.6.**
  All 8 derivable structs are now on the accessor surface; the CI gate
  registers 5 cross-file `check_struct` guards and 4 field-count bound
  checks. See `.github/workflows/ci.yml`'s `Raw-offset guard` step.
  - [x] `meta` (`src/model_format.cyr`) — 5 fields. **2.1.3.**
  - [x] `storage` (`src/system_io.cyr`) — 3 fields, first cross-file
    raw-offset CI gate (param `sd` unambiguous). **2.1.3.**
  - [x] `ic` (interconnect, `src/system_io.cyr`) — 4 fields. **2.1.4.**
  - [x] `plan` (sharding, `src/system_io.cyr`) — 5 fields. **2.1.4.**
  - [x] `est` (MemoryEstimate, `src/training.cyr`) — 4 fields, field-count
    bound (param `e` shared with `runtime_env`). **2.1.4.**
  - [x] `reg` (accelerator_registry, `src/registry.cyr`) — 4 fields.
    Cleaned up 9 internal `load64(r)` shortcuts. **2.1.5.**
  - [x] `model` (`src/model.cyr`) — 4 fields, field-count bound (param
    `m` shared with `meta`). **2.1.5.**
  - [x] `profile` (`src/profile.cyr`) — **20 fields**, biggest struct
    in the codebase. Param `p` unambiguous — cross-file `check_struct`.
    Converted 4 cross-file raw `store64(p + 24, …)` writes (cuda /
    rocm / vulkan / gaudi memory_bytes overrides) to
    `profile_set_memory_bytes(p, …)`. **2.1.6.**
- [x] **Multi-return `(value, error)` in detect/* — investigated, doesn't
  fit.** Closed in 2.1.3 review without code change. The detect/ entry
  points are `detect_<backend>(profiles, warnings)` — both vec
  OUT-params — that push 0..N profiles and 0..M warnings, then return
  an unused 0. There is no single value to multi-return, and errors are
  already accumulated into `warnings` as structured entries (not
  collapsed to a sentinel int). The 2.1.0 entry assumed a `() →
  profile_or_sentinel` shape that doesn't match the codebase. Keeping
  the out-param-vec pattern as canonical.
- [ ] **`case N: { ... }` switch blocks** for the enum dispatch tables —
  `accel_name()`, `family_name()`, `format_name()`,
  `_gguf_file_type_name()`, `requirement_satisfied()`. Currently long
  if/else-if chains; switch blocks compile to a jump table and document
  intent better. *Attempted 2.1.2-pre; reverted*: cc5 5.10.x's
  `PARSE_CASE` accepts numeric literals only — enum identifiers like
  `case FAMILY_CPU:` fail with `expected number, got identifier`. The
  v5.10.48 enum-const-fold landed for `PARSE_ARRAY` / `PARSE_GVAR_ARR`
  only. Revisit after an upstream patch extends fold to case labels;
  using `case 0:` with enum-name comments was rejected as too brittle
  to enum renumbering.
- [ ] **Defer-on-all-paths** — audit `system_io.cyr` and the file-handle
  paths in `cache.cyr` for missed close-on-error. cc4+ runs defers on
  every exit including early returns; some current code returns without
  closing on the error branch.

### CI / tooling tighten-up

- [x] **`cyrius vet`** — include-graph audit. Added as CI step in slot 1
  (reports `36 deps, 0 untrusted, 0 missing` for the current tree).
- [ ] **`cyrius capacity --check`** — global-var ceiling gate. *Stalled on
  toolchain*: cc5 5.10.x's `cyrius capacity` doesn't honour the manifest
  `[deps].stdlib` auto-prepend, so it warns on every stdlib symbol when
  src/main.cyr relies on the implicit include path. Reach out upstream
  or re-evaluate after a cc5 patch; revisit when the warning floor is 0.
- [ ] **`cyrius.lock` committed + `cyrius deps --verify` enforced** —
  *not applicable while stdlib-only*: cyrius only writes a lockfile for
  `[deps.<git>]` entries. The CI step stays in place so it engages the
  moment a git dep gets added (e.g. an agnosys / libro pin for some
  future feature). The "soft-skip" wording was renamed to
  "no cyrius.lock (stdlib-only project) — nothing to verify".
- [x] **fmt drift gate** — expanded to cover `tests/tcyr/*.tcyr`,
  `fuzz/*.fcyr`, `benches/*.bcyr` in slot 1.

### Test infrastructure

- [x] **`tests/test_phase{1..11}.cyr` → `tests/tcyr/<descriptive>_test.tcyr`**
  — shipped in 2.1.0; **6 of 11 names corrected in 2.2.0** after audit
  revealed the original rename assumed phase numbers mapped 1:1 to
  subjects, which several didn't. Corrected mapping:
  - `foundation_test.tcyr`     (errors, accel types, units, quantization)
  - `profile_test.tcyr`        (profile construction + accessors + throughput)
  - `requirement_test.tcyr`    (accelerator requirement matching)
  - `json_output_test.tcyr`    (JSON serialization)
  - `model_format_test.tcyr`   (SafeTensors / GGUF / ONNX / PyTorch)
  - `registry_test.tcyr`       (registry + builder + suggest_quant — *was system_io_test*)
  - `io_test.tcyr`             (which / run_tool / CSV / sysfs — *was registry_test*)
  - `gpu_parser_test.tcyr`     (CUDA + Gaudi + Neuron parsing — *was detect_gaudi_test*)
  - `backend_test.tcyr`        (Apple + Intel + AMD XDNA + cloud ASIC — *was detect_neuron_test*)
  - `topology_test.tcyr`       (interconnect / bandwidth / PCIe / storage — *was sharding_test*)
  - `planning_test.tcyr`       (sharding plans + training memory + model — *was cost_model_test*)
- [ ] **Adopt `lib/test.cyr`** stdlib module — drops the local `assert`
  helpers in favor of the toolchain-tracked surface. Test summary
  format ("0 failed") is what CI greps for; `lib/test` already emits it.

### New stdlib adoption (where the win is concrete)

- [x] **`lib/regex.cyr` for parser output — investigated, no fit.** Closed
  in 2.1.3 review. The detect/ parsers don't actually hand-roll
  string scanning that regex would replace — they go `run_tool` →
  `str_split` (lines) → `parse_csv_line` (fields) →
  `str_contains_cstr` (single-token substring checks like `"GH200"`,
  `"gaudi3"`). Substring checks aren't what regex is for; the CSV
  helpers are already idiomatic. Regex would be a sledgehammer for
  cases that are already a finishing nail.
- [x] **`lib/chrono.cyr` for cache TTL — investigated, rejected.** Shipped
  as a no-op in 2.1.2. The local `syscall(228, CLOCK_MONOTONIC, &ts)`
  in `cache.cyr` is 4 lines; replacing with `clock_now_ms() / 1000`
  adds chrono as a `[deps].stdlib` entry for a 3-line save. Cost / benefit
  doesn't justify the dep. (Revisit if any future code wants chrono's
  ISO-8601 / duration / sleep_ms surface — then the chrono dep pays for
  itself and this monotonic helper rides along.)
- [ ] **`lib/json.cyr` audit** — we already use it (stdlib dep). Verify
  the cc5-era version's API matches what `json_out.cyr` is doing; the
  `str_builder` pattern may now have a more direct
  `json_writer` surface upstream.
- [x] **`lib/test.cyr` adoption — closed, was a misread.** Looked at in
  2.1.2 review. `lib/test.cyr` is a `test_each` parameterised-test
  helper, not an alternative assertion framework. The tests already
  use stdlib `lib/assert.cyr` (the `assert`, `assert_eq`,
  `assert_summary` surface). Nothing to migrate.

### Dist bundle (defer until consumer demand)

- [ ] **`[lib]` section + `cyrius distlib`?** Pattern-match against
  agnosys/libro: a single bundled `dist/ai-hwaccel.cyr` lets consumers
  `include "lib/ai_hwaccel.cyr"` from their own manifest. Today the
  consumer list is hoosh / daimon / Irfan / AgnosAI / murti / tazama —
  all consume the binary, not a library. Hold until a consumer asks.

### Out of scope (kept for 2.2+)

- New hardware backends, new detection paths, new CLI subcommands,
  changes to `--json` output schema. 2.1.0 is structural cleanup only.

---

## 2.2.x — Platform Validation
*(was 1.3.0 in the Rust roadmap; 2.2.0 itself shipped as a test-rename /
README refresh slot — not a real Platform Validation release. The items
below are **open and queued for pickup**, not deferred.)*

The pattern for every item here: implement the detection / parsing path
against a synthesized fixture first (no hardware needed — `tests/fixtures/`
or inline test strings), then wire to real hardware as access becomes
available. Hardware-access gating doesn't block the source-side work;
the parser + struct construction can ship without ever booting the
target device.

### Cross-platform (no hardware needed for source work)

- [ ] **Windows: DXGI adapter enumeration** — DXGI `EnumAdapters1` →
  adapter LUID, dedicated VRAM, shared memory, driver version. Stack:
  add `cc5_win` to the install bundle (or invoke the `cc5_win_cross`
  artifact directly from the cyrius source build), wire
  `src/detect/windows.cyr` behind `#ifdef CYRIUS_TARGET_WIN64`, COM
  binding for `IDXGIFactory1`. Ship CI cross-build (best-effort,
  mirroring the aarch64 pattern). One slot for the skeleton + fixture
  test; follow-up slot for the COM binding. *Next pickup target.*
  **Real-hardware validation: `ssh cass`** — project's Windows host
  (same `cass` referenced in cyrius's v5.10.x cross-host smoke
  testing). Use the `.bat`-indirection or `cmd /v /c "…
  !errorlevel!"` wrapper for exit-code propagation under the current
  5.10.34 pin — `%errorlevel%` in `cmd /c` parses at the wrong time
  and false-reports `exit=0`. **The wrapper gotcha is addressed in
  cyrius 5.11.6** (next release in line); bumping the pin to 5.11.6+
  alongside this slot collapses the cross-host smoke back to plain
  shell. See `memory/reference_windows_host.md`.

### Hardware validation (fixture-first, hardware-second)

- [ ] **NVIDIA H100 / A100 / GH200** — capture real `nvidia-smi`
  CSV from AWS p5 / GCP a3-high instances → `tests/fixtures/cuda/`,
  add fixture tests that exercise `parse_cuda_output`. GH200's unified
  memory (`mem_bytes + 480 GiB`) is already coded — fixture locks it in.
- [ ] **AMD MI300X / MI250** — capture `/sys/class/drm/*/device/*`
  contents → `tests/fixtures/rocm/`. CXL memory path (`mem_info_vis_vram_total`)
  is already coded — fixture locks in the MI300X case.
- [ ] **Google TPU v5e / v5p** — capture `/sys/class/accel/*` contents
  on a GCE v5 slice → `tests/fixtures/tpu/`. Multi-host pod slice
  testing carries to the second-pass slot.
- [ ] **AWS Neuron trn1 / inf2** — capture `neuron-ls --json` from
  trn1.32xlarge → `tests/fixtures/neuron/`. Multi-device fixture
  covers the per-core count math.
- [ ] **Intel Gaudi 3** — capture `hl-smi --query-aip` on Gaudi3
  (AWS DL2) → `tests/fixtures/gaudi/`. HL-325 device-name override
  path locks in here.

### Untested backends (open — implement parser against fixture, then
verify on hardware when access happens)

- [ ] **Cerebras WSE** — `/dev/cerebras*` + sysfs. Fixture capture
  needed (sample from Cerebras docs or contributor with access).
- [ ] **Graphcore IPU** — `gc-info` output → fixture. Synthesizable
  from public Graphcore SDK documentation.
- [ ] **Groq LPU** — `/dev/groq*` sysfs. Driver isn't on public Linux
  distros today, but the sysfs format is documented; fixture-first
  is still viable.
- [ ] **Samsung NPU** — `/sys/class/npu` on Galaxy S24+ (Exynos).
  Fixture from a Samsung dev portal capture.
- [ ] **MediaTek APU** — `/sys/class/misc/apusys` on Dimensity.
  Fixture from MediaTek NeuroPilot docs.

### Supporting infrastructure (any slot)

- [ ] **`tests/fixtures/` directory** — move the inline sample tool
  outputs in `gpu_parser_test.tcyr` / `backend_test.tcyr` into per-
  backend fixture files. Establishes the contribution pattern for
  every backend item above ("here's a capture of `<tool> <args>` on
  `<hardware>` — add it to fixtures/").
- [ ] **`cyrius vet` widened to tests/tcyr/** — currently scans only
  `src/main.cyr`. Each test unit is its own compilation root with
  its own include graph; vetting tests too closes a coverage gap.

---

## 2.3.0 — Ecosystem
*(was 1.4.0 in the Rust roadmap)*

Bindings and packaging.

### Python bindings

- [ ] **Complete API surface** — `AcceleratorProfile`, `SystemIo`,
  `Interconnect`, `StorageDevice`, `ShardingPlan`, `TrainingMethod`
- [ ] **`pip install ai-hwaccel`** — wheels for Linux (manylinux),
  macOS (universal2), Windows (x86_64). cyrius cc5 emits ELF/Mach-O/PE,
  so the wheel build is `cyrius build` per-target rather than maturin
- [ ] **Python-native ergonomics** — dict-like objects, JSON
  serialization, pandas DataFrame export

### WASM / JS

- [ ] **JS/TS bindings** — depends on cyrius WASM target (not in
  5.10.x; gate this on upstream readiness)

---

## 2.4.0 — Multi-Node & Hot-Plug
*(was 1.5.0 in the Rust roadmap)*

### Multi-node detection

- [ ] **SSH probe** — `registry_detect_remote(hosts)`, merge into
  cluster-wide registry
- [ ] **Cluster-aware sharding** — distribute across nodes, consider
  IB/RoCE bandwidth for pipeline vs data parallelism

### Hot-plug support

- [ ] **`udev` watcher (Linux)** — `registry_watch()` returns
  `DeviceEvent::Added` / `DeviceEvent::Removed` stream. cc5's defer
  + `lib/thread.cyr` make this cleaner than the Rust plan assumed.
- [ ] **Dynamic registry updates** — `CachedRegistry` auto-invalidates
  on hot-plug events

### Remaining platforms

- [ ] **Android** — HAL `hwbinder` for NNAPI accelerator list
- [ ] **FreeBSD** — DRM sysctl equivalents for GPU detection
- [ ] **Intel oneAPI** — DC GPU Max HBM vs DDR tiers on real hardware

---

## 2.5.0 — Fleet & Scale
*(was 2.0.0 in the Rust roadmap)*

Fleet-wide inventory, health monitoring, capacity planning at datacenter
scale.

### Fleet inventory

- [ ] **Fleet registry** — `FleetRegistry` aggregates registries from
  multiple nodes. Tracks hostname, IP, rack, zone
- [ ] **Discovery protocols** — mDNS/DNS-SD, Consul, Kubernetes node labels
- [ ] **Inventory persistence** — SQLite or JSON lines, diff against
  previous scan
- [ ] **Fleet CLI** — `ai-hwaccel fleet scan --subnet 10.0.0.0/24`

### Health monitoring

- [ ] **Continuous health checks** — periodic re-detection, trend tracking
- [ ] **Alert rules engine** — configurable thresholds, webhook/Slack/PagerDuty
- [ ] **Dead GPU detection** — flag nodes with missing accelerators
- [ ] **ECC error tracking** — nvidia-smi ECC, ROCm ras_features

### Capacity planning

- [ ] **Fleet-wide sharding** — recommend nodes and distribution strategy
- [ ] **Bin packing** — optimal multi-model placement across fleet
- [ ] **Scaling recommendations** — "you need 3 more H100 nodes for 405B"

### Observability & export

- [ ] **Prometheus metrics** — per-device gauges, fleet aggregates
- [ ] **OpenTelemetry spans** — instrument detection with OTel traces
- [ ] **Grafana dashboard template** — fleet GPU utilization heatmap
- [ ] **Structured event log** — JSON lines for ELK/Loki/Datadog

### Multi-tenancy

- [ ] **Device reservation** — `registry_reserve(device_id, owner)`
- [ ] **Namespace isolation** — Kubernetes pod-scoped detection
- [ ] **Quota management** — per-team GPU hour budgets

---

## Future

- [ ] **Power budget planning** — recommend device mix for power cap
- [ ] **Thermal throttling prediction** — warn on approaching thresholds
- [ ] **Plugin system** — third-party backends via dynamic loading

---

## Non-goals

- **Runtime execution** — detection and planning only, not inference/training
- **Kernel driver management** — no installing or configuring drivers
- **Cloud provisioning** — detect what's present, not what could be spun up
