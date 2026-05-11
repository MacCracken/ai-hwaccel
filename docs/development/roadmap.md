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

## 2.1.0 — cc5 adoption arc

The toolchain bump in 2.0.1 was mechanical only — no source changed. 2.1.0
adopts the cc5-era features where they remove boilerplate, plug a real
gap, or tighten the CI gate. Each item is independent; ship in any order.

### Language features (adopt where they earn their keep)

- [~] **`#derive(accessors)` on the major struct types** — proof-of-concept
  + first two structs shipped in 2.1.3:
  - [x] `meta` (`src/model_format.cyr`) — 5 fields
  - [x] `storage` (`src/system_io.cyr`) — 3 fields, paired with the
    first cross-file raw-offset CI gate (param `sd` is unambiguous).
  - [ ] `interconnect` (`src/system_io.cyr`) — 4 fields. *Naming
    decision needed*: current accessors are `ic_*` (shorthand) but
    `#derive(accessors) struct interconnect` would generate
    `interconnect_*`. Either rename the struct to `ic` or mass-update
    call sites to the long form. Carries to 2.1.4.
  - [ ] `profile` (`src/profile.cyr`) — big struct (≈20 fields,
    multiple optional). The largest single conversion in the arc;
    canonical param is `p`, which is also used as the `model` param
    in `src/model.cyr`. Needs the field-count bound check (libro
    pattern) alongside the cross-file guard.
  - [ ] `accelerator_registry` — `reg_*` accessors today. Param canonical
    is `r`; check ambiguity before adding the cross-file guard.
  - [ ] `sharding_plan` (`src/system_io.cyr`).
  - [ ] `training_method` (`src/training.cyr`).
  - [ ] `model` (`src/model.cyr`). Note: param `m` is shared with the
    just-derived `meta` struct, so the libro-style cross-file guard
    can't use `check_struct meta src/model_format.cyr m`. Both go
    onto the field-count-bound list once `model` is derived.

  Each conversion ships with a CI gate update — the new accessor
  surface should be the only way in. See `.github/workflows/ci.yml`'s
  `Raw-offset guard` step.
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
  — shipped in slot 1. Final mapping:
  - `foundation_test.tcyr`     (errors, accel types, units, quantization)
  - `profile_test.tcyr`         (profile construction + setters)
  - `registry_test.tcyr`        (registry assembly)
  - `detect_gaudi_test.tcyr`    (Gaudi detection)
  - `detect_neuron_test.tcyr`   (Neuron detection)
  - `sharding_test.tcyr`        (plan / training)
  - `system_io_test.tcyr`       (sysfs / proc reading)
  - `cost_model_test.tcyr`      (cost / recommend)
  - `json_output_test.tcyr`     (JSON serialization)
  - `model_format_test.tcyr`    (SafeTensors / GGUF / ONNX / PyTorch)
  - `requirement_test.tcyr`     (requirement matching)
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

## 2.2.0 — Platform Validation
*(was 1.3.0 in the Rust roadmap)*

Live cloud hardware validation and remaining cross-platform gaps.

### Live cloud validation

- [ ] **NVIDIA H100 / A100** — capture real nvidia-smi output (AWS/GCP)
- [ ] **NVIDIA Grace Hopper GH200** — validate unified memory on real GH200
- [ ] **AMD MI300X** — validate CXL memory, ROCm sysfs (Azure)
- [ ] **Google TPU v5e / v5p** — multi-host pod slice testing (GCE)
- [ ] **AWS Neuron (trn1/inf2)** — validate on mixed instances
- [ ] **Intel Gaudi 3** — validate on AWS DL1/DL2

### Cross-platform

- [ ] **Windows: DirectX adapter enumeration** — DXGI `EnumAdapters1`.
  Returns adapter LUID, dedicated VRAM, shared memory, driver version.
  cc5 ships a Win64 PE backend (5.10.x), so this is now reachable from
  cyrius directly rather than via FFI.

### Untested backends

- [ ] **Cerebras WSE** — needs Cerebras Cloud access
- [ ] **Graphcore IPU** — needs Paperspace or IPU cloud access
- [ ] **Groq LPU** — blocked on public Linux driver
- [ ] **Samsung NPU** — needs Exynos device (Galaxy S24+)
- [ ] **MediaTek APU** — needs Dimensity device

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
