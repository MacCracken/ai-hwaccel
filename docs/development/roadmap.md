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

- [ ] **`#derive(accessors)` on the major struct types** — `profile`,
  `accelerator_registry`, `model_meta`, `sharding_plan`, `training_method`,
  `interconnect`, `storage_device`. Drops the hand-rolled `_set_*` setter
  zoo and the `load64(p + N)` raw-offset reads. Pair each conversion
  with the libro-style raw-offset CI gate (per-file allowlist + per-struct
  field-count bound) so the new accessor surface is genuinely the only
  way in.
- [ ] **Multi-return `(value, error)`** in the `detect/` family. Today
  the backends return sentinel values (`-1`, empty string, `ERR_*` int)
  and callers re-derive intent. `detect_cuda()` → `(profile, err)`,
  `detect_rocm()` → same, etc. Cleans up `system_io.cyr` callers and
  makes the partial-detection-with-warnings flow explicit.
- [ ] **`case N: { ... }` switch blocks** for the enum dispatch tables —
  `accel_name()`, `family_name()`, `format_name()`,
  `_gguf_file_type_name()`, `requirement_satisfied()`. Currently long
  if/else-if chains; switch blocks compile to a jump table and document
  intent better.
- [ ] **Defer-on-all-paths** — audit `system_io.cyr` and the file-handle
  paths in `cache.cyr` for missed close-on-error. cc4+ runs defers on
  every exit including early returns; some current code returns without
  closing on the error branch.

### CI / tooling tighten-up

- [ ] **`cyrius vet`** — include-graph audit. Catches the `include "src/X.cyr"`
  ↔ `src/main.cyr` drift that's currently invisible (no [lib] section
  to validate against). Add as a workflow step.
- [ ] **`cyrius capacity --check`** — global-var ceiling gate. The
  CLAUDE.md "1024 global var limit" principle becomes machine-checked
  rather than reviewer-checked.
- [ ] **`cyrius.lock` committed + `cyrius deps --verify` enforced** —
  drop the soft-skip in the CI workflow once the lockfile lands.
- [ ] **fmt drift gate** — already in place from 2.0.1, but expand to
  cover `tests/` and `fuzz/` (currently only `src/`-level files in
  some sweeps).

### Test infrastructure

- [ ] **Migrate `tests/test_phase*.cyr` → `tests/tcyr/*.tcyr`** — matches
  the agnosys / libro / cyrius-internal convention. Each `.tcyr` is a
  standalone compilation unit `cyrius build` recognizes and CI can
  iterate over uniformly. Lets us drop the per-file `include "lib/*"`
  preamble (cyrius auto-prepends from manifest deps for `.tcyr`).
- [ ] **Adopt `lib/test.cyr`** stdlib module — drops the local `assert`
  helpers in favor of the toolchain-tracked surface. Test summary
  format ("0 failed") is what CI greps for; `lib/test` already emits it.

### New stdlib adoption (where the win is concrete)

- [ ] **`lib/regex.cyr` for parser output** — `cuda.cyr`, `gaudi.cyr`,
  `neuron.cyr`, `apple.cyr` currently hand-roll string scanning of
  nvidia-smi / hl-smi / neuron-ls / system_profiler output. A pinned
  set of regex patterns is more readable and survives output-format
  drift better. Bench before/after — regex isn't free, but the parsing
  isn't hot-path.
- [ ] **`lib/chrono.cyr` for cache TTL** — `cache.cyr` and
  `lazy.cyr` currently call `syscall(time)` directly. chrono wraps
  it with monotonic-clock helpers and `time_since(t)` semantics.
- [ ] **`lib/json.cyr` audit** — we already use it (stdlib dep). Verify
  the cc5-era version's API matches what `json_out.cyr` is doing; the
  `str_builder` pattern may now have a more direct
  `json_writer` surface upstream.

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
