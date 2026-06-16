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

## 2.3.0 — Toolchain 6.0.25 + serialization/dedup audit (SHIPPED, 2026-06-01)

A toolchain-modernization + audit release, in the same vein as 2.0.1 and
the 2.1.x arc. The "Ecosystem" scope (Python/WASM bindings) that the
Rust roadmap had penciled in for this slot moves to 2.3.1 (below) — it's
a separate effort and ships after this release. Contents:

- **Pin: cyrius 6.0.0 → 6.0.25**, stdlib re-synced into `./lib/`,
  drift warning gone.
- **JSON serializer hot path** — single-byte structural punctuation
  moved from `str_builder_add_cstr` to `str_builder_putc`.
  `json_serialize_13dev` **−8.7%** (26946 → 24602 ns, min-of-6 @ 2000
  iters); `json_summary_13dev` flat (within noise — too few single-byte
  appends to move). Byte-identical output.
- **`Str` → owned cstr dedup** — the hand-inlined
  `alloc + memcpy + NUL` idiom consolidated onto stdlib `str_cstr`
  across 11 detectors / 16 sites. DCE binary −1592 bytes, dist bundle
  −76 lines, perf-neutral on parsing benches.
- **Bench harness** — `registry.bcyr` JSON benches now print
  nanosecond averages (the µs-truncated `bench_report` hid the delta).
- All gates green: 11 test units / 518 assertions, 6 fuzz harnesses,
  `vet`, raw-offset guard, distlib determinism.

Per the (now mandatory — see CLAUDE.md) benchmarking policy, the
before/after deltas are recorded in `bench-history.csv` and the
CHANGELOG 2.3.0 table.

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

### Dist bundle — promoted to 2.2.4

- [x] **`[lib]` section + `cyrius distlib`** — promoted to its own
  release slot at **2.2.4** (2026-05-19) after `mihi` arrived as the
  first library-shaped consumer: mihi's no-exec rule forbids shelling
  out to the binary, so the GPU surface mihi needs (M3 in mihi's own
  roadmap) is gated on this reshape. See 2.2.4 below for the
  acceptance criteria.

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

- [~] **Windows: DXGI adapter enumeration** — DXGI `EnumAdapters1` →
  adapter LUID, dedicated VRAM, shared memory, driver version. *In
  progress*:
  - [x] **2.2.2** — `src/detect/windows.cyr` skeleton behind
    `#ifdef CYRIUS_TARGET_WIN`, wired into main.cyr include graph,
    Linux build byte-identical.
  - [ ] **2.2.3** — DXGI COM binding + `DXGI_ADAPTER_DESC1` parser,
    Linux-side fixture tests under `tests/fixtures/windows/`.
  - [ ] **CI cross-build + cass smoke** — re-verify against
    `cycc_win` 6.0.0 (legacy name `cc5_win`). The 5.11.5-era PE
    emit regression documented in
    `memory/feedback_cc5_win_exit_propagation.md` may already be
    closed by the 6.0.0 toolchain; smoke probe required before
    declaring this unblocked. Linux-hosted fixture tests in 2.2.3
    don't depend on this either way.

### 2.2.4 — `[lib]` reshape (mihi unblock) — SHIPPED 2026-05-19

Promoted out of the 2.1.0 "Dist bundle (defer until consumer demand)"
slot now that a real library consumer landed. `mihi` v0.3.0 (released
2026-05-19) needs the GPU primitives via `include`, not `exec` — its
CLAUDE.md forbids spawning processes from probes. This release adds
the `[lib].modules` surface and the `cyrius distlib` output so mihi
(and any future library consumer) can pin against ai-hwaccel from
their own `cyrius.cyml`.

- [x] **`[lib].modules` declared in `cyrius.cyml`** — 35 modules in
  `src/main.cyr` include order. Excluded: `src/main.cyr` (CLI argv
  parsing), `src/json_out.cyr` (binary output formatting — consumers
  do their own). Every detection backend, registry/profile/plan
  surface, the cost / training / model-format / requirement /
  async / cache / lazy modules all ride along.
- [x] **`cyrius distlib` produces deterministic `dist/ai-hwaccel.cyr`**
  — 5392 lines / 168 KiB at 2.2.4. Two sequential invocations sha256
  to the same digest. Bundle is committed at the tag so consumer
  `cyrius deps` resolves at git-archive-fetch.
- [x] **Consumer-facing entry shim + README example** —
  `[deps.ai-hwaccel]` block plus an `include "lib/ai-hwaccel.cyr"`
  call site in the README "Using as a library" subsection. `cyrius
  deps` lands the bundle at `lib/ai-hwaccel.cyr` (filename verified
  empirically against the cyrius 6.0.0 toolchain).
- [x] **No CLI regression** — `build/ai-hwaccel` rebuilds to 287096
  bytes (byte-identical to 2.2.3); all 11 test units pass with 518
  assertions clean post-`[lib]` addition.
- [x] **Determinism + freshness guard in CI** — new `distlib drift +
  determinism` step regenerates the bundle, diffs against the
  committed copy, then re-runs and sha256-compares. Sits between
  `Lint` and `Build (DCE)`. Mirrors the libro / mihi / yukti gates.
- [ ] **`mihi-side smoke`** — after 2.2.4 publishes, mihi M3 lands
  `mihi_gpu_vendor` / `mihi_gpu_model` against the new `[lib]`
  surface; smoke on archaemenid (Ryzen 7 5800H with Radeon Graphics)
  prints non-null GPU lines. *External — tracked in mihi's roadmap.*

**Acceptance**: a consumer manifest with
`[deps.ai-hwaccel] tag = "2.2.4" modules = ["dist/ai-hwaccel.cyr"]`
resolves via `cyrius deps`, the consumer can
`include "lib/ai-hwaccel.cyr"`, and the detection entry points are
callable without invoking the CLI binary. **Met for the in-repo
deliverables; mihi-side verification queues for v0.4.0.**

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

## 2.3.x — Ecosystem
*(was 1.4.0 in the Rust roadmap, penciled in as 2.3.0; bumped because the
2.3.0 slot shipped as the toolchain 6.0.25 + serialization/dedup audit on
2026-06-01 — see the SHIPPED section above. Sequenced across patches: the
compiled binary + its JSON contract is the language-neutral substrate;
Python is the first binding, an AgnosAI / agnos-kernel target follows.)*

Bindings and packaging.

### 2.3.1 — JSON surface extension (SHIPPED, 2026-06-01)

The data layer every binding consumes. Schema bumped v3 → v4.

- [x] **Full detection surface reachable as JSON** — `system_io`
  (interconnects / storage / runtime environment) added to the default
  registry JSON; new `--plan` (ShardingPlan), `--train` (training
  memory), and `--cost --json` modes. Serializers in `src/json_out.cyr`
  + `cost_to_json` in `src/cost.cyr`. 36 JSON assertions; ns-resolution
  benches for each new serializer.

### 2.3.2 — Python bindings (SHIPPED, 2026-06-01)

`bindings/python/` — a thin, dependency-free wrapper over the binary +
schema-v4 JSON. No `.cyr` changed (binary identical to 2.3.1).

- [x] **Complete API surface** — typed dataclasses for `Registry`,
  `AcceleratorProfile`, `SystemIo`, `Interconnect`, `StorageDevice`,
  `RuntimeEnvironment`, `ShardingPlan`, `ModelShard`, `TrainingMemory`,
  `CostReport`. API: `detect / summary / plan / training_memory / cost /
  version`. Binary discovery via arg / `AI_HWACCEL_BIN` / bundled / PATH.
- [x] **Python-native ergonomics** — dataclasses, fixed-point
  convenience properties, optional pandas `to_dataframe()`. 15-test
  `unittest` suite (9 model + 6 e2e).
- Known limitation tracked for 2.3.3: binary reads `VERSION` /
  `data/cloud_pricing.json` cwd-relative.

### 2.3.3 — Working-dir-independent data files (SHIPPED, 2026-06-01)

- [x] **`AI_HWACCEL_DATA_DIR` resolution** — binary locates `VERSION` +
  `data/cloud_pricing.json` via the env var (cwd fallback preserved);
  the Python wrapper sets it for the bundled binary. `version()` /
  `cost()` now work regardless of cwd. `scripts/stage_binary.sh` stages
  `_bin/{binary, VERSION, data/}`; `version-bump.sh` made dist-aware.
  Closes the 2.3.2 known limitation.

### 2.3.4 — Linux wheels + machinery (SHIPPED, 2026-06-01)

- [x] **`pip install ai-hwaccel` (Linux)** — platform wheels bundling a
  static `ai-hwaccel` binary + data (subprocess + JSON; no FFI).
  `manylinux2014_x86_64` (built + venv-validated) and
  `manylinux2014_aarch64` (cross-built). `setup.py` tags
  `py3-none-<plat>`; `scripts/{stage_binary,build_wheel,build_remote}.sh`;
  CI `wheels.yml` matrix. Extensible to an **agnos-kernel** target.

### 2.3.5 — Toolchain pin 6.0.25 → 6.0.30 (SHIPPED, 2026-06-01)

- [x] **Pin bump** — resolves the wrapper drift; stdlib re-synced from
  the 6.0.30 snapshot. No `.cyr` changed. Benches confirm no regression
  (`json_serialize_13dev` 25535 → 25245 ns, noise). All gates green on
  6.0.30.

### 2.3.6 — macOS wheel (SHIPPED, 2026-06-02)

- [x] **macOS arm64 wheel** — `ai_hwaccel-2.3.6-py3-none-macosx_11_0_arm64`.
  Arc: 6.0.30 shipped no Darwin compiler; 6.0.38 shipped the arm64
  compiler but surfaced a new blocker (`cyrius build` false-negatived the
  install check on `cyrius = "<pin>"` + `[deps] stdlib`); **cyrius
  6.0.40–6.0.43 fixed it** (three stacked Darwin-ABI defects — see cyrius
  issue `2026-06-02-macos-arm64-deps-stdlib-pin-check.md`, RESOLVED). Pin
  bumped 6.0.30 → 6.0.43. Built on `ecb` via `build_remote.sh ecb
  macosx_11_0_arm64`; verified end-to-end (Mach-O arm64, stdlib executes,
  all subcommands run). CI `wheels.yml` `macos` job enabled. `lib sync`
  on Darwin is bypassed (still-unported getdents64 surface) — `cyrius
  build` resolves `[deps] stdlib` directly. universal2 deferred until the
  backend also emits x86_64 Mach-O (see the sibling x86 issue).

### 2.3.7 — Windows wheel (SHIPPED, 2026-06-03)

- [x] **Windows x86_64 wheel** — `ai_hwaccel-2.3.7-py3-none-win_amd64`,
  with real CPU + GPU detection. Verified end-to-end on `cass` (Win 11):
  real total RAM + Intel UHD Graphics 600 detected, all subcommands run.
  - **Cross-built on Linux** — `cycc_win` (Linux-hosted, emits PE32+); no
    Windows runner. `stage_win_cross.sh` pipes `[deps] stdlib` +
    `src/main.cyr` → `cycc_win`. `wheels.yml` `windows` job on
    `ubuntu-latest`, enabled.
  - **cyrius unblocked it:** 6.0.50 unfroze `cycc_win` (was `cc5_win`
    5.11.69; fixed the PROT_READ build break); 6.0.51 routed Win32
    process creation (CreateProcessW) so detection's subprocess spawns
    work — cyrius issue
    `2026-06-03-windows-pe-syscall-surface-blocks-detection.md` (RESOLVED).
  - **Our side:** implemented `detect_windows` (wmic Win32_VideoController
    → `ACCEL_WIN_GPU`) + a Windows `detect_system_memory` branch (wmic
    ComputerSystem), replacing the DXGI stub / 16 GiB fallback. Pin →
    6.0.54.
  - **Follow-up (2.3.8):** native DXGI for precise VRAM (WMI AdapterRAM
    caps at 4 GiB) — gated on cyrius PE COM-vtable + dxgi.dll IAT (filed:
    `cyrius/docs/development/issues/2026-06-03-windows-pe-com-vtable-dxgi-for-gpu-enum.md`).

### 2.3.8 — Toolchain 6.0.70 (SHIPPED, 2026-06-05)

- [x] **Pin 6.0.54 → 6.0.70**, stdlib re-synced (89 files). Builds clean,
  12/12 test units pass. Accepted a ~5–10 % allocation-path cost from
  cyrius 6.0.64's global allocator spinlock — a heap-corruption fix the
  threaded `async_detect` path needs (justified in CHANGELOG 2.3.8).
- [~] **DXGI precise VRAM → deferred to 2.3.9.** 6.0.70 lands the
  foundation (`callptr`/`IR_CALL_INDIRECT`, `dxgi.dll!CreateDXGIFactory1`
  import + S_OK on cass, COM-vtable dispatch capability) but the real
  Win64 COM callee (`EnumAdapters`/`GetDesc` VRAM read) corrupts the
  caller frame on cass — fixed upstream in **6.0.71** (cyrius issue
  `2026-06-05-windows-com-vtable-real-callee-frame-corruption.md`).

### 2.3.9 — Windows DXGI precise VRAM + structured logging (cyrius 6.1.18)

- [x] **Pin 6.1.5 → 6.1.18**, stdlib re-synced (94 files; sakshi
  v2.2.6 → **v2.2.10** + new `fs_win.cyr` Windows `dir_list` port). The
  **compiler** bump is a pure codegen no-op: cycc 6.1.15 and 6.1.18 emit a
  **byte-identical** 370,776 B Linux binary (same bytes as the 6.1.5
  baseline). The synced **stdlib** adds **+16 B** (370,776 → 370,792 B),
  entirely sakshi 2.2.10 source content linked through `log.cyr` — not a
  hot path. 13/13 test units (606 assertions) pass;
  `fmt`/`lint`/`vet`/distlib-determinism
  gates clean; benchmark delta within noise (see CHANGELOG /
  bench-history.csv).
- [x] **DXGI precise VRAM enabled by default.** The `.rdata`-corruption
  residual that kept the DXGI pass gated behind `-D AI_HWACCEL_DXGI` was
  fixed upstream in **cyrius 6.1.7** (m128 array-padding / PE `.rdata`
  layout divergence; GPU-confirmed on cass via ai-hwaccel's own
  `str_builder_add_cstr("true")` repro). `src/detect/windows.cyr` un-gated:
  `detect_windows` parses WMI then enriches with `EnumAdapters1 → GetDesc1
  → DedicatedVideoMemory`, reporting `max(WMI, DXGI)`. **Verified on cass**
  (Windows 10.0.26200): clean, uncorrupted JSON; Intel UHD 600 reports
  1 GiB (WMI's shared figure correctly kept — integrated GPUs have ~0
  *dedicated* VRAM). Discrete precise-VRAM override path GPU-confirmed
  upstream.
- [x] **Structured logging** (`src/log.cyr` over stdlib `sakshi`) — now
  works on **Linux/macOS *and* Windows PE.** It was previously silently
  dropped on PE: sakshi holds syscall numbers in `var` slots, and cyrius's
  PE syscall reroute only fired for a **compile-time-literal** number, so
  `syscall(_SK_SYS_WRITE, …)` fell through to a non-functional raw `0F 05`
  and wrote nothing (no fault, exit 0). **Fixed in cyrius 6.1.18**: the PE
  reroute now resolves a `var`-held syscall number to its constant value,
  so `n=1` (write) routes to `WriteFile` (the original blockers —
  `cyrius/.../2026-06-09-pe-syscall-variable-number-not-rerouted.md` and
  `sakshi/.../2026-06-09-windows-pe-var-syscall-no-reroute.md`, sakshi
  roadmap **W1** — are resolved/archived). **Verified on cass** (Windows
  10.0.26200) with the synced sakshi v2.2.10: `detect -vv` (stdout→NUL)
  emits the full span on stderr — `[ENTER] detect`, `[DEBUG]
  detect: builder_mask=…`, `[DEBUG] windows: dxgi vram below wmi, kept wmi
  (bytes) 1073741824`, `[INFO] detect: profiles=2`, `[EXIT] detect`; the
  default WARN level stays correctly silent on success. stdout remains
  byte-clean for JSON consumers; no measurable hot-path cost. Bundle
  consumers must add `sakshi` to their `[deps] stdlib`. *Cosmetic residual:*
  the trace-id prefix is `[0]` on PE because `getpid` (syscall 39) isn't in
  the reroute whitelist; delivery is unaffected.
- Windows runtime CI gate → **deferred to 2.3.11** (the PE feature surface
  is only manually cass-smoked today; see below).

### 2.3.10 — Toolchain bump to cyrius 6.2.11 (json → bayan stdlib reorg)

- [x] **Pin 6.1.18 → 6.2.11**, stdlib re-synced (97 files). Headline is a
  stdlib reorganization: the standalone `lib/json.cyr` is gone, folded
  (with `base64`/`csv`/`u128`/`bigint`/`toml`/`cyml`) into the new
  `lib/bayan.cyr` distribution bundle; its functions renamed `json_*` →
  `bayan_json_*`. ai-hwaccel never called the stdlib JSON parser
  (`src/json_out.cyr` is a hand-rolled `str_builder` serializer;
  `src/model_format.cyr` does its own byte-level safetensors header parse),
  so the now-dangling `"json"` entry in `[deps] stdlib` was **dropped**
  rather than re-pointed at `bayan`. No `src/` changes.
- [x] **Benchmark-neutral, no regression.** Full suite A/B'd 6.1.18 vs
  6.2.11, same machine/iters. All moves within run-to-run noise (confirmed
  across 3× re-runs of the deterministic sub-µs rows — `total_memory_13dev`
  new 141–157 ns vs baseline 144–149 ns overlap). 12/12 test units pass,
  6/6 fuzz harnesses build, binary smoke-tested. See CHANGELOG /
  bench-history.csv.

### 2.3.11 — Windows runtime CI gate (sakshi roadmap W2)

**Why:** `wheels.yml` cross-builds the PE but only asserts the `MZ` magic, so
a silently-broken *runtime* feature ships green. 2.3.9 proved the risk is
real — both the DXGI `.rdata` corruption and the dropped PE logging were
caught **only** by manual `ssh cass` smoke, never by CI. The PE is the one
target with no automated runtime coverage. Close that gap.

- [x] **Added a PE runtime smoke leg** — the `windows-smoke` job in
  `wheels.yml` (`needs: windows`). Downloads the `wheels-windows` artifact,
  extracts the bundled `_bin/ai-hwaccel.exe`, and asserts on a real Windows
  execution:
  - [x] **(a) Detection** — bare `ai-hwaccel.exe` exits 0 and emits
    **well-formed JSON** (`schema_version` + non-empty `profiles[]`; a CPU
    profile is always present). Regression gate for the COM/`.rdata`
    corruption class. *(The CLI is flag-based — no `detect` subcommand; bare
    invocation is the full registry JSON.)*
  - [x] **(b) Structured logging** — `ai-hwaccel.exe -vv` writes the
    `[ENTER]`/`[EXIT] detect` span to **stderr** while stdout stays clean,
    and the **default level stays silent**. Regression gate for the PE
    var-syscall reroute (W1).
- [x] **Execution venue chosen: `windows-latest` ephemeral runner** — the
  preferred option (self-contained, no secrets, no self-hosted wiring).
  `ssh cass` remains the documented fallback if the cross-built PE ever
  fails to run cleanly on the GH runner.
- [x] **`VERSION` resolution checked** — `ai-hwaccel.exe --version`, run from
  `_bin/` (where the wheel co-locates VERSION beside the EXE), reports the
  bundled version (not `unknown`). The smoke runs it cwd-relative because of
  the PE env-var limitation below.

**Follow-up surfaced by this gate — `AI_HWACCEL_DATA_DIR` is a no-op on PE.**
`cmd_getenv` (`src/detect/command.cyr`) reads `/proc/self/environ`, which
doesn't exist on Windows, so the env var the Python `_runner` sets to point
the bundled binary at its data files is silently ignored on PE — `--version`
/ `--cost` fall back to **cwd-relative** resolution there. Detection (the
primary path) is unaffected. **→ Resolved in 2.3.12 (below) via a
`--data-dir` flag**, since the planned `GetEnvironmentVariable` path is
blocked on a cyrius reroute we can't add.

### 2.3.12 — `--data-dir` flag (PE-safe data-file resolution)

**Why:** the 2.3.11 gate confirmed `AI_HWACCEL_DATA_DIR` is unreadable on PE
(no `/proc/self/environ`; no `GetEnvironmentVariable` reroute in cyrius
6.2.11), so the bundled Windows wheel's `--version`/`--cost` were
cwd-dependent. The env channel can't be fixed in-repo. Route the data dir
through argv instead — readable on PE (`GetCommandLineW`/`CommandLineToArgvW`,
already wired).

- [x] **`--data-dir <path>` CLI flag** (`cmd_data_dir_arg()` in
  `command.cyr`). `data_file_path` precedence is **flag → env → cwd**;
  self-contained argv scan (works on PE and before main's parser). In
  `--help`.
- [x] **Python `_runner`** passes `--data-dir <bundled _bin>` on the argv
  for the bundled binary (env var still exported for back-compat;
  caller-set env still respected). `version()`/`cost()` now work on the
  Windows wheel from any cwd.
- [x] **Tests** — `test_bundled.py`: flag resolves VERSION with env stripped
  + foreign cwd; runner places `--data-dir` on argv. 20/20 Python + 12/12
  cyrius units pass; benchmark-neutral (no hot path touched).
- [ ] **Upstream (cyrius), drafted not filed:** add a PE environment-read
  reroute (`GetEnvironmentVariableW`) so `cmd_getenv` / stdlib `getenv`
  work on PE and the env channel reaches parity. ai-hwaccel no longer
  depends on it; tracked for completeness.

*Out of scope (upstream, not ours):* the `[0]` trace-id prefix on PE —
`getpid` (syscall 39) isn't in cyrius's PE reroute whitelist. Log a cyrius
issue if we want real trace ids on Windows; log delivery is unaffected.

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
  `DeviceEvent::Added` / `DeviceEvent::Removed` stream. cycc's defer
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
