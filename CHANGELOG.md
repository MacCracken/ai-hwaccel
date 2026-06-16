# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [semantic versioning](https://semver.org/) as of v0.19.3.

## [2.3.11] — 2026-06-15

**Windows PE runtime CI gate.** `wheels.yml` cross-builds the Windows PE
on Linux but previously asserted only the `MZ` magic — a silently-broken
*runtime* feature could ship green. The risk is proven: both 2.3.9
regressions (the DXGI `.rdata` corruption and the dropped PE structured
logging) were caught **only** by manual `ssh cass` smoke, never by CI.
The PE was the one target with no automated runtime coverage. This closes
that gap with a `windows-smoke` job that runs the actual cross-built EXE
on an ephemeral `windows-latest` runner and asserts three runtime
contracts.

#### Added

- **`windows-smoke` job — `.github/workflows/wheels.yml`** (`runs-on:
  windows-latest`, `needs: windows`). Downloads the `wheels-windows`
  artifact, extracts the bundled `_bin/ai-hwaccel.exe` from the wheel, and
  asserts on a real Windows execution:
  - **(a) Detection** — bare `ai-hwaccel.exe` exits 0 and emits
    **well-formed JSON** (`schema_version` + a **non-empty `profiles[]`**;
    a CPU profile is always present even on a GPU-less / `wmic`-less
    runner). Regression gate for the COM/`.rdata` corruption class.
  - **(b) Structured logging** — `ai-hwaccel.exe -vv` writes the
    `[ENTER]`/`[EXIT] detect` span to **stderr** while **stdout stays
    clean**, and the **default level stays silent**. Regression gate for
    the PE var-syscall reroute (sakshi roadmap W1).
  - **(c) VERSION resolution** — `ai-hwaccel.exe --version`, run from the
    `_bin/` dir, reports the **bundled version** (not `unknown`). Confirms
    the wheel layout's `data_file_path` resolution on PE.
- **Venue:** `windows-latest` ephemeral runner — the roadmap's preferred
  option (self-contained, no secrets, no self-hosted wiring). Fallback to
  an `ssh cass` self-hosted leg if the cross-built PE ever fails to *run*
  cleanly on the GH runner.

#### Notes

- The gate matches the **real CLI surface** — flag-based (`ai-hwaccel` =
  full JSON, `-vv` = trace, `--version`), not the `detect <flag>`
  subcommand the roadmap sketch had assumed.
- **PE limitation surfaced by this work (follow-up, not fixed here):**
  `AI_HWACCEL_DATA_DIR` is **unreadable on PE** — `cmd_getenv`
  (`src/detect/command.cyr`) reads `/proc/self/environ`, which doesn't
  exist on Windows — so the bundled binary's `--version`/`--cost` resolve
  data files **cwd-relative** on Windows, *not* via the env var the Python
  `_runner` sets. Detection (the primary path, which spawns `wmic` and
  needs no data files) is unaffected. The gate runs `--version` from
  `_bin/` to match this cwd-relative reality. A Windows
  `GetEnvironmentVariable` path in `cmd_getenv` is filed as a follow-up so
  the documented env-var contract holds on PE too.

#### Performance

- **CI-only change; binary byte-unchanged.** No `src/` touched (workflow +
  `VERSION` + docs only) — `VERSION` is read at runtime, not compiled in —
  so the compiled binary and every hot path are identical to 2.3.10.
  Benchmark suite re-run for the record: within run-to-run noise, **no
  regression.** 12/12 test units pass.

## [2.3.10] — 2026-06-15

**Toolchain bump to cyrius 6.2.11.** Pin update + stdlib re-sync,
proving no regression. The headline of the 6.1.18 → 6.2.11 jump is a
stdlib reorganization: the standalone `lib/json.cyr` module is gone,
folded (with `base64`/`csv`/`u128`/`bigint`/`toml`/`cyml`) into the new
`lib/bayan.cyr` distribution bundle, and its functions are renamed
`json_*` → `bayan_json_*`. ai-hwaccel never called the stdlib JSON
parser — `src/json_out.cyr` is a hand-rolled `str_builder` serializer
and `src/model_format.cyr` does its own byte-level safetensors header
parse — so the `"json"` entry in `[deps] stdlib` was dead weight that
now points at a module that no longer exists. **Dropped it** rather
than re-pointing at `bayan` (which would pull the whole bundle for
functions we don't use; the compiler DCEs it either way, but the dep
list should reflect reality). No `src/` changes.

#### Changed

- **`cyrius.cyml`**: pin 6.1.18 → **6.2.11**. Stdlib re-synced (97
  files). Removed the unused **`"json"`** entry from `[deps] stdlib`
  (module folded into `bayan.cyr` upstream; nothing in `src/` references
  it).
- **`VERSION`** 2.3.9 → 2.3.10; **`dist/ai-hwaccel.cyr`** regenerated.

#### Performance

- **Benchmark-neutral.** Full suite re-run on the 6.2.11 tree against
  the 6.1.18 baseline, same machine/iteration counts. All moves sit
  inside run-to-run noise: the deterministic sub-µs rows that looked
  like a regression on a single pair of runs (`total_memory_13dev`
  141→148 ns) were confirmed as noise across 3× re-runs each — new
  141–157 ns vs baseline 144–149 ns, overlapping ranges;
  `has_accelerator_13dev` and `count_family_gpu_13dev` likewise overlap.
  The µs-resolution rows (`json_serialize_13dev` ~23 µs,
  `json_plan` ~21 µs, `parse_cuda_8gpu` ~35 µs) are flat within their
  max-spike variance. **No regression.** 12/12 test units pass, 6/6
  fuzz harnesses build, binary smoke-tested.

## [2.3.9] — 2026-06-09

**Windows DXGI precise VRAM + structured logging, on cyrius 6.1.18.** ai-hwaccel previously emitted nothing to stderr: every
detector shelled out to `nvidia-smi` / `wmic` / `rocm-smi` and swallowed
failures into the warnings vec, visible only if you parsed the JSON. This
wires the AGNOS `sakshi` structured logger through the whole pipeline
(detectors, planning, cache, async spans) so the tool is observable in the
field — while keeping **stdout byte-clean** for the JSON that consumers
parse. With the **cyrius 6.1.18** bump the logger now reaches stderr on
**Windows PE** as well (previously dropped — see *Changed*), so the feature
ships on all three platforms.

#### Added

- **Structured logging — `src/log.cyr`** over stdlib `sakshi`. All output
  goes to **stderr (fd 2)**; stdout is untouched. Default level **WARN**
  (quiet on success, visible on failure). Raise via the **`AI_HWACCEL_LOG`**
  env var (`off|fatal|error|warn|info|debug|trace`) or the
  **`--log-level <name>`** / **`-v`** (debug) / **`-vv`** (trace) /
  **`-q`** (off) CLI flags. Process tagged with `getpid()` as the sakshi
  trace id for log correlation. `hwlog_*_n` message builders and span
  wrappers are level-guarded so nothing allocates (and spans don't emit)
  below the active level.
- **Pipeline instrumentation** at operation granularity (never inside a
  per-iteration parse loop): `registry_detect_with_opts` and
  `registry_detect_threaded_with` get a `detect` span + a profiles/warnings
  summary (info); `cached_get` / `disk_cached_get` log hit/miss (debug) and
  disk write-failure (warn); `reg_plan_sharding` logs the model size (debug)
  and warns when no accelerator is available; the `cuda` / `gaudi` CSV parse
  failures — previously silent — now emit a `warn`. Default-level runs stay
  silent on success.

#### Changed

- **Toolchain pin 6.1.5 → 6.1.18** (stdlib re-synced, 94 files; sakshi
  v2.2.6 → **v2.2.10**, new `fs_win.cyr` Windows `dir_list` port). The
  **compiler** bump is a codegen no-op — cycc 6.1.15 and 6.1.18 emit a
  byte-identical 370,776 B Linux binary (same as 6.1.5). The synced
  **stdlib** adds +16 B (370,776 → 370,792 B), entirely sakshi 2.2.10 source
  linked through `log.cyr`. 13/13 test units (606 assertions) pass;
  `fmt`/`lint`/`vet`/
  distlib-determinism clean; bench delta within noise (bench-history.csv).
- **Windows PE structured logging fixed (cyrius 6.1.18).** Logging was
  silently dropped on PE: sakshi holds syscall numbers in `var` slots and
  cyrius's PE syscall reroute only fired for compile-time-literal numbers,
  so `syscall(_SK_SYS_WRITE, …)` fell through to a non-functional raw
  `0F 05` (no fault, exit 0). 6.1.18 resolves a `var`-held syscall number to
  its constant, so `n=1` (write) routes to `WriteFile`. **Verified on cass**
  (Windows 10.0.26200): `detect -vv` emits the full `detect` span on stderr;
  default WARN stays silent; stdout stays byte-clean. (`getpid` is still
  unrouted on PE, so the trace-id prefix shows `[0]` — cosmetic; delivery is
  unaffected.)
- **Windows wheel cross-build fixed (CI).** `stage_win_cross.sh` now builds
  the PE with **`cyrius build --win`** (the Linux ELF `cycc` emits PE32+
  natively) instead of piping a hand-synthesized translation unit to the
  standalone `cycc_win`. `cycc_win` is itself a **Windows PE**, so on a Linux
  host it only runs under Wine (the local `DOSWin` binfmt handler); a bare CI
  runner (`ubuntu-latest`, no Wine) failed it with `cannot execute binary
  file: Exec format error` (exit 126). `--win` is the correct Linux-native
  cross path — no Wine, deps resolved by the wrapper. PE re-verified on cass
  (`--version` 2.3.9, DXGI JSON, `-vv` logging).
- **`cyrius.cyml`**: **`sakshi`** added to `[deps] stdlib`; **`src/log.cyr`**
  added to `[lib] modules` (so the consumer bundle carries it).
- **Bundle consumers** (mihi et al.): `dist/ai-hwaccel.cyr` now references
  `sakshi_*`, so consumers must add **`sakshi`** to their own
  `[deps] stdlib`. The bundle's "unresolved symbols" note at distlib time
  is expected — stdlib is consumer-supplied.

#### Performance

- **Logging adds nothing measurable.** Full-instrumentation tree vs
  6.0.70-without-logging, interleaved: `parse_cuda_8gpu` min **28 µs**
  (identical), `total_memory_13dev` 139→140 ns, `has_accelerator_13dev`
  28→28 ns, `count_family_gpu_13dev` 292→281 ns, `json_serialize_13dev`
  ~22–23 µs — **all within noise.** Instrumentation lives at dispatch
  sites and error paths, off the benched hot loops; below-threshold log
  calls build no messages. Binary grows **301 KB → 369 KB** (+68 KB, the
  reachable `sakshi` surface; DCE NOPs 623 unreachable fns). At the 6.1.18
  pin the stripped binary is **370,792 B**; 13/13 test units (606
  assertions) pass.

## [2.3.8] — 2026-06-05

**Toolchain bump to cyrius 6.0.70.** Pin update + stdlib re-sync, proving
no regression. Groundwork for the 2.3.9 Windows DXGI precise-VRAM work:
6.0.70 lands the *foundation* (`callptr`/`IR_CALL_INDIRECT`, the
`dxgi.dll!CreateDXGIFactory1` import, COM-vtable dispatch capability —
`CreateDXGIFactory1` returns `S_OK` on cass), but `callptr` to a *real*
Win64 COM callee (`EnumAdapters`/`GetDesc` over the DXGI vtable — the
actual VRAM read) corrupts the caller frame on cass and is fixed upstream
only in **6.0.71** (cyrius issue
`2026-06-05-windows-com-vtable-real-callee-frame-corruption.md`). DXGI
precise VRAM therefore stays deferred to 2.3.9; Windows GPU VRAM remains
on the WMI `AdapterRAM` path.

#### Changed

- **`cyrius.cyml`**: pin 6.0.54 → **6.0.70**. Stdlib re-synced (89 files).
  **CLAUDE.md** pin → 6.0.70.
- **`VERSION`** 2.3.7 → 2.3.8; **`dist/ai-hwaccel.cyr`** regenerated.

#### Performance

- **cyrius 6.0.64 global allocator spinlock — accepted, justified.**
  6.0.70's `lib/alloc.cyr` serializes the bump pointer behind a CAS
  spinlock (a heap-corruption fix: concurrent `alloc()` across real
  threads was overlap-allocating). This costs ~5–10 % on allocation-heavy
  paths — `parse_cuda_8gpu` min **16→28 µs** is the worst case. **It is not
  a regression to fix but a correctness fix we need**: `async_detect.cyr`
  spawns real threads (`thread_create`/`thread_join`) that all `alloc()`
  while parsing detector output; without the lock that path corrupts the
  heap. Single-threaded callers pay the tax; the threaded path stops being
  unsound. Accepted per the no-regression rule's "explicitly justified"
  clause. Otherwise neutral: `total_memory_13dev` 137→139 ns,
  `has_accelerator_13dev` 27→28 ns, `json_serialize_13dev` ~22 µs flat —
  **no other regression.** 12/12 test units pass.

## [2.3.7] — 2026-06-03

**Windows x86_64 wheel ships with real CPU + GPU detection —
`pip install ai-hwaccel` is now self-contained on Windows.** Closes the
roadmap 2.3.7 gate. The Windows PE blockers were fixed upstream (cyrius
6.0.50 unfroze `cycc_win`; 6.0.51 routed Win32 process creation via
CreateProcessW), and this release implements the Windows-side detection
that turns a running-but-blind binary into a useful one. Verified
end-to-end on `cass` (Windows 11): real total RAM + the actual GPU
(Intel UHD Graphics 600) are detected.

#### Added

- **Windows x86_64 wheel**
  (`ai_hwaccel-2.3.7-py3-none-win_amd64`). **Cross-built on Linux** —
  `cycc_win` is a Linux-hosted compiler that emits PE32+, so no Windows
  runner is needed (cass is used only for runtime smoke). New
  `bindings/python/scripts/stage_win_cross.sh` synthesizes the
  translation unit (`[deps] stdlib` + `src/main.cyr`) and pipes it to
  `cycc_win`; `wheels.yml` `windows` job **enabled**, now on
  `ubuntu-latest`.
- **Windows GPU detection** — `src/detect/windows.cyr` (was a stub) now
  spawns `wmic path win32_VideoController` and emits one **`ACCEL_WIN_GPU`**
  ("Windows GPU") profile per controller with device name + VRAM. New
  `BACKEND_WINDOWS` backend (gated into `registry_detect_with` under
  `#ifdef CYRIUS_TARGET_WIN`). The Unix detectors still run, so
  `nvidia-smi.exe` continues to give precise NVIDIA VRAM where present.
- **Windows CPU RAM detection** — `detect_system_memory()` gained a
  `#ifdef CYRIUS_TARGET_WIN` branch spawning `wmic computersystem get
  TotalPhysicalMemory`, replacing the 16 GiB fallback (`/proc/meminfo`
  doesn't exist on Windows) with real total memory.
- **`tests/tcyr/windows_test.tcyr`** — Linux-hosted fixture tests for the
  pure parsers (`win_parse_videocontrollers`, `win_parse_total_memory`):
  single/multi GPU, CRLF, key-order, empty, absent. The parsers are
  ungated so they're testable without a Windows host. 12 test units total.

#### Changed

- **`cyrius.cyml`**: pin 6.0.43 → **6.0.54**. Stdlib re-synced (87 files).
- **`build_wheel.sh`** clears `build/` before packaging, and
  `stage_*.sh` drop the foreign-platform binary, so a per-platform wheel
  bundles only its own binary (a Linux ELF was leaking into the
  win_amd64 wheel via setuptools' build cache).
- **`pyproject.toml`** 2.3.6 → 2.3.7; **`VERSION`** 2.3.6 → 2.3.7;
  **`dist/ai-hwaccel.cyr`** regenerated; **`CLAUDE.md`** pin → 6.0.54.

#### Performance

No hot-path `.cyr` changed (the new code is Windows-gated + the keyed
`accel_*` functions gained one branch each). 6.0.54 + feature vs 6.0.47
baseline, same box/iters: `total_memory_13dev` 141 vs 143,
`count_family_gpu_13dev` 300 vs 303, `has_accelerator_13dev` 28 vs 28,
`json_serialize_13dev` ~23 µs flat — **neutral, no regression.**

#### Known limits / follow-ups

- **VRAM caps at 4 GiB on non-NVIDIA GPUs** —
  `Win32_VideoController.AdapterRAM` is a 32-bit field. Native DXGI
  (`EnumAdapters1`, 64-bit `DedicatedVideoMemory`, no subprocess) is the
  2.3.8 precision upgrade, gated on cyrius PE COM-vtable + dxgi.dll IAT
  support (filed:
  `cyrius/docs/development/issues/2026-06-03-windows-pe-com-vtable-dxgi-for-gpu-enum.md`).
- On Windows the irrelevant Unix probes (`system_profiler`, etc.) still
  emit "tool not found" warnings — cosmetic; keeping the probes enabled
  is what lets `nvidia-smi.exe` work when present.

## [2.3.6] — 2026-06-02

**macOS arm64 wheel ships — `pip install ai-hwaccel` now self-contained
on Apple silicon.** This closes the roadmap 2.3.6 gate. Two things had
to land together: the toolchain pin moves to **6.0.43**, and the Darwin
build bug that blocked the wheel on 6.0.38 is fixed upstream.

The 2.3.5 notes left macOS gated on "no compiler in the installer."
6.0.38 delivered the arm64 Darwin compiler but surfaced a *new* blocker:
`cyrius build` false-negatived its own install check on a manifest with
`cyrius = "<pin>"` + a `[deps] stdlib` block (our exact shape), claiming
the present snapshot lib was "not installed." Filed as cyrius issue
`2026-06-02-macos-arm64-deps-stdlib-pin-check.md` (three stacked
Darwin-ABI defects: an `is_dir` install-probe false-negative, a getcwd
SIGSYS, and a wrong Darwin `st_size` offset truncating the include
scan). **Fixed across cyrius 6.0.40–6.0.43**; verified end-to-end on
`ecb` (real Apple silicon): the arm64 Mach-O builds, the resolved stdlib
executes, and every ai-hwaccel subcommand runs.

#### Added

- **macOS arm64 wheel** (`ai_hwaccel-2.3.6-py3-none-macosx_11_0_arm64`).
  Built on `ecb` via `bindings/python/scripts/build_remote.sh ecb
  macosx_11_0_arm64`, bundling the Mach-O arm64 binary + `VERSION` +
  `data/cloud_pricing.json` under `ai_hwaccel/_bin/` (subprocess + JSON,
  no FFI — same model as the Linux wheels). Verified: `--version`,
  `--json`, `--summary`, `--cost`, `--plan` all run on Darwin arm64.
- CI `wheels.yml` `macos` job **enabled** (`if: true`) — the `macos-14`
  runner installs pinned 6.0.43 and builds natively.

#### Changed

- **`cyrius.cyml`**: pin 6.0.30 → **6.0.43**. Stdlib re-synced into
  `./lib/` (82 files from the 6.0.43 snapshot); drift warning gone.
- **`bindings/python/scripts/build_remote.sh`**: `cyrius lib sync` is now
  best-effort. It false-negatives on Darwin (the directory-listing /
  `getdents64` surface is still unported there — a separate, still-open
  item in the same cyrius issue), and it is unnecessary: `cyrius build`
  resolves `[deps] stdlib` into `./lib` by name (the path fixed in
  6.0.40+). The build populates its own lib.
- **`bindings/python/scripts/stage_binary.sh`**: fixed an "unbound
  variable" crash on the `macos-14` CI runner — the native staging path
  leaves `EXTRA_ARGS` empty, and bare `"${EXTRA_ARGS[@]}"` is a `set -u`
  error under the runner's bash 3.2. Switched to
  `${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}` (expands to nothing on an empty
  array, no spurious empty argv). Verified both the native and
  `--aarch64` paths.
- **`bindings/python/pyproject.toml`**: 2.3.4 → 2.3.6.
- **`CLAUDE.md`**: pinned-version notes 6.0.0/6.0.30 → 6.0.43.
- **`VERSION`**: 2.3.5 → 2.3.6; **`dist/ai-hwaccel.cyr`** regenerated
  (embeds 2.3.6).

#### Performance

No `.cyr` source changed; any delta is codegen-only via the re-synced
stdlib. Built on 6.0.43 vs the 6.0.38 lib snapshot, same machine and
iteration counts. The deterministic ns rows drift run-to-run on this box
(`total_memory_13dev` 145–163 ns, `count_family_gpu_13dev` 303–354 ns,
`has_accelerator_13dev` 30–42 ns) and the 6.0.38 baseline (159 / 354 /
33 ns) sits inside that band — **neutral, within noise, no regression.**
µs-level JSON/parse paths flat within jitter. (This subsumes the
unreleased 6.0.30 → 6.0.38 pin step, which was also neutral.)

#### Compatibility

- 11/11 test units green on 6.0.43 (Linux). macOS arm64 verified by
  running the staged binary on `ecb`.
- **Known cosmetic warning (macOS only):** the Darwin `[deps]` build
  emits `duplicate fn 'arena_new' (last definition wins)` because both
  `lib/alloc.cyr` and `lib/alloc_macos.cyr` define it and Darwin pulls
  both layers. "Last definition wins" selects the macOS variant —
  correct behavior; the binary is verified working. Filed upstream (P4)
  at `cyrius/docs/development/issues/2026-06-02-macos-alloc-arena-duplicate-fns.md`;
  not a consumer blocker.
- **Still pending:** Windows wheel (2.3.7) — needs the PowerShell build
  flow + a PE `cyrius build` target.

## [2.3.5] — 2026-06-01

**Toolchain pin 6.0.25 → 6.0.30.** Resolves the long-standing wrapper
drift (the local toolchain had auto-advanced well past the pin). Pure
toolchain bump — no `.cyr` source changed. Per the mandatory-benchmark
policy, before/after deltas confirm no regression.

#### Changed

- **`cyrius.cyml`**: pin 6.0.25 → 6.0.30. Stdlib re-synced into `./lib/`
  (82 files from the 6.0.30 snapshot). Build drift warning gone.
- **`VERSION`**: 2.3.4 → 2.3.5.
- **`dist/ai-hwaccel.cyr`** regenerated (embeds 2.3.5).
- **`CLAUDE.md`**: pinned-version note 6.0.25 → 6.0.30.

#### Performance

Before/after (min-of-6 @ 2000 iters, ns; built with the respective
pinned toolchain):

| benchmark              | 6.0.25    | 6.0.30    | delta |
| ---------------------- | --------: | --------: | ----- |
| `json_serialize_13dev` | 25535 ns  | 25245 ns  | −1.1% (noise) |
| `json_summary_13dev`   |  5051 ns  |  5025 ns  | flat (noise) |
| `parse_cuda_8gpu` (min)|    18 µs  |    17 µs  | flat (noise) |

No regression; all within run-to-run noise. The compiled binary is
functionally unchanged.

#### Compatibility

- **Gates** (on 6.0.30): 11 test units, 6 fuzz harnesses, `vet` (37
  deps, 0 untrusted), fmt, lint, raw-offset guard, distlib
  drift+determinism — all green.
- **Note**: macOS toolchain still incomplete — the 6.0.30 installer
  sets up the version layout + `cyriusly` on Darwin but does not yet
  deliver the `cyrius`/`cycc` compiler binaries, so the macOS wheel
  (2.3.6) remains blocked. Windows wheel (2.3.7) still pending the
  PowerShell build flow.

## [2.3.4] — 2026-06-01

**Linux wheels + the wheel-build machinery.** `bindings/python` can now
produce platform-tagged wheels that bundle a prebuilt, statically-linked
`ai-hwaccel` binary (+ `VERSION` + `data/`), so `pip install` gives a
self-contained package (subprocess + JSON; no FFI). Linux x86_64 +
aarch64 ship and are validated. **macOS and Windows are deferred to 2.3.5,
gated on cyrius toolchain support** (see below) — no `.cyr` changed, so
the core binary is identical to 2.3.3.

#### Added

- **Wheel machinery** (`bindings/python/`):
  - `setup.py` — `BinaryDistribution` + a `bdist_wheel` override that
    emits a `py3-none-<platform>` wheel (python-agnostic, platform-
    specific) with the tag overridable via `AIH_WHEEL_PLAT`.
  - `scripts/build_wheel.sh <plat-tag>` — package the staged `_bin/`.
  - `scripts/build_remote.sh <host> <plat-tag>` — ship source to an SSH
    host, `lib sync` + build there, retrieve the binary (for platforms
    with no local cross-compiler, e.g. macOS once supported).
  - `pyproject` package-data now bundles `_bin/{ai-hwaccel[.exe],
    VERSION, data/*}`.
- **`.github/workflows/wheels.yml`** — CI matrix (workflow_dispatch +
  tag): linux (x86_64 + aarch64) active; macOS / Windows jobs scaffolded
  but **gated `if: false`** pending toolchain support.

#### Shipped wheels

- `ai_hwaccel-2.3.4-py3-none-manylinux2014_x86_64.whl` — built,
  installed in a clean venv, and run from an unrelated cwd:
  `version`/`detect`/`cost` all work, bundled binary preserved `+x`
  (0o755), data resolved via `AI_HWACCEL_DATA_DIR`.
- `ai_hwaccel-2.3.4-py3-none-manylinux2014_aarch64.whl` — cross-built
  (static ARM binary) and packaged (run-validated in CI/on-device only).

#### Deferred (gated on cyrius toolchain)

- **macOS arm64 wheel (2.3.5, near-term)** — the Darwin binaries are
  built but not yet integrated into `install.sh` (it still rejects
  darwin). Unblocks the moment the installer fix lands.
- **Windows x86_64 wheel (2.3.6, later in 6.0.x)** — needs a full
  PowerShell (`.ps1`) build flow; the PE backend (`cycc_win`) is frozen
  at `cc5_win 5.11.69` and there's no `cyrius build --win` target.
  Both CI jobs are scaffolded and flip on with a one-line `if:` change
  when their toolchain support lands — same "gate on upstream readiness"
  posture as the WASM/JS roadmap item.

#### Changed

- **`VERSION`**: 2.3.3 → 2.3.4; Python package + `__version__` track it.
- **`dist/ai-hwaccel.cyr`** regenerated (embeds 2.3.4); no `.cyr` source
  changed, so the binary is otherwise identical to 2.3.3.

#### Compatibility

- **cyrius core / CLI**: unchanged. No benchmarked path touched (no
  `.cyr` edits) — bench policy satisfied trivially.

## [2.3.3] — 2026-06-01

**Working-directory-independent data files.** Closes the 2.3.2 known
limitation: the binary read `VERSION` (`--version`) and
`data/cloud_pricing.json` (`--cost`) relative to the current working
directory, so a binary invoked from elsewhere (a pip-installed wheel)
returned `"unknown"` / no cost recommendations. The binary now honors an
`AI_HWACCEL_DATA_DIR` env var to locate them, and the Python wrapper sets
it automatically for the bundled binary. Cross-platform and
language-neutral — any consumer (agnos later) sets one env var.
Multi-platform wheel CI is 2.3.4.

#### Added

- **`AI_HWACCEL_DATA_DIR` resolution** (`src/detect/command.cyr`
  `data_file_path`). When set, `VERSION` and `data/cloud_pricing.json`
  resolve under it (`<dir>/<rel>`); unset → cwd-relative (unchanged
  fallback). Used by `_print_version` (`src/main.cyr`) and
  `load_cloud_instances` (`src/cost.cyr`).
- **`bindings/python` wheel-bundling prep**:
  - `_runner._run` sets `AI_HWACCEL_DATA_DIR` to the bundled `_bin/` dir
    when the bundled binary is used (never overriding a caller's value;
    PATH/explicit binaries are left to the caller's environment).
  - `scripts/stage_binary.sh` — builds the binary with the pinned
    toolchain and stages `_bin/{ai-hwaccel, VERSION, data/cloud_pricing.json}`
    (`--aarch64` supported). Foundation for the 2.3.4 wheel build.
  - `tests/test_bundled.py` — 3 tests proving `version()` / `cost()` work
    from a foreign cwd via the bundled binary (skip if not staged).
    Python suite now 18 tests.

#### Fixed

- `version()` / `cost()` no longer depend on the caller's working
  directory when using the bundled binary (the 2.3.2 known limitation).

#### Changed

- **`VERSION`**: 2.3.2 → 2.3.3; Python package + `__version__` track it.
- **`dist/ai-hwaccel.cyr`** regenerated (embeds 2.3.3) by the now
  dist-aware `scripts/version-bump.sh`.

#### Performance

- No benchmarked path touched — `data_file_path` is on the
  `--version` / `--cost` startup paths only, not the JSON/parse hot
  paths. `json_serialize_13dev` 25433 → 24520 ns (within noise),
  confirming no regression (min-of-5 @ 2000 iters).

#### Compatibility

- **CLI**: additive. With `AI_HWACCEL_DATA_DIR` unset, behavior is
  byte-identical to 2.3.2 (cwd-relative). New env var is opt-in.
- **Next**: 2.3.4 — multi-platform wheel CI (manylinux x86_64/aarch64,
  macOS universal2, Windows) using `stage_binary.sh` per target.

## [2.3.2] — 2026-06-01

**Python bindings (`bindings/python/`).** A thin, dependency-free Python
package over the compiled binary + schema-v4 JSON contract from 2.3.1.
There is no FFI (the cyrius toolchain emits executables only); each call
shells out to the binary and parses its JSON into typed dataclasses.
**No cyrius source changed** — the binary is byte-identical to 2.3.1, so
the core benchmarks are unaffected (verified by rebuild; the
mandatory-benchmark policy is satisfied trivially with no core perf
surface touched). Multi-platform wheels are 2.3.3.

#### Added

- **`bindings/python/` package** (`ai_hwaccel`, GPL-3.0-only,
  zero runtime deps, Python ≥ 3.8):
  - **Typed model** (`models.py`) for the full schema-v4 surface:
    `Registry`, `AcceleratorProfile`, `SystemIo`, `Interconnect`,
    `StorageDevice`, `RuntimeEnvironment`, `ShardingPlan`, `ModelShard`,
    `TrainingMemory`, `CostReport`, `CostRecommendation`. Lossless field
    mapping; fixed-point `*_x1000` values surfaced via convenience
    properties (`est_tokens_per_sec`, `total_gib`, `price_per_hour_usd`).
  - **API**: `detect()`, `summary()`, `plan()`, `training_memory()`,
    `cost()`, `version()` — each accepts `binary=` and `timeout=`.
  - **Binary discovery** (`_runner.py`): explicit arg → `AI_HWACCEL_BIN`
    → bundled `_bin/` (2.3.3) → `PATH`. Clear `BinaryNotFoundError` /
    `CommandError`.
  - **Optional pandas export**: `Registry.to_dataframe()` (extra
    `ai-hwaccel[pandas]`); clean `ImportError` when pandas is absent.
  - **`pyproject.toml`** (setuptools, src layout; `pandas` + `test`
    extras), package **README**, and a **`unittest`** suite: 9
    fixture-based model tests + 6 e2e tests against the real binary
    (15 total, all passing locally).
- **`.gitignore`**: Python artifacts (`__pycache__/`, `*.pyc`,
  `*.egg-info/`, `bindings/python/{dist,build}/`, the build-time
  `_bin/`).

#### Known limitations (tracked for 2.3.3)

- The binary reads `VERSION` and `data/cloud_pricing.json` **relative to
  the current working directory**. Run from outside the repo root,
  `version()` returns `"unknown"` and `cost()` yields no recommendations;
  detection is unaffected. The 2.3.3 packaging work bundles these next to
  the binary and resolves them relative to the executable.
  `ai_hwaccel.__version__` always reflects the package version.

#### Changed

- **`VERSION`**: 2.3.1 → 2.3.2 (Python package version tracks it).
- **`dist/ai-hwaccel.cyr`** regenerated so its embedded `# Version:`
  header matches 2.3.2 (content otherwise unchanged from 2.3.1).

#### Fixed

- **CI "distlib drift" gate on version bumps.** The dist bundle embeds
  the project version in its header, so a `VERSION` bump alone makes it
  stale even with no `.cyr` change — CI regenerates with the new version
  and the diff fails. `scripts/version-bump.sh` now **auto-regenerates
  `dist/ai-hwaccel.cyr`** using the *pinned* toolchain
  (`~/.cyrius/versions/<pin>/bin/cyrius`, matching what CI installs)
  after writing `VERSION`. Confirmed distlib output is byte-identical
  across cyrius 6.0.25/6.0.27, so the local wrapper drift is not a
  factor — only the missing regen was.

#### Compatibility

- **cyrius core / CLI**: unchanged (no `.cyr` edits; the regenerated
  bundle differs from 2.3.1 only in the version header line).
- **Next**: 2.3.3 — multi-platform wheels (manylinux x86_64/aarch64,
  macOS universal2, Windows), bundling the per-target binary + data
  files; CI matrix.

## [2.3.1] — 2026-06-01

**JSON surface extension — the data layer for language bindings.** The
roadmap's "Ecosystem" work (Python bindings + packaging) needs the full
detection surface reachable as JSON; today only `AcceleratorProfile`s
were serialized. This release bumps the JSON schema to **v4** and makes
`SystemIo`, `Interconnect`, `StorageDevice`, runtime environment,
`ShardingPlan`, and the training-memory estimate reachable as JSON, plus
new CLI modes to emit them. The compiled binary + this JSON contract is
the **language-neutral substrate** the Python package (2.3.2) and a
later AgnosAI/agnos-kernel target consume — no consumer-specific logic
in the core. CLI text output is unchanged and backward-compatible.

Per the mandatory benchmarking policy, the before/after deltas are in
the table below and `bench-history.csv`.

#### Added

- **`system_io` in the default registry JSON (schema v3 → v4).**
  `registry_to_json` now appends a `"system_io"` object:
  `interconnects[]` ({kind, name, bandwidth_bytes_per_sec, state}),
  `storage[]` ({name, kind, bandwidth_bytes_per_sec}), and
  `environment` ({is_docker, is_k8s, k8s_namespace, cloud_provider,
  instance_type, region, k8s_gpu_count, k8s_gpu_source} or `null`).
  Bandwidth is normalized to bytes/sec uniformly (interconnects store
  GB/s×1000, storage stores MB/s; both → bytes/sec via ×1e6).
  Serializers: `system_io_to_json` + `_ic_to_json` / `_storage_to_json`
  / `_env_to_json` in `src/json_out.cyr`.
- **`--plan <model>`** → `plan_to_json` of `reg_plan_sharding`:
  {strategy, strategy_count, total_memory_bytes,
  est_tokens_per_sec_x1000 (optional), shards[] {id, layer_start,
  layer_end, device, device_id, memory_bytes}}.
- **`--train <model> [--method <m>]`** → `training_to_json` of
  `estimate_training_memory`: model/optimizer/activation/total emitted
  as both `*_bytes` and the lossless `*_gib_x1000` fixed-point. Default
  method `full`, target GPU. New `training_method_from_str` in
  `src/training.cyr`.
- **`--cost <model> --json`** → `cost_to_json` (`src/cost.cyr`):
  {model, quantization, memory_required_bytes, recommendations[]
  {instance, provider, gpu, gpu_count, total_memory_gb,
  price_per_hour_usd_x100}}. Without `--json`, `--cost` prints the same
  text as before.
- **CLI help** updated for the new flags; shared `_quant_arg` /
  `_parse_model_b` helpers in `src/main.cyr`.
- **Tests**: `tests/tcyr/json_output_test.tcyr` +4 cases (system_io
  populated + null-env, plan, training) — 21 → 36 assertions.
- **Benchmarks**: `benches/registry.bcyr` gains `json_system_io`,
  `json_plan`, `json_training` (ns-resolution, 2000 iters).

#### Performance

Mandatory before/after (min-of-7 @ 2000 iters, ns resolution):

| benchmark              | 2.3.0     | 2.3.1     | delta |
| ---------------------- | --------: | --------: | ----- |
| `json_serialize_13dev` | 24265 ns  | 25433 ns  | +4.8% — inherent cost of the always-present (empty) `system_io` object, not a regression |
| `json_summary_13dev`   |  4947 ns  |  4985 ns  | flat (noise) — summary JSON unchanged |
| `json_system_io` (new) |     —     |  7167 ns  | populated: 2 interconnects + 2 storage + env |
| `json_plan` (new)      |     —     | 21299 ns  | mock 70B sharding plan |
| `json_training` (new)  |     —     |  4069 ns  | 70B full-train estimate |

The `json_serialize_13dev` increase is the new feature's payload
(`system_io` is now always serialized), measured and documented rather
than hidden. All other registry/parsing benches untouched.

#### Changed

- **`VERSION`**: 2.3.0 → 2.3.1.
- **`SCHEMA_VERSION`**: 3 → 4 (`src/units.cyr`); `foundation_test`
  assertion updated.
- **`dist/ai-hwaccel.cyr`** — regenerated (json_out/cost/training in the
  bundle); deterministic.

#### Compatibility

- **CLI binary**: additive. Default JSON gains `system_io` +
  `schema_version: 4`; existing keys unchanged. `--cost` text output is
  byte-identical without `--json`. New flags are opt-in.
- **Library consumers**: the bundle exposes the new serializers; unused
  ones DCE away. Bump `[deps.ai-hwaccel] tag = "2.3.1"`, re-run
  `cyrius lib sync` + `cyrius deps`.
- **Next**: 2.3.2 Python package (`bindings/python/`), 2.3.3
  multi-platform wheels.

#### Aside

- Toolchain pin drift: `cyrius.cyml` pins 6.0.25, wrapper is now 6.0.26.
  Per policy that's its own bench-gated point release; left at 6.0.25
  here (builds run with `CYRIUS_NO_WARN_PIN_DRIFT=1`).

## [2.3.0] — 2026-06-01

**Toolchain modernization + serialization hot-path + a codebase-wide
Str→cstr dedup.** Pins the compiler to cyrius 6.0.25 (was 6.0.0),
re-syncs the vendored stdlib snapshot, and lands two pieces of audit
work: the JSON serializer now emits single-byte structural punctuation
via `str_builder_putc` (a real, benchmarked win on the per-profile
path), and the `alloc + memcpy + NUL` "copy a `Str` slice into an owned
C string" idiom that had been hand-inlined across 11 detectors is
consolidated onto the stdlib `str_cstr`. No CLI behavior change; JSON
output is byte-identical (all 21 `json_output_test` assertions pass
unchanged).

Per the development loop, every claim below is backed by a
before/after benchmark — see the delta table.

#### Changed

- **Toolchain pin: cyrius 6.0.0 → 6.0.25** (`cyrius.cyml`). The wrapper
  was already 6.0.25 (the manifest pin had drifted); this aligns the
  manifest, silences the `toolchain drift` build warning, and makes
  `cyrius lib sync` resolve against the installed snapshot. Stdlib
  re-synced into `./lib/` (82 files) from
  `~/.cyrius/versions/6.0.25/lib/`.
- **`VERSION`**: 2.2.6 → 2.3.0.
- **`dist/ai-hwaccel.cyr`** — regenerated by `cyrius distlib`. Shrinks
  by 76 lines (the detector dedup below); deterministic across re-runs.

#### Performance

- **JSON serialization hot path** (`src/json_out.cyr`). Structural
  punctuation (`{` `}` `[` `]` `,` `"` `:`) now goes through
  `str_builder_putc` — a single `store8` — instead of
  `str_builder_add_cstr`, which pays a `strlen` + `memcpy` round-trip
  per call even for a one-byte literal. Variable-length pieces (keys,
  values, the multi-byte `,"profiles":[` / `true` / `false` literals)
  stay on `add_cstr`. On the 13-device registry benchmark
  (`benches/registry.bcyr`, 2000 iters, min-of-6 at nanosecond
  resolution):

  | benchmark              | 2.2.6 (add_cstr) | 2.3.0 (putc) | delta   |
  | ---------------------- | ---------------: | -----------: | ------- |
  | `json_serialize_13dev` |        26946 ns  |    24602 ns  | **−8.7%** |
  | `json_summary_13dev`   |         5335 ns  |     5297 ns  | −0.7% (noise) |

  The win concentrates on `json_serialize_13dev` because it walks a
  13-element profile array × ~13 fields each — hundreds of single-byte
  appends. `json_summary_13dev` (8 scalar fields, no nested array) has
  little single-byte punctuation to save, so it lands within run-to-run
  noise — reported honestly rather than rounded up.

#### Refactored

- **`Str` → owned C string consolidated onto `str_cstr`** across 11
  detectors (16 call sites). Every site previously hand-wrote
  `var d = str_data(s); var n = str_len(s); var c = alloc(n + 1);
  memcpy(c, d, n); store8(c + n, 0);` — the exact body of the stdlib
  `str_cstr(s)`. Replaced with a single `str_cstr` call, dropping the
  now-dead `str_data` / `str_len` locals:
  - `src/detect/cuda.cyr` — compute_cap, driver_version, device_name (×3)
  - `src/detect/rocm.cyr` — product_name, vbios_version, revision (×3)
  - `src/detect/apple.cyr` — parsed chip name
  - `src/detect/gaudi.cyr` — hl-smi CSV name field (comment updated)
  - `src/detect/vulkan.cyr` — device name
  - `src/detect/intel.cyr` — device name
  - `src/detect/bandwidth.cyr` — sysfs device path
  - `src/detect/interconnect.cyr` — interconnect name + state (×2)
  - `src/detect/environment.cyr` — k8s namespace + AWS instance type (×2)
  - `src/detect/pcie.cyr` — sysfs device path
  - `src/detect/disk.cyr` — block device name
  Behavior is identical (same allocator, same bytes). The DCE build
  shrinks 289584 → 287992 bytes (−1592). Perf-neutral on the parsing
  benchmarks (`parse_cuda_8gpu` min holds at 16–17 µs); confirmed not a
  regression rather than claimed as a win.

#### Tooling

- **`benches/registry.bcyr`** — `json_serialize_13dev` /
  `json_summary_13dev` bumped 100 → 2000 iters and now print an
  explicit `avg_ns=` line. `bench_report`'s human-readable output
  truncates to microseconds, which hides a sub-microsecond delta; the
  raw nanosecond average makes the serialization win measurable.

#### Compatibility

- **CLI binary**: no change — same flags, byte-identical JSON / table /
  summary output.
- **Library consumers (mihi et al.)**: source-compatible. The bundle
  regenerates smaller but exposes the same symbols; consumers DCE
  unused detectors as before. Bump `[deps.ai-hwaccel] tag = "2.3.0"`
  and re-run `cyrius lib sync` + `cyrius deps`.
- **Gates**: 11 test units (518 assertions), 6 fuzz harnesses,
  `cyrius vet` (37 deps, 0 untrusted), raw-offset guard, distlib
  drift + determinism — all green.

## [2.2.6] — 2026-05-19

**Library-consumer follow-ups from mihi 0.4.0 integration.** Three
gaps surfaced when `mihi` shipped its M3 GPU probe against the 2.2.5
no-exec API: four detectors weren't populating `device_name` (so
mihi reported `(unnamed)` for any ROCm / TPU / Gaudi / Neuron card),
and `cache.cyr`'s disk-write path referenced `registry_to_json` from
the bundle-excluded `json_out.cyr`, leaking a persistent linker
warning into every consumer build. This release closes all four
name gaps and brings `json_out.cyr` into the bundle so the dangling
reference resolves. Pure consumer ergonomics — no API changes.

#### Added

- **`src/json_out.cyr` is now in the bundle.** Previously excluded
  with the rationale "library consumers build their own
  serialization", but `cache.cyr::disk_cached_get_or_detect` calls
  `registry_to_json` unconditionally on the write path. Excluding
  the symbol left every library consumer with an `undefined function
  'registry_to_json'` linker warning (DCE elides the call, so the
  binary was always correct — but the noise was real). Library
  consumers that don't need JSON output still DCE the entire
  serializer; binary size on `cyrius build src/main.cyr build/...`
  is unchanged at 2.2.6.
- **README "Architecture" comment**: `json_out.cyr` reclassified
  from "CLI-only" to "available to library consumers".

#### Fixed

- **`detect_rocm` populates `profile_device_name`** (`src/detect/rocm.cyr`).
  Tries `/sys/class/drm/cardN/device/product_name` first (newer
  amdgpu kernels expose this on some discrete cards); falls back to
  synthesizing `"AMD Radeon (PCI 0x<vendor>:0x<device>)"` from the
  always-present `vendor` + `device` sysfs files. Worst case (both
  unreadable) the string is `"AMD Radeon"`. On archaemenid (Ryzen
  5800H iGPU, PCI 1002:1638) the CLI now reports
  `AMD Radeon (PCI 0x1002:0x1638)` where 2.2.5 reported nothing.
- **`detect_tpu` populates `profile_device_name`** (`src/detect/tpu.cyr`).
  Builds `"Google TPU <version>"` (v4 / v5e / v5p) via the existing
  `tpu_version_name` helper.
- **`detect_gaudi` populates `profile_device_name`** (`src/detect/gaudi.cyr`).
  Prefers the hl-smi CSV `name` field (`HL-225` / `HL-325`) when
  present; falls back to `"Intel Gaudi2"` / `"Intel Gaudi3"` via
  `gaudi_gen_name`. CSV field is copied to a null-terminated buffer
  (same `alloc + memcpy + NUL` pattern apple.cyr already uses).
- **`detect_neuron` populates `profile_device_name`** (`src/detect/neuron.cyr`).
  Both the `neuron-ls --json-output` parser and the `/dev/neuron*`
  sysfs fallback now set `"AWS Inferentia"` / `"AWS Trainium"` via
  `neuron_chip_name`.

#### Changed

- **`VERSION`**: 2.2.5 → 2.2.6.
- **`dist/ai-hwaccel.cyr`** — regenerated by `cyrius distlib`.
  Grows slightly with the bundled `json_out.cyr` (deterministic
  across re-runs).
- **`cyrius.cyml [lib].modules`** — appends `"src/json_out.cyr"`.
  Comment updated to call out the 2.2.6 inclusion + rationale.

#### Compatibility

- **CLI binary**: no change. Same output formatting, same flags;
  the per-detector name strings flow through the existing JSON /
  table / summary paths. All 11 test units (518 assertions) still
  pass.
- **2.2.4 / 2.2.5 binary consumers**: zero impact.
- **2.2.5 library consumers (mihi 0.4.0)**: source-compatible. After
  bumping `[deps.ai-hwaccel] tag = "2.2.6"` and re-running
  `cyrius deps`, mihi's `mihi_gpu_name(0)` returns the populated
  string for ROCm/TPU/Gaudi/Neuron where it returned 0 (null) on
  2.2.5. The `undefined function 'registry_to_json'` warning is
  gone.

## [2.2.5] — 2026-05-19

**No-exec detection contract — `mihi`-shaped consumers can call the
detection surface without transitively spawning processes.** 2.2.4
shipped the `[lib]` reshape, but `registry_detect()` still fans out to
eight backends that shell out to vendor CLIs (`nvidia-smi`,
`system_profiler`, `vulkaninfo`, `hl-smi`, `neuron-ls`, `xpu-smi`,
`cerebras_cli`, `gc-info`) plus a `detect_interconnects` post-pass
that calls `ibstat`/`nvidia-smi topo`. Consumers with a no-subprocess
contract — `mihi` first, whose CLAUDE.md forbids `exec_*` from inside
a probe — couldn't safely call the entry point. This release classifies
every backend, adds a `builder_no_exec()` mask, and adds a
`registry_detect_no_exec()` entry point that masks off exec backends
(defense-in-depth even if the caller's mask had them set) and skips
`detect_interconnects`. The eight sysfs/syscall-only backends — ROCm,
Intel NPU, AMD XDNA, TPU, Qualcomm, Groq, Samsung NPU, MediaTek APU —
plus the sysfs post-passes (`enrich_bandwidth/pcie/numa`,
`detect_storage`, `detect_environment`) still run.

#### Added

- **`backend_uses_exec(b)` in `src/types.cyr`** — predicate over the
  `Backend` enum. Returns 1 for CUDA / APPLE / VULKAN / GAUDI /
  NEURON / INTEL_ONEAPI / CEREBRAS / GRAPHCORE; 0 for the remaining
  eight. Source-of-truth grep: any `detect_<backend>` in
  `src/detect/<backend>.cyr` that calls `run_tool*` is EXEC. Note
  `cloud_asic.cyr` is mixed (Cerebras and Graphcore EXEC, Groq sysfs)
  and `intel.cyr` is split across two backend bits (NPU sysfs, oneAPI
  EXEC) — the predicate is correct per-backend, not per-file.
- **`builder_no_exec()` in `src/registry.cyr`** — returns a builder
  mask with only the sysfs/syscall backends enabled. Built by
  iterating `BACKEND_COUNT` and inverting `backend_uses_exec`, so
  the source of truth stays in one place.
- **`registry_detect_no_exec()` in `src/registry.cyr`** — convenience
  entry point. Calls the new `registry_detect_with_opts(builder_no_exec(), 0)`
  internal. Library consumers with a no-subprocess contract (mihi M3,
  any future read-only probe library) include this and never have to
  audit the per-backend exec status themselves.
- **`registry_detect_with_opts(mask, allow_exec)` in `src/registry.cyr`** —
  generalized orchestrator. When `allow_exec == 0` it force-strips the
  mask through `builder_no_exec()` and skips `detect_interconnects`.
  Existing `registry_detect()` / `registry_detect_with()` are now thin
  wrappers that pass `allow_exec = 1`, preserving 2.2.4 behavior exactly.

#### Changed

- **`VERSION`**: 2.2.4 → 2.2.5.
- **`dist/ai-hwaccel.cyr`** — regenerated by `cyrius distlib` to pick
  up the new entry points. Deterministic across runs.
- **README "Using as a library"** — documents `registry_detect_no_exec()`
  as the preferred entry point for read-only consumers; bumps the
  example `tag` to `"2.2.5"`.

#### Compatibility

- **CLI binary**: no change. `registry_detect()` semantics preserved —
  same backends, same post-passes, same warnings. All 11 test units
  (518 assertions) still pass; smoke output byte-identical to 2.2.4.
- **Existing binary consumers**: zero impact (they invoke the CLI,
  which calls `registry_detect()`, unchanged).
- **2.2.4 library consumers**: source-compatible. `registry_detect()`
  and `registry_detect_with(mask)` still resolve and behave identically.
- **New no-exec consumers** (mihi M3): `include "lib/ai-hwaccel.cyr"`,
  call `registry_detect_no_exec()`, walk `reg_profiles(r)`. The no-exec
  contract is now part of the library API rather than tribal knowledge.

## [2.2.4] — 2026-05-19

**`[lib]` reshape — first library-shaped consumer (`mihi`) unblocked.**
ai-hwaccel has been binary-only since v1.0.0; every consumer to date
(hoosh, daimon, Irfan, AgnosAI, murti, tazama) calls the CLI and parses
JSON. `mihi` v0.4.0 (M3 — GPU probe) cannot: its CLAUDE.md forbids
spawning processes from probes, so the GPU detection surface it needs
has to be reachable via `include`, not `exec`. This release adds the
`[lib].modules` surface and the `cyrius distlib`-produced
`dist/ai-hwaccel.cyr` bundle so mihi (and any future library consumer)
can pin against ai-hwaccel from their own `cyrius.cyml`. Mirrors the
agnosys / libro / patra / yukti / mihi pattern.

#### Added

- **`[lib].modules` in `cyrius.cyml`** — 35 modules listed in
  `src/main.cyr` include order. Excluded: `src/main.cyr` (CLI argv
  parsing) and `src/json_out.cyr` (CLI output formatting — library
  consumers serialize on their own terms). Every detection backend,
  the registry/profile surface, the sharding planner, the cost model,
  the training memory estimator, the model-format header parser, and
  the async/cache/lazy registry wrappers all ride along.
- **`dist/ai-hwaccel.cyr`** — single-file bundle produced by
  `cyrius distlib`. 5392 lines, 168 KiB at 2.2.4. Byte-deterministic
  across runs (verified — same SHA-256 from two sequential invocations).
  Consumers pull via `[deps.ai-hwaccel] modules = ["dist/ai-hwaccel.cyr"]`;
  `cyrius deps` drops it as `lib/ai-hwaccel.cyr` for `include`.
- **README "Using as a library" subsection** — copy-pasteable
  `[deps.ai-hwaccel]` block + `include` example. Documents the
  bundle-included surface and the CLI-only exclusions.
- **CI: `distlib drift + determinism` step** — runs `cyrius distlib`,
  diffs against the committed bundle (drift = consumers shipping stale
  code), then re-runs and SHA-256-compares (non-determinism = breaks
  the reproducible-build contract). Mirrors the libro / mihi / yukti
  gates. Sits between `Lint` and `Build (DCE)` in the workflow.

#### Changed

- **`VERSION`**: 2.2.3 → 2.2.4.
- **`docs/development/roadmap.md`** — 2.2.4 deliverable checkboxes
  marked complete for the in-repo work (manifest, bundle, README,
  no-regression, determinism guard); the `mihi-side smoke` remains
  unchecked pending external mihi M3 work.

#### Compatibility

- **CLI binary**: no change. `cyrius build src/main.cyr build/ai-hwaccel`
  still produces a 287 KiB ELF; all 11 test units (518 assertions) pass;
  output byte-identical to 2.2.3.
- **Existing binary consumers**: zero impact. They pin to the binary
  release artifacts and invoke `ai-hwaccel` as a subprocess, which is
  untouched.
- **New library consumers**: `[deps.ai-hwaccel] tag = "2.2.4"` resolves
  via `cyrius deps`, then `include "lib/ai-hwaccel.cyr"` exposes the
  detection entry points (`registry_detect`, `registry_detect_async`,
  the family-specific `detect_<backend>` calls, the profile / plan /
  cost / training APIs) directly — no subprocess, no JSON parsing,
  no environment leakage.

## [2.2.3] — 2026-05-19

**Toolchain rename slot — `cycc` canonical, `cc5` only as a legacy
symlink. Push back DXGI feature work; absorb the cyrius 6.0.0 compiler
rename as the 2.2.3 slot.** Cyrius 6.0.0 renames the compiler binary
from `cc5` to `cycc` (with `cc5_aarch64` → `cycc_aarch64`, `cc5_win`
→ `cycc_win`). The legacy `cc5*` names ship as symlinks for now, so
nothing breaks immediately — but every live, forward-looking reference
moves to the canonical name so the codebase stops drifting behind the
toolchain.

#### Changed

- **`cyrius.cyml`**: `cyrius = "5.11.8"` → `cyrius = "6.0.0"`.
- **`VERSION`**: 2.2.2 → 2.2.3.
- **`CLAUDE.md`**:
  - Compiler line: "Cyrius cc5 5.11.8" → "Cyrius cycc 6.0.0", with
    a note that `cc5` remains as a legacy symlink in `~/.cyrius/bin/`.
  - `#derive(accessors)` version note: `cc5 v3.7.1+` → `cycc v3.7.1+`.
- **`README.md`** Key Numbers row: `Cyrius cc5 5.10.34` → `Cyrius cycc 6.0.0`.
- **`src/detect/windows.cyr`** header comments: forward-looking
  references to `cc5_win` → `cycc_win` (with the legacy-symlink note
  inline). The 5.11.5 PE emit blocker note is preserved as a status
  question against 6.0.0 rather than as a hard block.
- **`docs/development/roadmap.md`**:
  - 2.2.3 CI cross-build entry — reframed from "upstream-blocked on
    cc5_win 5.11.5" to "re-verify against `cycc_win` 6.0.0; smoke
    probe required before declaring it unblocked."
  - 2.3.0 Python-bindings entry: "cyrius cc5 emits ELF/Mach-O/PE" →
    "cyrius cycc emits ELF/Mach-O/PE."
  - 2.4.0 hot-plug entry: "cc5's defer" → "cycc's defer."
- **`.github/workflows/release.yml`**: aarch64 gate uses
  `~/.cyrius/bin/cycc_aarch64` (warning text updated to match).
- **`memory/reference_windows_host.md`** forward guidance — `cc5_win`
  / `cc5_win_cross` → `cycc_win` / `cycc_win_cross`.
- **`memory/MEMORY.md`** index entry phrasing updated.
- **CI stdlib resolution (`.github/workflows/ci.yml`,
  `.github/workflows/release.yml`)**: insert a "Sync stdlib into
  `./lib/`" step that runs `cyrius lib sync` before the existing
  "Resolve non-stdlib deps" (`cyrius deps`) step. Pre-6.0.0 the
  single `cyrius deps` invocation handled both; in 6.0.0 the
  responsibilities split, so CI now mirrors the local workflow.
  Without this change, CI fails with 18 × "cannot read
  ./lib/<name>.cyr" because `cyrius deps` no longer fills the
  stdlib subset.

#### Not changed (intentional)

- Shipped CHANGELOG entries (`cc5 adoption arc`, all 2.1.x sections,
  the 2.2.1 toolchain-bump entry, the 2.2.2 cc5_win 5.11.5 blocker
  filing) — historical record using the names in use at the time.
- `memory/feedback_cc5_win_exit_propagation.md` filename — preserved
  to keep cross-refs from the cyrius issue tree intact.
- Source files outside `src/detect/windows.cyr` — no other source
  references the compiler name; nothing to update.

#### Pushed to 2.2.4

- **DXGI COM binding + `DXGI_ADAPTER_DESC1` parser** (was 2.2.3 plan).
- **Linux-hosted fixture tests under `tests/fixtures/windows/`**.

#### Verification

- Linux build (`CYRIUS_DCE=1 cyrius build src/main.cyr build/ai-hwaccel`)
  against cycc 6.0.0: **287,096 bytes** (was 286,152 at 2.2.2, +944 bytes).
  The size change comes from the 6.0.0 stdlib snapshot, not project source —
  the stdlib added new modules (`atomic`, `async`, `mmap`, per-arch
  `syscalls_*_linux`, …) that the include graph now pulls in. 638
  unreachable fns NOPed by DCE (was 629 at 2.2.2).
- Test suite: 11 units, **520 assertions, 0 failures** (was 518 at 2.2.2;
  +2 from the new stdlib).
- fmt sweep: clean across `src/`, `tests/tcyr/`, `fuzz/`, `benches/`.

#### Stdlib repopulation note (workflow change)

- **`cyrius deps` no longer repopulates the stdlib in 6.0.0.** Use
  `cyrius lib sync` instead — it copies the snapshot at
  `~/.cyrius/versions/<pin>/lib/*.cyr` into `./lib/`. After fresh
  clone or cyrius upgrade, run `cyrius lib sync` then `cyrius deps`
  (the latter still resolves non-stdlib `[deps.*]` entries).
  CLAUDE.md updated to reflect this.

## [2.2.2] — 2026-05-11

**Windows backend — source-side skeleton.** `src/detect/windows.cyr`
exists, wired into the include graph behind `#ifdef CYRIUS_TARGET_WIN`,
ready to receive the DXGI binding. **Real Win64 cross-build is not
viable yet** — cc5_win 5.11.5 has a PE emit regression (documented
below) that prevents end-to-end validation. Source structure ships;
cross-build + cross-host smoke gate on a cc5_win patch.

### Added

- **`src/detect/windows.cyr`** — single-file skeleton behind
  `#ifdef CYRIUS_TARGET_WIN`. Stubs `detect_windows(profiles,
  warnings)` matching the `detect_<backend>(profiles, warnings)`
  convention used by the other detectors. Records a
  `warning_tool_not_found("dxgi-binding-pending")` warning so the
  runtime makes the stub state obvious at JSON level. Header
  comments document the planned scope (COM binding, DXGI_ADAPTER_DESC1
  parsing, ACCEL_DXGI / BACKEND_WIN_DXGI additions) for 2.2.3.
- **`src/main.cyr` include**: `include "src/detect/windows.cyr"`
  added between environment.cyr and registry.cyr. On Linux / macOS
  builds the entire file preprocesses out — `build/ai-hwaccel`
  stays **byte-identical at 286,152 bytes**.

### Investigated and recorded as upstream blocker

- **cc5_win 5.11.5 PE emit broken on cass**. Probed end-to-end this
  slot:
  - `syscall(60, 42); ` source → cc5_win builds a 1536-byte PE.
  - PE loads on cass (Win11 26200), process starts.
  - Exit code under `cmd /v /c "exe & echo exit=!errorlevel!"` is
    **0x40001000** (1073745920), not 42.
  - WriteFile output ("hello\n" from `syscall(1, 1, "hello\n", 6);`)
    never reaches stdout.
  - Same byte-identical emit from `cc5_win` (install) and
    `cc5_win_cross` (cyrius source build, v5.10.37) — both versions
    produce the same broken PE.
  - Linux build of the same source works fine.

  Recorded as `memory/feedback_cc5_win_exit_propagation.md` and
  filed in the cyrius tree at
  `docs/development/issues/2026-05-11-ai-hwaccel-cc5-win-pe-exit-propagation.md`
  (pending user-side review + commit on cyrius). ai-hwaccel's Win64
  cross-build + cass smoke gate on the fix. Linux fixture-based tests
  for the DXGI parser (2.2.3) don't depend on this — the parser
  logic can be exercised against synthetic `dxdiag` output without
  needing a Win64 binary to load on cass.

- **Full ai-hwaccel cross-build via cc5_win not yet viable.** A
  full `cc5_win src/main.cyr` invocation runs to exit 0 but emits
  a 1,536-byte stub PE rather than a real binary — cc5_win silently
  fails on something deeper in the transitive include tree (likely
  one of the Linux-only stdlib paths that the PE backend doesn't
  yet reroute). Cyrius's own 5.11.6 release notes call out this
  shape as "compile-path quirk." Skipping the CI cross-build step
  until cc5_win can handle the full tree.

### Verification

- Linux build: **286,152 bytes** (byte-identical to 2.2.1 — the
  windows.cyr include adds zero on non-Win targets).
- 11 test units, **518 assertions, 0 failures**.
- fmt sweep clean.
- `cyrius vet src/main.cyr`: 37 deps (was 36 — `+1` for
  `src/detect/windows.cyr`), 0 untrusted, 0 missing.

### Next slot — 2.2.3

Either:
- **Wait for cc5_win patch** (cyrius-side) and resume cross-build +
  COM binding in one go.
- **Or proceed Linux-side**: implement the DXGI parser against
  synthesized `dxdiag` text fixtures under `tests/fixtures/windows/`.
  Parser logic ships and ride along once the cross-build is viable.

The user can redirect.

## [2.2.1] — 2026-05-11

**Cyrius toolchain bump 5.10.34 → 5.11.8.** Mechanical prerequisite
slot — same shape as 2.0.1 (which bumped 3.10.0 → 5.10.34). Unblocks
the Windows DXGI work queued for 2.2.x by bringing `cc5_win` into the
standard install bundle and folding the v5.11.6 PE exit-code fix into
the pinned toolchain.

### Changed

- **`cyrius.cyml`**: `cyrius = "5.10.34"` → `cyrius = "5.11.8"`.
- **`CLAUDE.md`**: compiler reference line bumped to match.

### Toolchain context

- **5.10.x → 5.11.x** absorbed in one bump (the cycle ran .35 → .50 →
  .0 → .8, 9 patches + the 5.11.0 minor open). Highlights relevant to
  ai-hwaccel:
  - **5.10.49** — premise-debunk: PE exit-code propagation was always
    working; the false-negative was a chat-side wrapper bug
    (`%errorlevel%` vs `!errorlevel!`).
  - **5.11.6** — underlying PE exit-code path fully verified, plain
    `cmd /c "prog.exe & echo exit=%errorlevel%"` shape works again on
    Win64 PE binaries.
  - **5.11.8** — `cc5_win` shipped in the default install bundle at
    `~/.cyrius/bin/cc5_win` (612 KB). No longer needs to be invoked
    from the cyrius source-build `cc5_win_cross` artifact.
- **5.11.0 minor opens** with kavach P1 sandbox syscall wrappers
  (`sys_fchmod`, `sys_setresuid` / `sys_setresgid`, `sys_prctl`,
  `sys_seccomp`, `sys_execveat`) and the stdlib annotation arc. None
  of these affect ai-hwaccel directly — we don't use seccomp or
  unprivileged process management.

### Verification

- Clean rebuild against 5.11.8: `build/ai-hwaccel` byte-identical to
  2.2.0 / 2.1.7 (**286,152 bytes**). The toolchain bump produces the
  same emit for the existing source.
- Test suite: 11 units, **518 assertions, 0 failures**.
- fmt sweep: clean across `src/`, `tests/tcyr/`, `fuzz/`, `benches/`.
- `cc5_win` present at `~/.cyrius/bin/cc5_win` — Windows cross-build
  is now a `cc5_win <source> <output>` invocation away (skeleton
  lands next slot).

### Documentation

- **Memory** (`reference_windows_host.md`) — wrapper-gotcha section
  reorganised: 5.11.8 pin uses plain shell; the `cmd /v /c …
  !errorlevel!"` variant kept as a historical note for any
  5.10.x-pinned consumer that hits the same issue.
- **Roadmap** — Windows DXGI entry no longer carries the "install
  cc5_win" prerequisite; pickup target is now the source-side work
  only.

## [2.2.0] — 2026-05-11

**Test-file rename correction + README refresh.** A docs-and-renames
slot — accepted as a release but not a real Platform Validation
deliverable. The 2.2.x arc's actual feature work (Windows DXGI,
hardware-fixture coverage for the untested backends) is open and
queued for 2.2.1+ pickup, not deferred or blocked.

### Changed

- **6 test files renamed** to match actual content (audit revealed
  the 2.1.0 rename assumed phase numbers → subject 1:1, which several
  didn't):

  | 2.1.0 name (misleading) | Actual content | 2.2.0 name |
  |---|---|---|
  | `cost_model_test.tcyr` | sharding plans + training memory + model | `planning_test.tcyr` |
  | `detect_gaudi_test.tcyr` | CUDA + Gaudi + Neuron parsing | `gpu_parser_test.tcyr` |
  | `detect_neuron_test.tcyr` | Apple + Intel + AMD XDNA + cloud ASIC | `backend_test.tcyr` |
  | `registry_test.tcyr` | which + run_tool + CSV + sysfs | `io_test.tcyr` |
  | `sharding_test.tcyr` | interconnect + bandwidth + PCIe + storage | `topology_test.tcyr` |
  | `system_io_test.tcyr` | registry + builder + suggest_quant | `registry_test.tcyr` |

  The remaining 5 names (`foundation_test`, `profile_test`,
  `requirement_test`, `json_output_test`, `model_format_test`) were
  already accurate and stay unchanged. Git tracks all 6 as renames —
  content byte-identical, all 518 assertions still pass.

- **`README.md` refreshed**:
  - Binary size: 217 KB → 286 KB (cc5 stdlib growth + derived setter
    surface; documented in 2.0.1 / 2.1.x CHANGELOG entries).
  - Compiler line: Cyrius cc5 5.10.34 (was implicit).
  - Test-units table (the 11 `.tcyr` files with one-line descriptions)
    replaces the stale "11 phases" reference.
  - New section: "Pattern: derived struct accessors" with the
    `#derive(accessors)` shape + CI gate pointer.
  - Removed: Compile time + Source LOC rows (out of date and not
    load-bearing; binary size + assertion count carry the same
    signal).
  - Development workflow snippet: now includes `cyrius deps` and the
    `tests/tcyr/*.tcyr` test-loop pattern that matches CI.

- **`docs/development/roadmap.md`**: the 2.1.0 test-rename entry now
  carries the corrected mapping (was misleading documentation
  alongside the bad names).

### Test suite

- 11 units, 518 assertions, 0 failures (unchanged from 2.1.7 — pure
  rename slot).

### Carried forward — **open, queued for 2.2.1+**

These are the real 2.2.x arc items. Reframed from "deferred / access-
blocked" to **open work**, because the source-side path can ship
against synthesized fixtures even before hardware access lands.

- **Windows DXGI backend** — `src/detect/windows.cyr` behind
  `#ifdef CYRIUS_TARGET_WIN64`, COM binding for `IDXGIFactory1::
  EnumAdapters1`, `DXGI_ADAPTER_DESC1` parsing. Toolchain side:
  install `cc5_win` into `~/.cyrius/bin/` (the cross-compiler exists
  at v5.10.37 in cyrius's source build), wire CI cross-build behind
  the same `if [ -x $HOME/.cyrius/bin/cc5_win ]` gate the release
  workflow uses for aarch64. Next pickup target.
- **Hardware-fixture coverage** — move inline sample tool outputs
  in `gpu_parser_test.tcyr` / `backend_test.tcyr` to per-backend
  files under `tests/fixtures/`. Establishes the drop-zone for
  contributor captures from real H100 / MI300X / TPU v5 / Trn1 /
  Gaudi 3 hardware. The parser tests run against fixtures, so the
  source-side work doesn't gate on cloud access.
- **Untested-backend parsers** — Cerebras / Graphcore / Groq /
  Samsung NPU / MediaTek APU. Each one has a documented sysfs or
  CLI output format; the parser can ship against a synthesized
  fixture from vendor docs and graduate to a real-hardware fixture
  as captures come in.

See `docs/development/roadmap.md` § 2.2.x for the per-item plan.

## [2.1.7] — 2026-05-11

**P(-1) scaffold hardening: 8 more structs derived, every heap struct
in the codebase now uses `#derive(accessors)`.** Closes the 2.1.x
minor before 2.2.0's platform-validation arc opens. Baseline +
post-audit benchmarks captured; small perf cost documented.

### Pre-audit baseline

- 518 assertions, 0 failures across 11 test units
- `cyrius lint`: clean (no warnings on any `src/` / `src/detect/` file)
- `cyrius fmt`: clean across `src/`, `tests/tcyr/`, `fuzz/`, `benches/`
- Binary: 281,720 bytes
- Bench hot-paths:
  - `total_memory_13dev`: 136 ns
  - `has_accelerator_13dev`: 27 ns
  - `count_family_gpu_13dev`: 288 ns
  - `json_serialize_13dev`: 24 µs
  - `json_summary_13dev`: 5 µs
  - `parse_cuda_8gpu`: 22 µs

### Changed

Eight more structs converted to `#derive(accessors)`:

- **`env` struct** (`src/system_io.cyr`, was RuntimeEnvironment) — 8
  fields: `is_docker`, `is_k8s`, `k8s_namespace`, `cloud_provider`,
  `instance_type`, `region`, `k8s_gpu_count`, `k8s_gpu_source`. Two
  setters renamed (field-derived): `env_set_docker` →
  `env_set_is_docker`, `env_set_k8s` → `env_set_is_k8s`. 5 caller
  sites updated (4 in `src/detect/environment.cyr`, 1 in
  `tests/tcyr/profile_test.tcyr`).
- **`sio` struct** (`src/system_io.cyr`, was SystemIo) — 3 fields:
  `interconnects`, `storage`, `environment`. Also cleaned up 4
  internal `var ics = load64(sio);` shortcuts → `sio_interconnects(sio)`.
- **`shard` struct** (`src/system_io.cyr`, was ModelShard) — 6
  fields: `id`, `layer_start`, `layer_end`, `device_type`,
  `device_id`, `memory_bytes`. Constructor stays `model_shard_new`.
  Internal helpers `shard_num_layers` / `shard_is_valid` switched
  from `load64(ms + N)` to derived getters.
- **`cloud_inst` struct** (`src/cost.cyr`, was CloudGpuInstance) — 10
  fields. Field name `total_gpu_mem_gb` (layout) → `total_mem_gb` to
  match existing `cloud_inst_total_mem_gb` accessor callers in
  `src/main.cyr:210` and `src/cost.cyr:257`. No external constructor
  (built by `load_cloud_instances` JSON parser via
  `_parse_cloud_field_*` helpers, which take offset as a parameter
  and stay raw inside the defining file).
- **`rec` struct** (`src/cost.cyr`, was InstanceRecommendation) — 3
  fields: `instance`, `mem_required`, `headroom_x100`. Constructor
  stays `inst_rec_new`.
- **`cached` struct** (`src/cache.cyr`, was CachedRegistry) — 4
  fields: `registry`, `last_detect_secs`, `ttl_secs`, `mutex`. Five
  internal helpers (`cached_get`, `cached_invalidate`, `cached_ttl`,
  `cached_is_valid`) switched to derived accessors.
- **`disk_cached` struct** (`src/cache.cyr`, was DiskCachedRegistry)
  — 5 fields adding `cache_path`. Same internal-cleanup pattern.
- **`lazy` struct** (`src/lazy.cyr`, was LazyRegistry) — 3 fields:
  `profiles`, `probed` (i64 bitmask), `mutex`. Seven internal raw
  `load64(lr + N)` sites cleaned to derived getters. Bit-manipulation
  helpers (`_lazy_is_probed`, `_lazy_mark_probed`) use
  `lazy_probed` + `lazy_set_probed` plus inline mask compute.

### CI

- **Raw-offset guard at 15 entries** (10 cross-file + 5 field-count
  bound):
  - Cross-file `check_struct`: `storage` (sd), `ic` (ic), `plan` (sp),
    **`shard` (ms)**, **`sio` (sio)**, `reg` (r), `profile` (p),
    **`cloud_inst` (ci)**, **`rec` (rec)**, **`lazy` (lr)**. Five new.
  - Field-count bound: `est` (e, 4), **`env` (e, 8)** (was `runtime_env`,
    renamed to match the struct name), `meta` (m, 5), `model` (m, 4),
    **`disk_cached` (c, 5)** (covers both `cached` and `disk_cached`
    since both use param `c` in the same file; the larger bound
    accommodates both — see roadmap rationale). Two new.

### Documentation

- `CLAUDE.md` now lists `#derive(accessors)` as a project principle
  with the canonical pattern, the full 16-struct inventory, and a
  pointer to the CI gate.
- `docs/development/roadmap.md` marks the 2.1.x arc as **SHIPPED**
  with a per-slot summary; 2.2.0 is the next arc.

### Post-audit verification

- 518 assertions, 0 failures (unchanged across the slot)
- Binary: **286,152 bytes** (was 281,720) — **+4,432 bytes** (≈+1.6%)
  for derive-generated setters across 8 new structs. cc5's DCE
  inlines the truly-internal setters (constructor-only callers); the
  net cost is from setters that have external consumers and stay as
  callable functions.
- Bench deltas vs baseline:
  - `total_memory_13dev`: 136 → 145 ns (+9 ns, ≈+6.6%)
  - `count_family_gpu_13dev`: 288 → 304 ns (+16 ns, ≈+5.6%)
  - `has_accelerator_13dev`: 27 → 28 ns (+1 ns, within noise)
  - `json_serialize_13dev`: 24 → 25 µs (+1 µs, ≈+4%)
  - `json_summary_13dev`: 5 → 6 µs (+1 µs, ≈+20% but tiny absolute)
  - `parse_cuda_8gpu`: 22 → 20 µs (improved — within noise)

  The sub-µs hot paths regress slightly because derived getters are
  thin function calls (`fn reg_profiles(r) { return load64(r); }`)
  rather than inlined `load64(r + 0)`. The accessor surface offers
  much stronger guarantees (CI-gated, derive-traced, no offset
  arithmetic at call sites) — the trade is real but the absolute
  numbers stay in the sub-microsecond range for everything but
  `json_*`, which were already micro-second.

### Arc closes

- **2.1.x cc5 adoption arc complete.** Seven slots over two days.
  Every heap-allocated struct in the codebase (16 total) now uses
  `#derive(accessors)`. CI gate covers them all. The codebase is
  internally consistent before 2.2.0's external-feature work begins.
- **Next arc: 2.2.0 — Platform Validation** (Windows PE via cc5
  Win64 backend, live cloud hardware validation, untested-backend
  triage).

## [2.1.6] — 2026-05-11

**cc5 adoption arc closes — `profile` converted, 8 structs total on
`#derive(accessors)`.** The biggest struct in the codebase (20 fields,
160 bytes, most-called accessor surface) now uses derived getters and
setters. The cc5 adoption arc that opened with 2.1.0's mechanical
test/CI reorg ends here.

### Changed

- **`profile` struct** (`src/profile.cyr`, was AcceleratorProfile) —
  20 fields: `accel_type`, `device_id`, `available`, `memory_bytes`,
  `compute_cap`, `driver_version`, `device_name`, `mem_bw_x1000`,
  `mem_used`, `mem_free`, `pcie_bw_x1000`, `numa_node`, `temp_c`,
  `power_x1000`, `gpu_util`, `tpu_version`, `tpu_chips`, `gaudi_gen`,
  `neuron_chip`, `neuron_cores`. The 20 hand-rolled getters and 17
  hand-rolled setters are gone; the constructor `profile_new` calls
  the derived `profile_set_*` setters to initialise all 20 slots.
  Struct name `profile` matches the existing `profile_*` accessor
  convention — zero callsite changes for the dozens of detect/* /
  registry / json_out / sharding / training / cost helpers that
  consume the surface.
- **4 cross-file raw `store64(p + 24, …)` writes** (all setting
  `profile.memory_bytes` after hardware-specific detection adjusts the
  value) converted to `profile_set_memory_bytes(p, …)`:
  - `src/detect/gaudi.cyr:46` — hl-smi memory override
  - `src/detect/vulkan.cyr:62` — vulkaninfo heapSize parsing
  - `src/detect/cuda.cyr:136` — GH200 unified memory adjustment
    (`+ 480 GiB`)
  - `src/detect/rocm.cyr:100` — ROCm CXL visible-VRAM total

### CI

- **Raw-offset guard** now registers 9 entries (5 cross-file + 4
  field-count bound):
  - Cross-file `check_struct`: `storage` (sd), `ic` (ic), `plan` (sp),
    `reg` (r), **`profile` (p)** (new).
  - Field-count bound: `est` (e, 4), `runtime_env` (e, 8),
    `meta` (m, 5), `model` (m, 4).

### Binary size

- `build/ai-hwaccel`: **281,720 bytes** (was 281,592 at 2.1.5). +128
  bytes — `profile` had hand-rolled setters already, so the derive
  output mostly replaced existing code rather than adding fresh setter
  bodies. 518 assertions, 0 failures.

### Arc closes

- **2.1.x cc5 adoption arc complete.** Six slots over two days:
  - 2.1.0 — test reorg + CI tighten
  - 2.1.1 — Rust parity audit + gitignore cleanup
  - 2.1.2 — defer audit (clean), chrono investigated and rejected,
    build/ untracked
  - 2.1.3 — `#derive(accessors)` proof of concept (meta + storage),
    first raw-offset CI gate
  - 2.1.4 — three more structs (ic + plan + est), CI gate gains
    libro field-count bound check
  - 2.1.5 — two more structs (reg + model), gate at 8 entries
  - 2.1.6 — `profile` (this slot), gate at 9 entries
- **Next arc: 2.2.0 — Platform Validation.** Live cloud hardware
  validation (NVIDIA H100/A100/GH200, AMD MI300X, Google TPU v5,
  AWS Neuron, Intel Gaudi 3), Windows PE backend (now reachable
  via cc5 5.10.x), and the untested-backend list (Cerebras WSE,
  Graphcore IPU, Groq, Samsung NPU, MediaTek APU).

## [2.1.5] — 2026-05-11

**cc5 adoption arc — `#derive(accessors)` continues. Two more structs
converted (`reg` + `model`), CI gate expanded.** Bundled because both
were medium-complexity and shared the same review surface (registry
internals, model dispatch). Zero external call-site changes.

### Changed

- **`reg` struct** (`src/registry.cyr`, was accelerator_registry) —
  4 fields: `profiles`, `warnings`, `system_io`, `schema`. Struct named
  `reg` to match the existing `reg_*` accessor convention; constructor
  stays `registry_new`. Also cleaned up 9 internal `var profs =
  load64(r);` shortcuts in helpers (`reg_total_memory`,
  `reg_total_accel_memory`, `reg_has_accelerator`, `reg_best_available`,
  `reg_count_by_family`, `reg_by_family`, `reg_suggest_quant`) →
  `var profs = reg_profiles(r);` so the file is internally consistent
  with the new accessor surface.
- **`model` struct** (`src/model.cyr`, was ModelProfile) — 4 fields:
  `name`, `family`, `params_b_x1000`, `dtype`. Param `m` is shared with
  `meta` in `model_format.cyr` — both rely on the libro field-count
  bound CI check rather than a cross-file specific-struct guard. One
  in-place mutation in `_parse_models` line 137 — `store64(m + 16,
  whole * 1000 + frac)` — converted to the derived setter
  `model_set_params_b_x1000(m, …)`.

### CI

- **Raw-offset guard registry expanded** to 8 entries total:
  - Cross-file `check_struct`: `storage` (sd), `ic` (ic), `plan` (sp),
    **`reg` (r)** (new).
  - Field-count bound: `est` (e, 4), `runtime_env` (e, 8), `meta` (m, 5),
    **`model` (m, 4)** (new).

### Binary size

- `build/ai-hwaccel`: **281,592 bytes** (was 280,696 at 2.1.4). +896
  bytes for derive-generated setters across `reg` + `model`. 518
  assertions, 0 failures.

## [2.1.4] — 2026-05-11

**cc5 adoption arc — `#derive(accessors)` continues. Three more structs
converted + CI raw-offset gate expanded with libro's field-count bound
check.** Zero external call-site changes — every struct gets a name that
matches its existing accessor prefix, so derive generates the names the
code already imports.

### Changed

- **`ic` struct** (`src/system_io.cyr`, was Interconnect) — 4 fields:
  `kind`, `name`, `bw_x1000`, `state`. Struct named `ic` to match the
  existing `ic_*` accessor shorthand; constructor stays
  `interconnect_new`. 16 external call sites of `ic_kind` /
  `ic_name` / `ic_bw_x1000` / `ic_state` / `ic_set_state` unchanged.
- **`plan` struct** (`src/system_io.cyr`, was ShardingPlan) — 5 fields:
  `shards`, `strategy`, `strategy_count`, `total_memory`,
  `est_tps_x1000`. Struct named `plan` to match the existing `plan_*`
  accessor convention; constructor stays `sharding_plan_new`. The
  pre-existing manual setters (`plan_set_total_memory`,
  `plan_set_est_tps_x1000`) replaced by derive-generated setters; the
  rest of the surface unchanged.
- **`est` struct** (`src/training.cyr`, was MemoryEstimate) — 4 fields:
  `model_x1000`, `optimizer_x1000`, `activation_x1000`, `total_x1000`.
  Struct named `est` to match the `est_*` accessor convention;
  constructor stays `mem_est_new`. Param `e` is shared with
  `runtime_env` in `system_io.cyr` — see CI gate change below.

### CI

- **Raw-offset guard expanded with `check_offset_bound`** — libro's
  field-count bound check for structs whose canonical param name is
  ambiguous across files. For each `(file, param, struct, field_count)`
  tuple, every raw `load64(<param> + N)` / `store64(<param> + N, ...)`
  site in `<file>` must have `N ≤ (field_count − 1) * 8`. Catches
  off-by-one after a field-count shrink, and accidental access past a
  struct boundary after another struct grows. Currently registered:
  - `est`   (`src/training.cyr`,   param `e`, 4 fields → max +24)
  - `runtime_env` (`src/system_io.cyr`, param `e`, 8 fields → max +56)
  - `meta`  (`src/model_format.cyr`, param `m`, 5 fields → max +32)
- **Cross-file `check_struct` guards added** for the two newly-derived
  structs with unambiguous params:
  - `ic`   in `src/system_io.cyr`, param `ic`
  - `plan` in `src/system_io.cyr`, param `sp`

### Binary size

- `build/ai-hwaccel`: **280,696 bytes** (was 279,656 at 2.1.3). +1,040
  bytes for derive-generated `_set_*` setters across the three new
  structs. 518 assertions, 0 failures.

## [2.1.3] — 2026-05-11

**cc5 adoption arc — `#derive(accessors)` lands, first two structs
converted + raw-offset CI gate.** Proof-of-concept slot establishing
the pattern (and the CI guard rail) before the bigger structs
(`profile`, `accelerator_registry`, `model`) follow in 2.1.4+.

### Changed

- **`meta` struct** (`src/model_format.cyr`) — 5 hand-rolled
  `load64(m + N)` getter functions replaced with
  `#derive(accessors) struct meta { format; param_count; dtype;
  tensor_count; format_version; }`. The constructor `meta_new` stays
  manual (derive only generates accessors) but now calls the derived
  `meta_set_*` setters internally instead of raw `store64(m + N, …)`.
  All 21 external call sites (in `model_format.cyr` itself,
  `tests/tcyr/model_format_test.tcyr`) use the existing `meta_*`
  getter names unchanged — derive generates them under exactly the
  names the code already imports.
- **`storage` struct** (`src/system_io.cyr`) — 3 hand-rolled getters
  replaced with `#derive(accessors) struct storage { name; kind;
  bw_x1000; }`. Same shape as `meta`: constructor stays manual but
  uses derived setters; external call sites
  (`src/detect/disk.cyr:47`, `tests/tcyr/sharding_test.tcyr:147`,
  `tests/tcyr/profile_test.tcyr:195-204`) keep their `storage_*`
  getter names.

### Added

- **Raw-offset CI gate (`Raw-offset guard` step in
  `.github/workflows/ci.yml`)** — for each `#derive(accessors)`
  struct, no file outside its defining file may do raw
  `load64(<param> + N)` / `store64(<param> + N, …)` /
  `load64(<param>)` on it. Mirrors the libro v2.6.x pattern.
  Registers `storage` (param `sd`) — unambiguous across `src/`,
  works with the simple cross-file `check_struct` form. `meta`
  (param `m`) is held back because `src/model.cyr` legitimately
  uses `m` for its own (not-yet-derived) struct; that case is
  documented to use the libro field-count bound check once `model`
  itself moves to derive.

### Binary size

- `build/ai-hwaccel`: **279,656 bytes** (was 278,808 at 2.1.2).
  +848 bytes for derive-generated `_set_*` setters across both
  structs; setters for `meta` aren't called from outside the
  constructor, so stricter DCE passes will reclaim those later.
  Test suite: 518 assertions, 0 failures (unchanged).

### Investigated and rejected this slot

- **Multi-return `(value, error)` in detect/* (the 2.1.0-arc item) —
  doesn't fit.** The detect entry points are
  `detect_<backend>(profiles, warnings)`: both vec OUT-params,
  pushing 0..N profiles and 0..M structured warnings, returning an
  unused `0`. There is no single value to multi-return, and errors
  are already accumulated into `warnings` as structured entries
  rather than collapsed to a sentinel int. Closed in the roadmap;
  the out-param-vec pattern is now noted as canonical.
- **`lib/regex.cyr` for parser output (the 2.1.0-arc item) — no
  fit.** The detect parsers go `run_tool` → `str_split` (lines) →
  `parse_csv_line` (fields) → `str_contains_cstr` (single-token
  substring checks). Substring checks aren't what regex replaces;
  the CSV helpers are already idiomatic. Closed in the roadmap.
- **`lib/test.cyr` adoption (the 2.1.0-arc item) — closed as a
  misread.** `lib/test.cyr` is a `test_each` parameterised-test
  helper, not an alternative assertion framework. The tests already
  use stdlib `lib/assert.cyr`. Nothing to migrate.

## [2.1.2] — 2026-05-11

**cc5 adoption arc — verification slot.** Two roadmap items investigated;
both close cleanly with **no source change**. Documenting the outcomes so
future work doesn't re-tread the same ground.

### Verified

- **Defer-on-all-paths audit (cc4+ defer semantics)** — swept
  `src/system_io.cyr`, `src/cache.cyr`, `src/detect/command.cyr` for
  manual file-descriptor management on paths where an early return could
  leak. The only `file_open` / `file_close` pair in the entire tree is
  `cmd_getenv()` at `src/detect/command.cyr:17-48`; the `file_close` runs
  unconditionally on line 22 before any of the subsequent returns. All
  other I/O goes through `lib/fs.cyr`'s `file_read_all` / `file_write_all`
  atomic wrappers, which handle open/read/close internally. **Zero leaks
  found, no `defer` insertions needed.** Roadmap item closes; the cc4+
  defer-on-all-paths feature stays available for future code paths.

### Investigated and rejected

- **`lib/chrono.cyr` adoption for cache TTL** — attempted replacing
  `cache.cyr`'s `_monotonic_secs()` (4 syscall lines) with
  `clock_now_ms() / 1000` via stdlib chrono. The functional change
  works (all tests pass), but pulling in chrono adds the module to
  `[deps].stdlib` for the sake of saving 3 lines in one helper. **Net
  cost > net win**, so the local `syscall(228, CLOCK_MONOTONIC, &ts)`
  pattern stays. The roadmap entry is now annotated with the trade-off
  so the question doesn't get re-litigated.

### Removed

- **`build/ai-hwaccel` untracked from git.** The compiled binary had been
  accidentally committed at an early release (`c173383 cleanup for
  release`) and re-resurfaced as a modified file after every local build
  because `/build/` in `.gitignore` only ignores *untracked* files. Removed
  from the index; the directory stays gitignored.

### Carried forward

- **`case N: { ... }` switch blocks** — separately attempted (and
  reverted) in a pre-2.1.2 spike. cc5 5.10.x's `PARSE_CASE` rejects
  enum-name labels (`case FAMILY_CPU:` → `expected number, got
  identifier`); the v5.10.48 enum-const-fold landed for
  `PARSE_ARRAY` / `PARSE_GVAR_ARR` only. Roadmap entry documents the
  upstream limitation; the if-chain dispatch in `accel_name` /
  `family_name` / `format_name` / `_gguf_file_type_name` /
  `requirement_satisfied` stays until cyrius extends the fold to
  case labels.

## [2.1.1] — 2026-05-10

**Rust-port parity verification + scaffolding cleanup.** No code changes;
documents the audit and removes stale defensive lines.

### Verified

- **Parity audit against the 1.x Rust crate** — every public API in
  `docs/migration.md`'s Rust↔Cyrius mapping table (21 entries) has a
  matching `fn` in `src/`. Spot-checked: `registry_detect`,
  `registry_detect_threaded`, `builder_all`/`builder_none`/`builder_with`,
  `reg_best_available` / `reg_total_memory` / `reg_has_accelerator`,
  `reg_plan_sharding` / `reg_suggest_quant`, `cached_registry_new` /
  `cached_get`, `lazy_new` / `lazy_by_family`, `profile_cuda` /
  `profile_memory_bytes` / `profile_accel_type`, `accel_is_gpu`,
  `model_can_run`, `detect_model_format` / `detect_format_from_bytes`.
  No drift since the 2.0.0 mapping was authored.
- **JSON `schema_version` still `2`** — wire-compatible with v1.x
  Rust output.
- **Cyrius-only additions confirmed live**: `requirement.cyr` (scheduling
  integration), `async_detect.cyr` (threaded backends), `cache.cyr` (TTL
  + disk variant), `lazy.cyr` (per-family deferred detection),
  `model_format.cyr` (SafeTensors/GGUF/ONNX/PyTorch).
- **Documented intentional gaps stay accurate**: Windows detection is
  the only post-Rust feature gap; reachable now that cc5 5.10.x ships
  a Win64 PE backend, tracked in the roadmap's 2.2.0 slot.

### Removed

- **`.gitignore`: `rust-old/` line** — `rust-old/` was deleted from the
  tree at the 2.0.0 release (see [2.0.0] § Removed). The defensive
  ignore line has been a no-op for the project's entire Cyrius
  lifetime; removed now that the parity audit confirms nothing depends
  on the historical directory.
- **`.gitignore`: `tarpaulin-report.html`, `criterion/`** — Rust-era
  coverage / benchmark artifacts. No Rust toolchain runs against this
  tree anymore.

## [2.1.0] — 2026-05-10

**Slot 1 of the cc5 adoption arc — test reorg + CI tighten.** Pure
structural work; no source-code feature changes, no API surface changes
for consumers. Sets up cleaner ground for the `#derive(accessors)` /
multi-return / switch-block refactors planned later in the arc.

### Changed

- **Tests renamed to descriptive names + relocated under `tests/tcyr/`.**
  Phase numbers (`test_phase1.cyr` … `test_phase11.cyr`) mapped to no
  concept readable from the source; replaced with mission-named units:
  - `foundation_test.tcyr`     (errors, accel types, units, quantization)
  - `profile_test.tcyr`        (profile construction + setters)
  - `registry_test.tcyr`       (registry assembly)
  - `detect_gaudi_test.tcyr`   (Gaudi detection)
  - `detect_neuron_test.tcyr`  (Neuron detection)
  - `sharding_test.tcyr`       (plan / training)
  - `system_io_test.tcyr`      (sysfs / proc reading)
  - `cost_model_test.tcyr`     (cost / recommend)
  - `json_output_test.tcyr`    (JSON serialization)
  - `model_format_test.tcyr`   (SafeTensors / GGUF / ONNX / PyTorch)
  - `requirement_test.tcyr`    (requirement matching)

  Extension change `.cyr` → `.tcyr` matches the agnosys / libro /
  cyrius-internal convention for standalone test units. Content is
  byte-identical to the 2.0.1 phase files — git tracks them as pure
  renames. All 518 assertions still pass.

### CI

- **`cyrius vet` step added** — walks every `include` from `src/main.cyr`
  and reports `<N> deps, <untrusted>, <missing>`. Today: `36 deps, 0
  untrusted, 0 missing`. Catches drift between the in-tree include
  graph and the manifest snapshot the moment a new module lands without
  being wired into main.cyr (and vice versa).
- **Test loop iterates `tests/tcyr/*.tcyr`** instead of the
  `test_phase*.cyr` glob.
- **fmt drift sweep widened** to cover `tests/tcyr/*.tcyr`,
  `fuzz/*.fcyr`, `benches/*.bcyr` alongside `src/`.
- **`cyrius.lock` step reworded** — was misleadingly called "soft-skip".
  cyrius only writes a lockfile for `[deps.<git>]` entries, and
  ai-hwaccel is stdlib-only, so there is nothing to verify until a git
  dep gets added. The step stays in place so verification engages
  automatically the day that happens.

### Not yet adopted (carried in roadmap)

- `cyrius capacity --check` — stalled on toolchain. cc5 5.10.x's
  capacity doesn't honour the manifest `[deps].stdlib` auto-prepend,
  so it warns on every stdlib symbol against `src/main.cyr`. Revisit
  when the warning floor is 0.
- `lib/test.cyr` adoption — current local `assert` helpers work; the
  `"N passed, M failed"` summary line is what CI greps for. Pure
  cleanup, not unblocking anything.

## [2.0.1] — 2026-05-10

**Toolchain bump — pure mechanical slot.** No source-code feature changes,
no API surface changes for consumers. Compiles clean against the cc5
generation; sets up the scaffolding 2.1.0's adoption arc needs.

### Changed

- **Cyrius pin: 3.10.0 → 5.10.34.** Two majors (3 → 4 → 5) and ten minors
  absorbed in one bump. Highlights of what landed upstream during that
  window (none adopted yet — see 2.1.0 roadmap):
  - **4.0.0** (2026-04-13): `#derive(accessors)`, native multi-return
    (`return (a, b)` / `var x, y = fn()`), `case N: { ... }` switch
    blocks with scoped vars, defer-on-all-paths.
  - **5.0.0** (2026-04-15): cc5 generation — basic-block IR + CFG +
    LASE between parse and emit (transparent today, optimization-ready).
    `cyrius.cyml` becomes the first-class manifest.
  - **5.1 → 5.10.34**: `${file:VERSION}` interpolation, `cyrius.lock`
    hash verification, `cyrius vet` / `cyrius capacity` / `cyrius audit`,
    stricter `cyrius fmt`, 20+ new stdlib modules
    (regex, toml, sandhi, net, audit_walk, test, …).
- **Manifest: `cyrius.toml` → `cyrius.cyml`** with `version = "${file:VERSION}"`
  interpolation. `VERSION` is now the single source of truth — no more
  multi-file sed in `scripts/version-bump.sh`. Adds `repository` field.
- **`lib/` vendored stdlib removed from tree** and `/lib/` gitignored.
  `cyrius deps` repopulates it from the version-pinned `[deps].stdlib`
  snapshot in `cyrius.cyml`. Matches agnosys / libro / patra / yukti
  convention. Run `cyrius deps` after a fresh clone or cyrius upgrade.
- **`.cyrius-toolchain` retired.** CI now reads the pin directly from
  `cyrius.cyml`'s `cyrius = "X.Y.Z"` line — one source of truth, no
  drift between manifest and toolchain file.
- **`scripts/version-bump.sh` simplified to a one-liner.** Writes only
  `VERSION`; the manifest auto-tracks via `${file:VERSION}`. Reminder
  to add the CHANGELOG section before tagging.
- **`CLAUDE.md`** — compiler reference updated to cc5 5.10.34. New
  principles documented: vendored stdlib in `lib/` is gitignored;
  bump only `VERSION`.

### CI / Release workflows

- **Canonical installer** — `.github/workflows/{ci,release}.yml` now
  install Cyrius via `https://raw.githubusercontent.com/MacCracken/cyrius/main/scripts/install.sh`
  rather than ad-hoc tarball fetch. Lays out `~/.cyrius/{bin,lib,versions}`
  with the symlinks `cyrius deps` expects (without those, stdlib
  resolution fails with `cannot read ./lib/<name>.cyr`).
- **`cyrius deps` step** added before any compile, with `cyrius deps --verify`
  gating against `cyrius.lock` (soft-skips on first push that introduces
  a new dep before the lockfile lands).
- **fmt drift gate** — `cyrius fmt <file>` emits to stdout and CI diffs
  against the committed file. Catches over-indented blocks the older
  fmt tolerated.
- **Version consistency gate** — asserts `cyrius.cyml` carries the
  `${file:VERSION}` literal and the `VERSION` value appears in
  `CHANGELOG.md`. Release workflow additionally enforces tag ↔ VERSION
  match (accepts both `vX.Y.Z` and `X.Y.Z` tag styles).
- **Multi-arch release** — best-effort aarch64 cross-build when
  `cc5_aarch64` is in the toolchain bundle. Releases ship src tarball
  + per-arch binaries + SHA256SUMS, with the `## [x.y.z]` CHANGELOG
  section auto-extracted as the release body.

### Fixed

- **`cyrius fmt` drift in 4 files** — `src/json_out.cyr`,
  `src/model_format.cyr`, `tests/test_phase10.cyr`,
  `tests/test_phase11.cyr` had inner blocks over-indented by 4 spaces
  past their opening brace (a pattern the older 3.x fmt tolerated;
  cc5 `cyrius fmt` enforces strict 4-space-per-scope). Test files also
  had assertion-message continuations indented under the call's arg
  list instead of aligned to the argument column. All four re-formatted
  in place; entire `src/` / `tests/` / `fuzz/` / `benches/` swept clean.
- **`str_contains` API migration** — cc5's `lib/str.cyr` changed the
  second argument from cstr to `Str` (v5.10.25 generalize pass). The
  cstr-needle behavior moved to `str_contains_cstr`. Migrated 14 files
  (11 detection backends + 3 test phases) — every site passing a cstr
  literal as the needle. `src/model.cyr:172` already passed a `Str` and
  was left unchanged. The old call sites compiled silently against
  3.10.0 because the type system didn't catch the i64 vs Str mismatch;
  the new stdlib still doesn't reject it but treats the cstr pointer
  as a `Str` struct address, reading garbage as length — every detect
  branch that gated on `str_contains(...)` was effectively a no-op
  until this fix.
- **`str_from(str_data(json))` round-trip removed in test_phase9** — cc5
  registers `str_from_int` as `str_from`'s `_int` overload, so
  `str_from(str_data(...))` (where `str_data` returns `i64`) routes to
  `str_from_int` and produces a `Str` containing the decimal string of
  the pointer value (e.g. `"727597728"`) instead of wrapping the data.
  The round-trip was always semantically pointless — `str_builder_build`
  already returns a `Str`. Six call sites in test_phase9 collapsed to
  use the builder result directly; restored 17 previously-passing
  assertions across `test_profile_json` / `test_registry_json` /
  `test_summary_json` / `test_json_with_warnings`.

## [2.0.0] - 2026-04-13

### Breaking

Complete rewrite from Rust to [Cyrius](https://github.com/MacCracken/cyrius).
The Rust crate (`ai-hwaccel` on crates.io) is superseded by a native Cyrius
binary with zero external dependencies. API surface is equivalent but calling
conventions have changed from Rust method syntax to Cyrius function calls.

**Migration**: `AcceleratorRegistry::detect()` → `registry_detect()`.
See `docs/benchmarks-rust-v-cyrius.md` for full API mapping.

### Added

- **Cyrius port** — entire codebase rewritten in Cyrius (v3.10.0). 37 source
  modules (18 core + 19 detect), 5,602 LOC. Zero external dependencies.
  Binary: 217 KB (was 708 KB Rust release). Compile time: 215 ms (was ~1.8s).
- **`model_format.cyr`** — model file format detection from headers. Parses
  SafeTensors (JSON header → param count, dtype, tensor count), GGUF (magic +
  version + tensor count + file_type metadata), ONNX (protobuf ir_version
  validation), and PyTorch (ZIP magic). Reads only first 16 KB. Both
  file-path and byte-slice APIs.
- **`requirement.cyr`** — accelerator requirement matching for scheduling
  integration. 7 requirement types: None, GPU, TPU (with min_chips), Gaudi,
  AwsNeuron, GpuOrTpu, AnyAccelerator. `requirement_satisfied()`,
  `find_satisfying_profile()`, `count_satisfying()`.
- **`async_detect.cyr`** — threaded concurrent hardware detection. CLI-based
  backends (nvidia-smi, hl-smi, vulkaninfo, neuron-ls, xpu-smi,
  system_profiler) run in parallel threads via `thread.cyr`. Sysfs-only
  backends run on the main thread. Results merged after join.
  `registry_detect_threaded()` API.
- **`cache.cyr`** — detection result caching with configurable TTL.
  `cached_registry_new(ttl_secs)`, `cached_get()`, `cached_invalidate()`.
  Mutex-protected, thread-safe. `DiskCachedRegistry` variant writes JSON to
  `~/.cache/ai-hwaccel/registry.json` with atomic write (temp + rename).
- **`lazy.cyr`** — per-family lazy detection. Defers backend probing until a
  specific accelerator family is queried. `lazy_new()`, `lazy_by_family()`,
  `lazy_into_registry()`. Avoids spawning nvidia-smi when only TPU info needed.
- **`model.cyr` extensions** — `models_by_family()` for family-based filtering,
  `model_headroom_x100()` for spare memory percentage,
  `compatible_with_registry()` for registry-aware compatibility.
- **`cost.cyr` + `model.cyr` re-included** — previously excluded due to Cyrius
  compiler fixup table overflow (4096 limit). Compiler v3.7.0 expanded limit
  to 16384, unblocking inclusion. `--cost` CLI flag now uses real
  `recommend_instances()` instead of a stub.
- **`json_out.cyr`** — JSON serialization via `str_builder` (replaces serde).
  `registry_to_json()`, `registry_to_summary_json()`, `profile_to_json()`,
  `model_meta_to_json()`.
- **11 test phases** — 518 assertions covering all modules. Phase 10: model
  format detection (46 assertions). Phase 11: requirement matching (27
  assertions).
- **6 fuzz harnesses** — `cuda_parser.fcyr`, `model_format.fcyr`,
  `vulkan_parser.fcyr`, `neuron_parser.fcyr`, `apple_parser.fcyr`,
  `gaudi_parser.fcyr`. Edge cases: empty input, garbage data, truncated
  headers, adversarial bytes.
- **3 benchmark suites** — core (8 benchmarks: memory estimation, quantization,
  training), parsing (5: CUDA CSV, Vulkan, Neuron JSON, SafeTensors, GGUF),
  registry (7: queries, sharding, JSON serialization). 20 benchmarks total.
- **`bench-history.sh`** — rewritten for Cyrius. Appends results to CSV with
  timestamp, commit, branch.
- **Str auto-coercion** — 61 `str_from("literal")` wrappers removed across
  src/ and tests/ using Cyrius v3.6.0 `: Str` parameter annotations.
- **Vendored stdlib synced** — all 26 vendored modules identical to upstream
  Cyrius. Added: `thread.cyr`, `async.cyr`, `fnptr.cyr` (fncall3/4),
  `freelist.cyr`. Synced: `str.cyr`, `syscalls.cyr`, `hashmap.cyr`, and 9
  others.

### Changed

- **Language**: Rust → Cyrius 3.10.0. No LLVM, no cargo, no crates.io.
- **Binary size**: 708 KB → 217 KB (**-69%**).
- **Compile time**: ~1.8s → 215 ms (**-88%**).
- **Source LOC**: 11,278 → 5,602 (**-50%**).
- **Dependencies**: 131 crates → 0 (**-100%**).
- **Detection modules consolidated**: cerebras + graphcore + groq → `cloud_asic.cyr`,
  qualcomm + samsung + mediatek → `edge.cyr`, intel_npu + intel_oneapi → `intel.cyr`.
- **Hardware types unified**: `hardware/mod.rs` + `hardware/*.rs` → `types.cyr`
  (enums, classification, throughput multipliers, HBM lookups, rank — all in one file).
- **Sharding merged**: `sharding.rs` + `plan.rs` → `plan.cyr`.
- **Fixed-point arithmetic** throughout — `x1000` multipliers replace all
  floating-point operations. No floats in the entire codebase.
- **Lint limit**: 100 → 120 characters (Cyrius 3.8.0). `#skip-lint` for
  unavoidable long strings.
- **`--cost` mode**: now shows real cloud instance recommendations with pricing
  (was stub directing to JSON file).

### Removed

- **Rust source** — `rust-old/` directory removed. Final Rust benchmarks
  preserved in `docs/benchmarks-rust-v-cyrius.md`.
- **FFI bindings** (`ffi.rs`) — not applicable, Cyrius is native code.
- **Windows detection** (`detect/windows.rs`) — Cyrius doesn't target Windows
  yet (v4.0.0 roadmap).
- **Cargo/crates.io** — no `Cargo.toml`, `Cargo.lock`, or Rust toolchain files.
- **serde/tokio/tracing** — replaced by manual JSON, thread.cyr, and direct
  stderr output respectively.

### Fixed

- **`requirement.cyr` undefined function** — `profile_tpu_chip_count()` called
  but function is `profile_tpu_chips()`. Would have crashed at runtime. Caught
  by Cyrius 3.10.0 undefined function diagnostic.
- **`async_detect.cyr` undefined functions** — `enrich_disk()` (should be
  `detect_storage()`), `detect_interconnect()` (should be
  `detect_interconnects()`). Same diagnostic.
- **`lazy.cyr` undefined function** — `builder_enable()` (should be
  `builder_with()`). Same diagnostic.

### Performance

All benchmarks on the same machine. Cyrius compiles to direct x86_64 without
LLVM optimization (no jump tables, no register allocation, no LTO).

| Benchmark | Rust (LLVM) | Cyrius | Ratio |
|-----------|-------------|--------|-------|
| estimate_memory 70B BF16 | 256 ps | 8 ns | 31x |
| bits_per_param (5 levels) | 295 ps | 3 ns | 10x |
| train_7B_full_gpu | 3.29 ns | 30 ns | 9x |
| parse_cuda_output 8gpu | 5.80 µs | 18 µs | 3x |
| parse_vulkan_2gpu | 1.85 µs | 3 µs | 1.6x |
| best_available (13 dev) | 39.56 ns | 722 ns | 18x |
| total_memory (13 dev) | 7.88 ns | 122 ns | 15x |
| json_serialize (13 dev) | 4.89 µs | 27 µs | 6x |
| detect_safetensors | — | 931 ns | new |
| detect_gguf | — | 498 ns | new |

All times sub-microsecond to microsecond. Detection dominated by 100ms+ CLI
tool execution — per-call overhead is irrelevant to end-to-end latency.

## [1.2.0] - 2026-04-05

### Added

- **NVSwitch auto-detection** — probes sysfs
  `/sys/devices/virtual/nvidia-nvswitch/` and parses `nvidia-smi topo -m`
  output. High NVLink counts (NV8+) with multiple GPUs indicate NVSwitch
  fabric. Both sync and async detection paths supported.
- **AMD XGMI / Infinity Fabric detection** — reads XGMI hive IDs from
  `/sys/class/drm/card*/device/xgmi_hive_info` and parses
  `rocm-smi --showtopo` link-type matrix as fallback.
- **Google TPU ICI detection** — multi-chip TPU configurations now report
  `InterconnectKind::Ici` with version-specific bandwidth (v4: 192 GB/s/chip,
  v5e: 204.8, v5p: 409.6).
- **RoCE v2 detection** — new `InterconnectKind::RoCEv2` variant. Distinguished
  from RoCE v1 by reading sysfs `gid_attrs/types` for "RoCE v2" entries.
- **Fuzz targets** for NVSwitch topology and XGMI topology parsers.
- **Model compatibility database** (`model_compat` module) — embedded catalogue
  of 26 popular models (Llama, Mistral, Gemma, Phi, Qwen, DeepSeek, Falcon,
  etc.) with `can_run()`, `compatible_models()`, `find_model()`,
  `compatible_with_registry()` APIs. Answers "can I run Llama 70B on 2x RTX
  4090?" without manual memory math.
- **Model format detection** (`model_format` module) — parses `.safetensors`,
  `.gguf`, `.onnx`, and `.pt` file headers to extract format, parameter count,
  data type, and tensor count. Both file-path and byte-slice APIs (WASM-safe).
- **WASM target support** — `wasm32-unknown-unknown` builds cleanly with all
  features. `from_profiles()`, `from_json()`, planning, sharding, cost,
  training, model compat, and model format detection all work in WASM.
- **Kubernetes GPU detection** — detects GPU devices allocated via Kubernetes
  device plugins (`NVIDIA_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`,
  `GPU_DEVICE_ORDINAL`). New `KubernetesGpuInfo` struct in
  `RuntimeEnvironment`.
- **What-if analysis** — `what_if_add()`, `what_if_remove()`,
  `what_if_replace()` methods on `AcceleratorRegistry` for simulating hardware
  changes and re-planning sharding strategies.
- **Fuzz target** for model format byte-level parser.
- **130+ new tests** — interconnect parsers (NVSwitch, XGMI, ICI, RoCE v2),
  model compatibility database, model format detection (SafeTensors, GGUF,
  ONNX, PyTorch), what-if analysis, edge cases. Total: 538 tests.

### Changed

- **`DetectBuilder` uses `u32` bitmask** — replaces `Vec<bool>` with zero-alloc
  bitmask. `DetectBuilder` is now `Copy`. `enabled_count()` uses
  `count_ones()` instead of iterator filter.
- **Sharding planner now accounts for all interconnect types** —
  `InterconnectInfo::scan()` incorporates InfiniBand, RoCE, RoCEv2, and ICI
  bandwidth into `high_bw` for better sharding decisions.
- **Sharding throughput uses `reduce()` instead of `fold(INFINITY)`** —
  prevents NaN/Inf when device lists are empty. Throughput estimates now
  return `None` instead of `Some(NaN)` when non-finite.
- **Single-pass Vulkan/dedicated GPU check** — combined 3 separate iterator
  passes into one `fold` in `detect_with_builder` and timed variant.
- **Simplified `read_dir` patterns** — replaced 12 `read_dir().into_iter()
  .flatten().flatten()` double-flatten patterns across 5 files with idiomatic
  `let Ok(entries) = read_dir() else { return }`.
- **Async interconnect detection collects warnings** — CLI tool errors
  (non-tool-not-found) are now propagated as warnings in the async path.

### Fixed

- **`parse_nvswitch_topo` header row inflation** — the GPU column header
  (`GPU0 GPU1 ...`) was incorrectly counted as a data row, inflating the
  reported GPU count by 1.
- **`detect_xgmi_sysfs` false positive** — returned `true` (suppressing CLI
  fallback) even when all XGMI hives had only 1 GPU.
- **`detect_tpu_ici` non-TPU device counting** — non-TPU accelerator devices
  under `/dev/accel` are now skipped by requiring `tpu_version` sysfs file.

### Security

- **`read_sysfs_string` hard cap** — added 1 MiB absolute maximum to prevent
  DoS from callers passing huge `max_bytes` values.
- **NVLink link-count cap** — `parse_nvlink_output` caps at 256 links per GPU
  to bound bandwidth accumulation from malformed output.

### Dependencies

- Certified `semver 1.0.28` and `fastrand 2.4.0` via `cargo vet`.

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
