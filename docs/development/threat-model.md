# Threat Model

This document describes the security boundaries, trust assumptions, and
mitigations in `ai-hwaccel`.

## Attack surface

`ai-hwaccel` interacts with the host system in these ways:

1. **Reading filesystem paths** — `/proc/meminfo`, `/sys/class/drm/`, `/dev`
   device nodes, and on request the data files (`VERSION`,
   `data/cloud_pricing.json`, `data/models.json`, from `--data-dir` or
   `AI_HWACCEL_DATA_DIR`).
2. **Calling OS APIs** — `sysctl` on macOS, DXGI and `GlobalMemoryStatusEx`
   on Windows.
3. **Executing external tools** — `nvidia-smi`, `hl-smi`, `vulkaninfo`,
   `neuron-ls`, `xpu-smi`, `cerebras_cli`, `gc-info`, and the
   `system_profiler` (macOS) and `wmic` (Windows) fallbacks.
4. **Deserializing JSON** — `profile_from_json_str`, when a caller loads a
   saved profile.
5. **Writing one file** — only the opt-in disk cache (`disk_cached_get`)
   writes: `~/.cache/ai-hwaccel/registry.json`, or
   `/tmp/ai-hwaccel-cache.json` when `HOME` is unset. The CLI never uses it.

There is no network access.

## Trust assumptions

- The **kernel** and **sysfs/procfs** are trusted. A compromised kernel is out
  of scope.
- The **`$PATH` environment** is set by a trusted process (init, shell,
  container runtime). Whatever `$PATH` resolves a tool name to is what runs.
- **Tool output** is **untrusted input**: even legitimate tools can produce
  unexpected output because of version differences, locale settings or
  hardware quirks.
- **Deserialized JSON** is **untrusted input** when loaded from disk or the
  network.

## Mitigations

### Command execution

| Threat | Mitigation |
|---|---|
| `$PATH` hijacking (a malicious `nvidia-smi` earlier on `$PATH`) | Not mitigated beyond trusting `$PATH`: a tool is resolved once through `$PATH` and run by absolute path. On Linux it runs with an empty environment, so `LD_PRELOAD` and similar variables are not passed on. |
| Hung or slow tool (e.g. `nvidia-smi` stalls when a GPU is in an error state) | **Not mitigated in the binary:** tools run without a time limit (bounding them is on the roadmap). `registry_detect_no_exec()` runs no tool. The Python package bounds each CLI run (30 s by default). On Linux a tool is killed if ai-hwaccel itself dies (`PR_SET_PDEATHSIG`). |
| Excessive output (a tool writes gigabytes to stdout) | stdout is capped at **1 MiB** (`CMD_MAX_STDOUT`); the rest is not read. stderr goes to `/dev/null`. **Gap:** output that fills the cap is cut off without a warning, and its NUL terminator lands one byte past the buffer. The file reads have the same flaw at their caps (4 KiB for sysfs, 32 KiB for `cloud_pricing.json`, 64 KiB for the disk cache). Fixing both is on the roadmap. |
| Malformed output (unexpected CSV, JSON, or text format) | Parsed fields are validated: device IDs must be 0 to 1024, memory values 0 to 16 TiB. A bad line is skipped with a parse warning. |
| Injected arguments | No user-controlled data is interpolated into tool arguments: every argument is a string literal. |

### JSON deserialization

| Threat | Mitigation |
|---|---|
| Unknown or missing fields | `profile_from_json` reads the keys it knows through bayan-json, ignores the rest, and uses defaults for missing ones. |
| Excessively large payloads | Callers should impose their own size limits before parsing. ai-hwaccel's own reads are capped: 32 KB for `cloud_pricing.json` and `models.json`, 16 KB for model-format headers. |

### File writes

| Threat | Mitigation |
|---|---|
| Symlink attack on the disk cache's `/tmp` fallback (another local user creates `/tmp/ai-hwaccel-cache.json` as a symlink, and the cache write truncates its target) | **Not mitigated:** the write follows symlinks. It happens only when a caller uses the disk cache with `HOME` unset. Fixing it (a private directory, or refusing to write without `HOME`) is on the roadmap. |

### Supply chain

| Threat | Mitigation |
|---|---|
| Vulnerable or tampered dependencies | No third-party dependencies. The vendored stdlib comes from the pinned Cyrius toolchain snapshot, and bayan's JSON sublib (first-party, optional) is pinned by commit; `cyrius.lock` records a SHA-256 for every vendored file, and CI checks them on every run. |
| License violations | Single license (GPL-3.0-only). No third-party code. |
| Malicious Python package | The PyPI wheels are built by this repository's `wheels.yml`. The package itself has no Python dependencies (pandas is an optional extra). |

## Out of scope

- **Kernel/driver compromise** — if the kernel is compromised, sysfs data is
  unreliable and `ai-hwaccel` cannot detect this.
- **Physical hardware tampering** — ai-hwaccel trusts hardware
  self-reporting (e.g. VRAM size from `nvidia-smi`).
- **Container escapes** — ai-hwaccel probes the container's view of `/sys`
  and `/dev`, which is whatever the container runtime exposes.
