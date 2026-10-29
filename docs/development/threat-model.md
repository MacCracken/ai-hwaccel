# Threat Model

This document describes the security boundaries, trust assumptions, and
mitigations in `ai-hwaccel`.

## Attack surface

`ai-hwaccel` interacts with the host system in these ways:

1. **Reading filesystem paths** — on Linux, `/proc/meminfo`, the cgroup
   memory limit (`/sys/fs/cgroup`), `/sys/class/drm/`, `/dev` device nodes;
   on every target, on request, the data files (`VERSION`,
   `data/cloud_pricing.json`, `data/models.json`, from `--data-dir`,
   `data_dir_set()` or `AI_HWACCEL_DATA_DIR`).
2. **Calling OS APIs** — `sysctl` on macOS, DXGI and `GlobalMemoryStatusEx`
   on Windows.
3. **Executing external tools** — `nvidia-smi`, `hl-smi`, `vulkaninfo`,
   `neuron-ls`, `xpu-smi`, `cerebras_cli`, `gc-info`, and the
   `system_profiler` (macOS) and `wmic` (Windows) fallbacks.
4. **Deserializing JSON** — `profile_from_json_str`, when a caller loads a
   saved profile.
5. **Writing files** — none since 2.5.1. Through 2.5.0 the opt-in disk cache
   (`disk_cached_get`) wrote `~/.cache/ai-hwaccel/registry.json`, or
   `/tmp/ai-hwaccel-cache.json` when `HOME` was unset; it now caches in
   memory only.

There is no network access.

## Trust assumptions

- The **kernel** and **sysfs/procfs** are trusted. A compromised kernel is out
  of scope. They are read on Linux (and AGNOS) only. On Windows a rooted
  `/sys/...` path names a folder at the root of the current drive, which any
  local user may create, so PE builds read none of them (2.5.5): through
  2.5.4 a planted `C:\sys` fed the report a fake NVMe device or cloud
  provider, and could crash it through the XGMI probe.
- The **`$PATH` environment** is set by a trusted process (init, shell,
  container runtime). Whatever `$PATH` resolves a tool name to is what runs.
- A **library consumer's argv** is the host program's own: the library never
  reads it (2.5.5). Only the CLI turns `--data-dir` into `data_dir_set()`.
- **Tool output** is **untrusted input**: even legitimate tools can produce
  unexpected output because of version differences, locale settings or
  hardware quirks.
- **Deserialized JSON** is **untrusted input** when loaded from disk or the
  network.

## Mitigations

### Command execution

| Threat | Mitigation |
|---|---|
| `$PATH` hijacking (a malicious `nvidia-smi` earlier on `$PATH`) | Not mitigated beyond trusting `$PATH`: a tool is resolved once through `$PATH` and run by absolute path. Relative and empty `PATH` entries are skipped (2.5.3), so the current directory is never searched. On POSIX the first regular file the process may execute wins (`stat` and `access(X_OK)`, 2.5.5); through 2.5.4 it was the first path that opened, so a directory or a non-executable file of that name shadowed the real tool, and a FIFO blocked the lookup outside any deadline. On Windows a `PATH` of 4096+ characters is read whole (2.5.5; the stdlib `getenv` answered 0 for it, so no tool was found). On Windows the path is quoted on the command line when it holds a blank: unquoted, Windows starts the first space-delimited prefix that exists, which a planted file could fill. `wmic` is resolved through `PATH` too; through 2.5.2 it was started by its bare name, which Windows looks up in the current directory first. On Linux a tool runs with an empty environment, so `LD_PRELOAD` and similar variables are not passed on; on Windows it inherits the caller's environment. |
| Hung or slow tool (e.g. `nvidia-smi` stalls when a GPU is in an error state) | A tool silent for 10 s (`CMD_DEFAULT_TIMEOUT_MS`) is cut, then killed with its process group and reported as a timeout warning (2.5.1); what it read is dropped. On POSIX the bound is idle-based (a tool still printing is not cut; the output cap, next row, bounds it), and the cut is followed by a second 10 s wait, TERM, up to 5 s of grace and KILL: about 20 s per hung tool, 25 s if it ignores TERM. The reap after KILL is not bounded, so a tool stuck in the kernel (D state) is not either. On Windows the bound is wall-clock over the whole run. AGNOS has no deadline. `registry_detect_no_exec()` runs no tool. The Python package bounds each CLI run: 60 s by default since 2.5.5, up from 30 s, which two hung tools exceeded. On Linux a tool is killed if ai-hwaccel itself dies (`PR_SET_PDEATHSIG`). Issue 2026-10-29-cyrius-capture-deadline-shape. |
| Excessive output (a tool writes gigabytes to stdout) | stdout is capped one byte short of **1 MiB** (`CMD_MAX_STDOUT`), so its NUL fits; the rest is not read, the tool dies of SIGPIPE, and a warning says the output was cut short. The 1 MiB buffer is released after each run and only the output is kept (2.5.5; through 2.5.4 every run kept 1 MiB, so a long-lived `cached_get` consumer grew without bound). stderr goes to `/dev/null`; on Windows it shares the stdout pipe, and so the cap. The file reads stop one byte short of their buffers too (4 KiB for sysfs and `/proc/meminfo`, 32 KiB for `cloud_pricing.json`, which warns when full). Through 2.5.0 each of them put the NUL one byte past the buffer. |
| Malformed output (unexpected CSV, JSON, or text format) | Parsed fields are validated: device IDs must be 0 to 1024, memory values 0 to 16 TiB. A bad line is skipped with a parse warning. |
| Injected arguments | No user-controlled data is interpolated into tool arguments: every argument is a string literal. |
| Concurrent tools (`registry_detect_threaded`, `--threaded`) | Tools run one at a time (2.5.5). The stdlib's capture keeps its idle cut and kill-tree snapshot in globals and does not make its pipe close-on-exec, so through 2.5.4 a tool that hung on one detector thread made a healthy tool on another read as a timeout, and its devices were dropped. The detector threads still overlap everything around the tools. Issue 2026-10-29-cyrius-process-capture-state-not-thread-safe. |

### JSON deserialization

| Threat | Mitigation |
|---|---|
| Unknown or missing fields | `profile_from_json` reads the keys it knows through bayan-json, ignores the rest, and uses defaults for missing ones. |
| Excessively large payloads | Callers should impose their own size limits before parsing. ai-hwaccel's own reads are capped: 32 KB for `cloud_pricing.json` and `models.json`, 16 KB for model-format headers. |

### File writes

| Threat | Mitigation |
|---|---|
| Symlink attack on the disk cache's `/tmp` fallback (another local user creates `/tmp/ai-hwaccel-cache.json` as a symlink, and the cache write truncates its target) | **Removed in 2.5.1:** the disk cache writes no file. Through 2.5.0 the write followed symlinks when a caller used the disk cache with `HOME` unset. |

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
