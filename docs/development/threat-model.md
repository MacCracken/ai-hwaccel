# Threat Model

This document describes the security boundaries, trust assumptions, and
mitigations in `ai-hwaccel`.

## Attack surface

`ai-hwaccel` interacts with the host system in three ways:

1. **Reading filesystem paths** — `/proc/meminfo`, `/sys/class/drm/`, etc.
2. **Executing external CLI tools** — `nvidia-smi`, `hl-smi`, `vulkaninfo`,
   `neuron-ls`, `xpu-smi`, `system_profiler`.
3. **Deserializing JSON** — when loading a saved `AcceleratorRegistry`.

## Trust assumptions

- The **kernel** and **sysfs/procfs** are trusted. We read but never write
  system files. A compromised kernel is out of scope.
- The **`$PATH` environment** is assumed to be set by a trusted process
  (init, shell, container runtime). However, we mitigate `$PATH` hijacking
  (see below).
- **CLI tool output** is **untrusted input** — even legitimate tools can
  produce unexpected output due to version differences, locale settings, or
  hardware quirks.
- **Deserialized JSON** is **untrusted input** when loaded from disk or
  network.

## Mitigations

### Command execution

| Threat | Mitigation |
|---|---|
| `$PATH` hijacking (attacker places malicious `nvidia-smi` in a directory earlier on `$PATH`) | Tools are resolved to their **absolute path** at detection time via `which()` and invoked by absolute path. The resolved path is logged at `debug` level for audit. |
| Hung or slow tool (e.g. `nvidia-smi` stalls when GPU is in error state) | All subprocess calls enforce a **5-second timeout** (configurable via `DEFAULT_TIMEOUT`). The child process is killed via `child.kill()` after timeout. |
| Excessive output (tool writes gigabytes to stdout) | stdout is capped at **1 MiB** (`MAX_STDOUT_BYTES`). stderr is capped at **4 KiB**. Reads beyond the limit are silently discarded. |
| Malformed output (unexpected CSV, JSON, or text format) | All parsed fields are validated: device IDs must be 0--1024, memory values must be 0--16 TiB. Parse failures produce `DetectionError::ParseError` warnings, not panics. |
| Injected arguments | No user-controlled data is ever interpolated into subprocess arguments. All arguments are compile-time string literals. |

### Serde deserialization

| Threat | Mitigation |
|---|---|
| Unknown fields in JSON (forward-compatibility probing) | All struct types use `#[serde(deny_unknown_fields)]` — unexpected fields cause deserialization to fail rather than be silently ignored. |
| Excessively large payloads | Callers should impose their own size limits before passing data to `serde_json::from_str`. The crate itself does not read from network or disk. |

### Supply chain

| Threat | Mitigation |
|---|---|
| Vulnerable dependencies | `cargo-audit` runs in CI on every push. |
| License violations | `cargo-deny` enforces an allowlist of OSS licenses (`MIT`, `Apache-2.0`, `BSD-*`, `ISC`). |
| Typosquatting / malicious crates | `cargo-deny` restricts to the official crates.io registry (`deny.toml` → `[sources]`). |

## Out of scope

- **Kernel/driver compromise** — if the kernel is compromised, sysfs data is
  unreliable and `ai-hwaccel` cannot detect this.
- **Physical hardware tampering** — the crate trusts hardware self-reporting
  (e.g. VRAM size from `nvidia-smi`).
- **Denial of service via detection latency** — detection runs CLI tools which
  may take up to 5 s each. On a system with many backends, total wall-clock
  time could be several seconds even with parallel execution.
- **Container escapes** — the crate probes the container's view of `/sys` and
  `/dev`, which is whatever the container runtime exposes.
