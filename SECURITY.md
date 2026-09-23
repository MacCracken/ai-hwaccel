# Security Policy

## Scope

During detection `ai-hwaccel` reads sysfs, procfs and `/dev` paths, calls OS
APIs (`sysctl` on macOS, DXGI and `GlobalMemoryStatusEx` on Windows) and runs
vendor tools found on `$PATH`: `nvidia-smi`, `vulkaninfo`, `hl-smi`,
`neuron-ls`, `xpu-smi`, `cerebras_cli`, `gc-info`, and the `system_profiler`
(macOS) and `wmic` (Windows) fallbacks. It makes no network connections. Bugs
in output parsing or path handling could still have security implications.

[docs/development/threat-model.md](docs/development/threat-model.md)
describes the trust boundaries, the mitigations and the known gaps.

## Supported versions

Only the latest released version receives security fixes.

| Version | Supported |
|---|---|
| Latest | Yes |
| Older | No |

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Instead, please report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/MacCracken/ai-hwaccel/security/advisories/new)
or by emailing the maintainer directly.

Include:

- A description of the vulnerability.
- Steps to reproduce or a proof of concept.
- The potential impact.

You should receive an acknowledgement within 72 hours. We aim to release a fix
within 14 days of confirmation.

## Security considerations

- **Command execution**: tools are resolved through `$PATH` and run without a
  shell, with fixed arguments; on Linux they get an empty environment. Make
  sure `$PATH` is trusted in your deployment. Tools run without a time limit,
  so a hung tool blocks detection: `registry_detect_no_exec()` runs none, and
  the Python package bounds each call (30 s by default).
- **sysfs/procfs reads**: the library reads system files but never writes to
  them.
- **File writes**: the CLI writes nothing. Only the opt-in disk cache
  (`disk_cached_get`) writes a file: `~/.cache/ai-hwaccel/registry.json`, or
  `/tmp/ai-hwaccel-cache.json` when `HOME` is unset. The write follows
  symlinks, so don't use the disk cache without `HOME` on a shared host.
- **Deserialization**: `profile_from_json_str` parses with bayan-json, reads
  the keys it knows and ignores the rest. If you deserialize untrusted input,
  apply your own size limits and validation.
