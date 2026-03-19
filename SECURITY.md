# Security Policy

## Scope

`ai-hwaccel` runs shell commands (`nvidia-smi`, `hl-smi`, `vulkaninfo`,
`neuron-ls`) and reads sysfs/procfs paths during hardware detection. Although
these operations are read-only and non-destructive, bugs in command parsing or
path handling could have security implications.

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

- **Command execution**: Detection functions execute external tools with no
  user-controlled arguments. Ensure `$PATH` is trusted in your deployment
  environment.
- **sysfs/procfs reads**: The crate reads system files but never writes to them.
- **Serialization**: `AcceleratorProfile` and related types derive `Serialize`
  and `Deserialize`. If you deserialize untrusted input, apply your own
  validation layer.
