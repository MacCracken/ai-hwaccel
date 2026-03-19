# ADR-001: sysfs and CLI tools over vendor SDKs

## Status

Accepted

## Context

Hardware detection can be done two ways:

1. **Link vendor SDKs** (CUDA Runtime, ROCm HIP, etc.) at compile time and call
   their device-query APIs.
2. **Probe the OS** via sysfs, `/dev`, and vendor CLI tools (`nvidia-smi`,
   `hl-smi`, etc.) at runtime.

## Decision

We use approach 2: sysfs + CLI tools.

## Consequences

**Benefits:**

- Zero compile-time dependencies on vendor toolchains. The crate builds on any
  system regardless of what drivers are installed.
- Cross-compilation is trivial — no need to cross-link CUDA libraries.
- Adding a new backend is a single file with no build system changes.
- Works in containers and VMs where only the kernel driver (not the full SDK)
  is exposed.

**Trade-offs:**

- Less precise than SDK APIs — we parse text output instead of calling typed
  functions.
- CLI tools may not be installed even when the hardware is present.
- Version-specific output format changes can break parsers.
- Cannot query advanced capabilities (NVLink topology, clock speeds) that are
  only exposed via SDK APIs.

**Mitigations:**

- Structured `DetectionError` warnings let callers know when a tool is missing.
- Input validation rejects out-of-range values from malformed output.
- Each backend is isolated in its own module so parser fixes don't affect others.
