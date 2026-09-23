# ADR-004: Compile-time flags per hardware backend

## Status

Not implemented. Nothing in `src/` reads `CUDA`, `ROCM`, `TPU` or
`NO_BACKENDS`, so a `-D` flag changes nothing and every backend is compiled
in. Backends are selected at run time instead, with a builder mask
(`builder_without(builder_all(), BACKEND_VULKAN)` passed to
`registry_detect_with`). Implementing these gates, or retiring this record,
is on the roadmap.

## Context

Not every consumer needs all detection backends. A Kubernetes scheduler
running on GPU-only nodes doesn't need TPU or Neuron detection code. Unused
backends add compile time and (for CLI tools) unnecessary subprocess spawns.

## Decision

Each backend is gated behind a `#ifdef` / `-D` compile-time flag. All are
enabled by default.

```sh
# Build with all backends (default)
cyrius build src/main.cyr build/ai-hwaccel

# Build with only CUDA and TPU
cyrius build src/main.cyr build/ai-hwaccel -DCUDA -DTPU

# Build with no backends (CPU-only)
cyrius build src/main.cyr build/ai-hwaccel -DNO_BACKENDS
```

When a flag is not set, the corresponding `detect/*.cyr` module is not
compiled (`#ifdef BACKEND_NAME`) and the backend is never spawned.

## Consequences

**Benefits:**

- Consumers can build with only the backends they care about:
  `-DCUDA -DTPU`.
- Faster compile times when most backends are disabled.
- Smaller binary size (dead code is eliminated).
- The `DetectBuilder` runtime toggle and `-D` compile-time toggle
  compose: flags control what *can* run, builder controls what *does* run.

**Trade-offs:**

- More CI surface: we should test `-DNO_BACKENDS` and a few
  representative flag combos.
- The `Backend` enum and `DetectBuilder` still exist even when backends are
  disabled — they just become no-ops.
