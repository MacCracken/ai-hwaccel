# ADR-004: Feature flags per hardware backend

## Status

Accepted

## Context

Not every consumer needs all 11 detection backends. A Kubernetes scheduler
running on GPU-only nodes doesn't need TPU or Neuron detection code. Unused
backends add compile time and (for CLI tools) unnecessary subprocess spawns.

## Decision

Each backend is gated behind a cargo feature flag. All are enabled by default
via the `all-backends` feature.

```toml
[features]
default = ["all-backends"]
all-backends = ["cuda", "rocm", "apple", "vulkan", "intel-npu", ...]
cuda = []
rocm = []
# ...
```

When a feature is disabled, the corresponding `detect/*.rs` module is not
compiled (`#[cfg(feature = "...")]`) and the backend is never spawned.

## Consequences

**Benefits:**

- Downstream crates can opt into only the backends they care about:
  `features = ["cuda", "tpu"]`.
- Faster compile times when most backends are disabled.
- Smaller binary size (dead code is eliminated).
- The `DetectBuilder` runtime toggle and feature-flag compile-time toggle
  compose: features control what *can* run, builder controls what *does* run.

**Trade-offs:**

- More CI surface: we should test `--no-default-features` and a few
  representative feature combos.
- The `Backend` enum and `DetectBuilder` still exist even when backends are
  disabled — they just become no-ops.
