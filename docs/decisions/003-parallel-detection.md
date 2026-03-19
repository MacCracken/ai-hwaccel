# ADR-003: Parallel detection via std::thread::scope

## Status

Accepted

## Context

Detection involves running multiple CLI tools (`nvidia-smi`, `hl-smi`,
`vulkaninfo`, etc.) and probing sysfs. Each operation is independent I/O that
can take 100ms+. Running them sequentially means total latency is the sum of
all backend latencies.

Options considered:

1. **Sequential** — simple, no threading.
2. **`std::thread::scope`** — parallel, zero extra dependencies.
3. **`tokio`/`async-std`** — async I/O, but requires an async runtime.
4. **`rayon`** — parallel iterators, but overkill for I/O-bound work.

## Decision

We use `std::thread::scope` (stable since Rust 1.63). Each backend spawns a
scoped thread that returns `(Vec<Profile>, Vec<Warning>)`. Results are merged
after all threads join.

## Consequences

**Benefits:**

- Total latency is the *maximum* of individual backend latencies instead of the
  sum. On a system with CUDA + Gaudi + Vulkan, detection drops from ~15s to ~5s.
- No external dependencies.
- Scoped threads guarantee no dangling references — the borrow checker enforces
  lifetime safety.

**Trade-offs:**

- The Vulkan detector previously checked whether CUDA/ROCm was already detected
  to avoid double-counting. With parallel execution, this ordering dependency
  doesn't work. Solved with a **post-pass** that removes Vulkan GPUs if a
  dedicated CUDA/ROCm GPU was also found.
- Thread spawn overhead (~50µs per backend) is negligible compared to the
  100ms+ I/O cost.
- Error handling is slightly more complex (thread panics are caught by `join()`).

**Future:**

- An async variant behind a `tokio` feature flag is planned for callers already
  running an async runtime.
