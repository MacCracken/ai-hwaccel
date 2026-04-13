# ADR-003: Parallel detection via thread.cyr

## Status

Accepted

## Context

Detection involves running multiple CLI tools (`nvidia-smi`, `hl-smi`,
`vulkaninfo`, etc.) and probing sysfs. Each operation is independent I/O that
can take 100ms+. Running them sequentially means total latency is the sum of
all backend latencies.

Options considered:

1. **Sequential** — simple, no threading.
2. **`thread.cyr`** — parallel, zero extra dependencies, built into Cyrius.
3. **External async runtime** — async I/O, but adds a dependency.

## Decision

We use `thread.cyr` (Cyrius built-in threading). Each backend spawns a
scoped thread that returns `(Vec<Profile>, Vec<Warning>)`. Results are merged
after all threads join.

## Consequences

**Benefits:**

- Total latency is the *maximum* of individual backend latencies instead of the
  sum. On a system with CUDA + Gaudi + Vulkan, detection drops from ~15s to ~5s.
- No external dependencies — threading is built into Cyrius.
- Scoped threads guarantee no dangling references.

**Trade-offs:**

- The Vulkan detector previously checked whether CUDA/ROCm was already detected
  to avoid double-counting. With parallel execution, this ordering dependency
  doesn't work. Solved with a **post-pass** that removes Vulkan GPUs if a
  dedicated CUDA/ROCm GPU was also found.
- Thread spawn overhead (~50µs per backend) is negligible compared to the
  100ms+ I/O cost.
- Error handling is slightly more complex (thread panics are caught by `join()`).

**Future:**

- An async variant via `async.cyr` is available for callers that need
  non-blocking detection (`registry_detect_threaded()`).
