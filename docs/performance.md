# Performance Tuning

How to minimize detection latency and binary size.

---

## When to use `CachedRegistry`

`AcceleratorRegistry::detect()` spawns subprocesses and reads sysfs on every
call. On a system with multiple backends, this takes 50–200ms.

Use `CachedRegistry` when you call detection more than once:

```rust
use ai_hwaccel::CachedRegistry;
use std::time::Duration;

// Detect once, reuse for 5 minutes.
let cache = CachedRegistry::new(Duration::from_secs(300));
let reg = cache.get();  // first call: detects
let reg = cache.get();  // subsequent calls: returns cached
```

Call `cache.invalidate()` to force a fresh detection (e.g., after hot-plug).

---

## Selective backend detection

If you know which hardware is present, skip unnecessary backends:

```rust
use ai_hwaccel::AcceleratorRegistry;

// Only probe CUDA — skips all other backends.
let registry = AcceleratorRegistry::builder()
    .with_cuda()
    .detect();
```

This avoids spawning processes for tools that aren't installed (which would
each add ~5ms of `$PATH` scanning overhead).

For the fastest possible detection (no backends, just CPU):

```rust
let registry = AcceleratorRegistry::builder()
    .detect(); // no with_*() calls → CPU only

// Or equivalently:
use ai_hwaccel::DetectBuilder;
let registry = DetectBuilder::none().detect();
```

---

## Feature flags and binary size

Each backend is a cargo feature. Disabling unused backends removes their
detection code and reduces binary size:

```toml
[dependencies]
ai-hwaccel = { version = "0.20", default-features = false, features = ["cuda", "rocm"] }
```

This is especially useful in embedded or container contexts where only
specific hardware is present.

| Feature | Adds | Subprocess? |
|---------|------|-------------|
| `cuda` | ~200 lines | Yes (`nvidia-smi`) |
| `rocm` | ~60 lines | No (sysfs only) |
| `apple` | ~150 lines | Yes (`system_profiler`) |
| `vulkan` | ~140 lines | Yes (`vulkaninfo`) |
| `tpu` | ~80 lines | No (sysfs only) |
| `gaudi` | ~75 lines | Yes (`hl-smi`) |
| `aws-neuron` | ~95 lines | Yes (`neuron-ls`) |
| `intel-npu` | ~25 lines | No (sysfs only) |
| `amd-xdna` | ~45 lines | No (sysfs only) |
| `intel-oneapi` | ~55 lines | Yes (`xpu-smi`) |
| `qualcomm` | ~40 lines | No (sysfs only) |

Sysfs-only backends have near-zero overhead (~1ms). CLI-based backends add
up to 5 seconds timeout per tool (but typically complete in <100ms).

---

## Async detection

With the `async-detect` feature, use `detect_async()` to avoid blocking
the tokio runtime:

```toml
[dependencies]
ai-hwaccel = { version = "0.20", features = ["async-detect"] }
```

```rust,ignore
let registry = AcceleratorRegistry::detect_async().await?;
```

This is important in async applications where blocking the runtime would
stall other tasks.

---

## Thread count

`detect()` uses `std::thread::scope` to run backends in parallel. When 2+
backends are enabled, each gets its own scoped thread. Scoped threads are
lightweight (no `Arc` overhead) and are joined before `detect()` returns.

If you're already in a thread-constrained environment, use
`DetectBuilder::none()` with specific `with_*()` calls to limit concurrency,
or call detection from a dedicated thread.
