# Performance Tuning

How to minimize detection latency and binary size.

---

## When to use `CachedRegistry`

`registry_detect()` spawns subprocesses and reads sysfs on every call. On a
system with multiple backends, this takes 50–200ms.

Use `CachedRegistry` when you call detection more than once:

```cyr
// Detect once, reuse for 5 minutes.
let cache = CachedRegistry::new(300);
let reg = cache.get();  // first call: detects
let reg = cache.get();  // subsequent calls: returns cached
```

Call `cache.invalidate()` to force a fresh detection (e.g., after hot-plug).

---

## Selective backend detection

If you know which hardware is present, skip unnecessary backends:

```cyr
// Only probe CUDA — skips all other backends.
let registry = registry_detect_builder()
    .with_cuda()
    .detect();
```

This avoids spawning processes for tools that aren't installed (which would
each add ~5ms of `$PATH` scanning overhead).

For the fastest possible detection (no backends, just CPU):

```cyr
let registry = registry_detect_builder()
    .detect(); // no with_*() calls -> CPU only

// Or equivalently:
var registry = registry_detect_with(builder_none());
```

---

## Feature flags and binary size

Each backend is controlled by a `-D` compile-time flag. Disabling unused
backends removes their detection code and reduces binary size:

```sh
cyrius build src/main.cyr build/ai-hwaccel -DCUDA -DROCM
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

## Threaded detection

Use `registry_detect_threaded()` for non-blocking detection via `thread.cyr`:

```cyr
let registry = registry_detect_threaded();
```

This is important in applications where blocking the main thread would
stall other tasks.

---

## Thread count

`registry_detect()` uses `thread.cyr` to run backends in parallel. When 2+
backends are enabled, each gets its own scoped thread. Scoped threads are
lightweight and are joined before `registry_detect()` returns.

If you're already in a thread-constrained environment, use
`builder_none()` with specific `builder_with()` calls to limit concurrency,
or call detection from a dedicated thread.
