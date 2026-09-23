# Performance Tuning

How to keep detection fast, and where its time goes.

---

## Where the time goes

Sysfs and OS-API backends cost almost nothing; vendor tools dominate. On the
Linux dev host (Ryzen 7 5800H, Radeon iGPU), a whole CLI run takes about
22 ms, most of it in one `vulkaninfo` run. With no vendor tool on `PATH` it
takes under 2 ms. Each installed tool adds its own startup time, typically tens
of milliseconds, often more for `nvidia-smi` on a many-GPU host.

Vendor tools also run without a time limit, so a hung tool blocks detection
(bounding them is on the roadmap).

---

## Cache the registry

`registry_detect()` probes sysfs and runs the tools on every call. When you
detect more than once, cache it:

```cyrius
# Detect once, reuse for 5 minutes
var c = cached_registry_new(300);
var r = cached_get(c);    # first call: detects
var r2 = cached_get(c);   # later calls: the cached registry
```

`cached_invalidate(c)` forces a fresh detection (after hot-plug, for example).

---

## Run only the backends you need

```cyrius
# Only CUDA
var r = registry_detect_with(builder_with(builder_none(), BACKEND_CUDA));

# CPU only: no backend at all
var r2 = registry_detect_with(builder_none());
```

`registry_detect_no_exec()` runs every backend that needs no subprocess and
skips the rest. It is the fastest full-coverage option on hosts whose
accelerators are all found through sysfs (AMD, TPU, NPUs) or OS APIs (Apple,
Windows).

---

## Threaded detection

`registry_detect_threaded()` runs the tool-based backends (CUDA, Gaudi,
Neuron, Vulkan, oneAPI, Apple) in parallel threads and the sysfs ones on the
calling thread. It returns when every thread has joined, so it cuts latency
when several tools are installed. It is not asynchronous. `registry_detect()`
runs the backends one after another, on one thread.

---

## Binary size

`CYRIUS_DCE=1` removes unreachable code: the x86_64 CLI is about 236 KB with
it and 461 KB without. Every backend is always compiled in: the per-backend
`-D` flags described in ADR-004 were never implemented.

---

## Benchmarks

Two suites, `benches/parsing.bcyr` and `benches/registry.bcyr`, cover the
parsers (CUDA, Vulkan, Neuron, model formats), registry queries, the duplicate
pass and JSON output. `./scripts/bench-history.sh` runs both from the
repository root and appends the results to `bench-history.csv`. Every release's
CHANGELOG section carries its before/after table.
