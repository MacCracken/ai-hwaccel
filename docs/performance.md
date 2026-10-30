# Performance Tuning

How to keep detection fast, and where its time goes.

---

## Where the time goes

Sysfs and OS-API backends cost almost nothing; vendor tools dominate. On the
Linux dev host (Ryzen 7 5800H, Radeon iGPU), a whole CLI run takes about
22 ms, most of it in one `vulkaninfo` run. With no vendor tool on `PATH` it
takes under 2 ms. Each installed tool adds its own startup time, typically tens
of milliseconds, often more for `nvidia-smi` on a many-GPU host.

On Windows, `vulkaninfo` runs since 2.5.3 and costs far more: about 0.9 s per
run on `cass` (a Celeron J4125), where the whole 2.5.2 run took 0.14 s. It
runs once: Windows sizes an integrated GPU from DXGI rather than from a second,
full `vulkaninfo` run.

A tool that prints nothing for 10 s is cut and reported as a timeout (2.5.1),
so a hung tool delays detection by about 20 s on Linux and macOS: it is cut after 10 s of silence, gets a second
10 s to exit, then TERM, up to 5 s of grace and KILL (25 s if it ignores
TERM), on Windows by 10 s, but not
for ever.

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

## Lazy detection

`lazy_by_family(lr, family)` runs only that family's backends, on its first
query, and `lazy_into_registry(lr)` runs the system passes once (2.5.2). A TPU
or NPU query spawns nothing. On the dev host a TPU query takes about 5 µs and
an NPU query about 116 µs (`benches/lazy.bcyr`); through 2.5.1 they took 63
and 175 µs, because each also ran the storage and environment passes and
re-read system memory.

---

## Threaded detection

`registry_detect_threaded()` runs every backend that can start a vendor tool
(`backend_may_exec`: CUDA, oneAPI, Vulkan, Apple, Gaudi, Neuron, Cerebras,
Graphcore) on a thread of its own and the sysfs ones on the calling thread. It
returns when every thread has joined. It is not asynchronous. Since 2.5.5 the
tools themselves run one at a time (`cmd_capture`): the stdlib's capture
state is process-wide (issue 2026-10-29-cyrius-process-capture-state-not-
thread-safe), so what overlaps is the parsing, sysfs reads and OS calls
around the tools, and where the tools dominate the latency is about that of
`registry_detect()`, which runs the backends one after another on one
thread. The CLI takes `--threaded` to detect this way (2.5.4).

---

## Binary size

`CYRIUS_DCE=1` removes unreachable code: the x86_64 CLI is about 289 KB with
it and 608 KB without (2.5.5; 216 KB and 536 KB in 2.5.4). Every backend is
always compiled in: the per-backend `-D` flags described in ADR-004 were never
implemented.

---

## Benchmarks

Three suites: `benches/parsing.bcyr` and `benches/registry.bcyr` cover the
parsers (CUDA, Vulkan, Neuron, model formats), registry queries, the duplicate
pass and JSON output; `benches/lazy.bcyr` times lazy TPU and NPU queries on
the host it runs on (real sysfs, no tool). `./scripts/bench-history.sh` runs
all three from the repository root and appends the results to
`bench-history.csv`. Every release's
CHANGELOG section carries its before/after table.
