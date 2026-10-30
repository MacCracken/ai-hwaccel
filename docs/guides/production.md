# Production Deployment Guide

Best practices for using `ai-hwaccel` in production systems.

## Installation

Build the binary from source with the toolchain `cyrius.cyml` pins:

```sh
cyrius lib sync && cyrius deps
CYRIUS_DCE=1 cyrius build src/main.cyr build/ai-hwaccel   # ≈289 KB, x86_64
```

`CYRIUS_DCE=1` drops unreachable code and gives the smallest binary. Every
backend is compiled in; which ones run is chosen at run time (see *Selective
detection*).

Python users can install the Python package from a wheel instead: the wheels
bundle the binary for Linux, macOS (arm64) and Windows. They come from the
Wheels workflow's artifacts, not from PyPI, where the project does not own the
name `ai-hwaccel`; never install it by name from an index (see
[bindings/python](../../bindings/python/README.md#install)).

`--version` prints the version built into the binary and reads no file
(2.5.5). `--cost` reads `data/cloud_pricing.json`: when the binary runs
outside the repository, point it at the directory holding `data/` with
`--data-dir <path>` or `AI_HWACCEL_DATA_DIR`. A long-lived library consumer
that replaces the file calls `cloud_pricing_reload()`.

## Detection behaviour

### Best-effort

Detection does not fail. If a tool is missing or produces unparseable output,
that backend reports nothing and a warning names the tool; the CPU profile is
always present:

```cyrius
var r = registry_detect();

# Always at least the CPU
assert(vec_len(reg_profiles(r)) >= 1);

# Tools that were missing or failed
var i = 0;
while (i < reg_num_warnings(r)) {
    # vec_get(reg_warnings(r), i): warning_tool(w) is the tool's name
    i = i + 1;
}
```

In the JSON, the same warnings are the `warnings` array of tool names.

### Tool deadline

A vendor tool that prints nothing for 10 s is cut and listed in `warnings` as
a timeout (2.5.1; `CMD_DEFAULT_TIMEOUT_MS`), so a hung `nvidia-smi` (common
when a GPU is in an error state) costs about 20 s on Linux and macOS: it is cut after 10 s of silence, gets a second
10 s to exit, then TERM, up to 5 s of grace and KILL (25 s if it ignores
TERM), not the whole detection.
Tools run one after another, so each hung tool adds that, and the NVLink probe
is skipped after `nvidia-smi` hit its deadline (2.5.5). A tool stuck in the
kernel (D state) is not bounded. On Windows the bound is 10 s of wall-clock.
The deadline is idle-based: a tool still printing is not cut, and its output
is capped at 1 MiB. AGNOS has no deadline. Size an outer timeout at about
2 x `CMD_DEFAULT_TIMEOUT_MS` + 5 s per tool that may hang (the Python
package's default is 60 s). Where that is too long, use
`registry_detect_no_exec()`. It spawns nothing; it misses CUDA and oneAPI
devices, and sizes Gaudi, Neuron, Cerebras and Graphcore from their device
nodes.

### PATH requirements

Vendor tools are found on `$PATH` and run by absolute path. In containers,
make sure the tools you need are installed:

```dockerfile
# Example: NVIDIA GPU container
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# nvidia-smi is included in the CUDA runtime image
```

On Windows the tools are found through `PATH` like anywhere else (2.5.3), and
DXGI lists every GPU without one. `vulkaninfo` (Windows' Vulkan Runtime puts it
in System32) is what tells an integrated GPU from a discrete one there. It
takes about 0.9 s per run on a low-end Celeron, where 2.5.2's detection took
0.14 s, so cache the registry on such hosts, or detect with
`registry_detect_no_exec()` when the carve-out figure is enough.

## Caching

For applications that detect repeatedly (schedulers, monitoring loops), cache
the registry to avoid rerunning the tools:

```cyrius
# Detect once, reuse for 60 seconds
var c = cached_registry_new(60);

while (1 == 1) {
    var r = cached_get(c);   # re-detects only after the TTL expires
    # ... use r ...
}
```

`cached_invalidate(c)` forces the next `cached_get` to re-detect (after a
GPU reset, for example). `disk_cached_new(ttl)` / `disk_cached_get(c)` are
deprecated (2.5.1): they wrote a file they never read back, and now cache in
memory exactly as `cached_*` does. They are removed in 2.6.0.

## Selective detection

If you know which hardware is present, run only those backends:

```cyrius
# Only CUDA
var r = registry_detect_with(builder_with(builder_none(), BACKEND_CUDA));

# Everything but Vulkan
var r2 = registry_detect_with(builder_without(builder_all(), BACKEND_VULKAN));
```

`registry_detect_with` lets the backends run tools when the mask holds a
tool-based backend. A mask without one spawns nothing (2.5.1): Vulkan uses its
sysfs scan, and the `system_profiler`, `wmic` and NVLink runs are skipped.
`registry_detect_with_opts(mask, 0)` forbids tools for any mask, and
`registry_detect_with_opts(mask, 1)` allows them for any mask.

## Security considerations

### Trusted PATH

Vendor tools are resolved through `$PATH`, so control it:

```sh
# Good: explicit PATH in a systemd unit
Environment=PATH=/usr/local/bin:/usr/bin

# Bad: inheriting an untrusted PATH
```

On Linux each tool runs with an empty environment, so `LD_PRELOAD` and similar
variables are not passed to it.

### Output validation

Values parsed from tool output are range-checked:
- Device IDs: 0 to 1024
- Memory sizes: 0 to 16 TiB
- Tool stdout is capped at 1 MiB. stderr is discarded on Linux and macOS; on
  Windows it shares the stdout pipe and counts toward the 1 MiB cap

### Deserialization

`profile_from_json_str` parses one profile with bayan-json. It reads the keys
it knows, ignores the rest, and falls back to defaults for missing ones. Apply
your own size limits to untrusted input before parsing it.

See [docs/development/threat-model.md](../development/threat-model.md) for the
full threat model.

## Logging

Logs go to stderr (stdout carries only the output). The default level is
`warn`. Set it with `AI_HWACCEL_LOG` or on the command line:

```sh
# Production: warnings only (the default)
AI_HWACCEL_LOG=warn build/ai-hwaccel --table

# Debugging: the detection trace, including a line per dropped duplicate
build/ai-hwaccel --table --log-level debug
```

The flags are `--log-level <level>`, `-v` (debug), `-vv` (trace) and `-q`
(silent).

## Monitoring with the CLI

```sh
# One-shot table
ai-hwaccel --table

# Compact JSON for Prometheus/Grafana ingestion
ai-hwaccel --summary | jq .

# Re-detect every 30 s
watch -n 30 ai-hwaccel --table
```

## Container / Kubernetes deployment

### Docker

Build the binary with the pinned toolchain (on the host, or in a builder stage
that has it), then copy it and its data files into the runtime image:

```dockerfile
FROM debian:bookworm-slim
COPY build/ai-hwaccel /usr/local/bin/
COPY VERSION /usr/share/ai-hwaccel/
COPY data/ /usr/share/ai-hwaccel/data/
ENV AI_HWACCEL_DATA_DIR=/usr/share/ai-hwaccel
# Install the vendor tools you need, or use a vendor base image
CMD ["ai-hwaccel", "--table"]
```

### Kubernetes

The CLI exits 0 whether or not it finds an accelerator (detection does not
fail), so an init container that only runs `ai-hwaccel --summary` always
succeeds. Make the check read the JSON: here the init container fails, and the
workload does not start, when no GPU is found. The image needs `sh` and `jq`:

```yaml
initContainers:
  - name: gpu-check
    image: your-image
    command: ["sh", "-c", "ai-hwaccel --summary | tee /dev/stderr | jq -e '.gpu_count > 0' > /dev/null"]
```

Use `.has_accelerator` instead of `.gpu_count > 0` when any accelerator family
will do.

### Resource requirements

- **Time:** on the Linux dev host, a whole CLI run takes about 22 ms, most of
  it in `vulkaninfo`. With no vendor tool on `PATH` it takes under 2 ms.
- **Threads:** `registry_detect()` uses one thread. `registry_detect_threaded()`
  (the CLI's `--threaded`) briefly starts one each for the backends that can
  run a vendor tool: CUDA, oneAPI, Vulkan, Apple, Gaudi, Neuron, Cerebras and
  Graphcore. The sysfs backends run on the calling thread. The tools
  themselves run one at a time (2.5.5, `cmd_capture`, until the upstream
  cyrius capture issue is fixed), so the threads overlap sysfs reads, DXGI
  calls and parsing, not the tool runs.
- **I/O:** sysfs reads, plus one run of each vendor tool that is installed.

## Other languages

There is no C API. Run the binary and parse its JSON, which
[docs/schema.json](../schema.json) describes. The Python package
([bindings/python](../../bindings/python/README.md)) does exactly that.

## Version compatibility

The `schema_version` field (6 since 2.3.28) is bumped whenever the JSON
format gains or changes keys, so a consumer can detect format changes:

```cyrius
var r = registry_detect();
assert(reg_schema(r) == SCHEMA_VERSION);
```
