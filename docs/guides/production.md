# Production Deployment Guide

Best practices for using `ai-hwaccel` in production systems.

## Installation

Build the binary from source with the toolchain `cyrius.cyml` pins:

```sh
cyrius lib sync && cyrius deps
CYRIUS_DCE=1 cyrius build src/main.cyr build/ai-hwaccel   # ≈236 KB, x86_64
```

`CYRIUS_DCE=1` drops unreachable code and gives the smallest binary. Every
backend is compiled in; which ones run is chosen at run time (see *Selective
detection*).

Python users can `pip install ai-hwaccel` instead: the wheels bundle the
binary for Linux, macOS (arm64) and Windows.

`--version` reads the `VERSION` file and `--cost` reads
`data/cloud_pricing.json`. When the binary runs outside the repository, point
it at a directory holding both with `--data-dir <path>` or
`AI_HWACCEL_DATA_DIR`.

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

### No timeout on vendor tools

Vendor tools run without a time limit: a hung `nvidia-smi` (common when a GPU
is in an error state) blocks detection until it exits. Bounding every run is
on the roadmap. Until then, use `registry_detect_no_exec()` where a hang is
unacceptable. It spawns nothing, so it cannot hang on a tool, but it misses the
backends that need one (CUDA, Gaudi, Neuron, oneAPI, Cerebras, Graphcore).

### PATH requirements

Vendor tools are found on `$PATH` and run by absolute path. In containers,
make sure the tools you need are installed:

```dockerfile
# Example: NVIDIA GPU container
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# nvidia-smi is included in the CUDA runtime image
```

On Windows no vendor tool runs yet (the PATH lookup does not understand
Windows paths; see the roadmap). GPUs are still found, through DXGI.

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
GPU reset, for example). `disk_cached_new(ttl)` / `disk_cached_get(c)` also
write each fresh registry's JSON to `~/.cache/ai-hwaccel/registry.json`, but
they do not read it back yet (see the roadmap), so a new process detects
again.

## Selective detection

If you know which hardware is present, run only those backends:

```cyrius
# Only CUDA
var r = registry_detect_with(builder_with(builder_none(), BACKEND_CUDA));

# Everything but Vulkan
var r2 = registry_detect_with(builder_without(builder_all(), BACKEND_VULKAN));
```

`registry_detect_with` allows the backends to run tools. For a mask that must
stay spawn-free, call `registry_detect_with_opts(mask, 0)`.

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
- Tool stdout is capped at 1 MiB; stderr is discarded

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

Use `ai-hwaccel --summary` in an init container to check GPU availability
before the workload starts:

```yaml
initContainers:
  - name: gpu-check
    image: your-image
    command: ["ai-hwaccel", "--summary"]
```

### Resource requirements

- **Time:** on the Linux dev host, a whole CLI run takes about 22 ms, most of
  it in `vulkaninfo`. With no vendor tool on `PATH` it takes under 2 ms.
- **Threads:** `registry_detect()` uses one thread. `registry_detect_threaded()`
  starts one per tool-based backend, briefly.
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
