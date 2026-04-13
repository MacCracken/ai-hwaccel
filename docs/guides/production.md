# Production Deployment Guide

Best practices for using `ai-hwaccel` in production systems.

## Installation

Build the binary from source:

```sh
cyrius build src/main.cyr build/ai-hwaccel
```

For minimal binary size (197 KB), enable only the backends you need:

```sh
cyrius build src/main.cyr build/ai-hwaccel -DCUDA
```

## Detection behaviour

### Best-effort, non-blocking

Detection never panics or returns errors that prevent operation. If a CLI tool
is missing, times out, or produces unparseable output, the backend is silently
skipped and a structured warning is recorded:

```cyr
let registry = registry_detect();

// Always at least CPU
assert(!registry.all_profiles().is_empty());

// Check for non-fatal issues
for w in registry.warnings() {
    log_warn("detection warning: {}", w);
}
```

### Timeouts

All CLI tool invocations have a **5-second timeout**. A hung `nvidia-smi`
(common when GPUs are in error state) will not block your application
indefinitely.

### PATH requirements

Detection tools must be on `$PATH` (or in standard locations like
`/opt/rocm/bin/`). In containers, ensure the relevant tools are installed:

```dockerfile
# Example: NVIDIA GPU container
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# nvidia-smi is included in the CUDA runtime image
```

## Caching

For applications that call `detect()` repeatedly (schedulers, monitoring
loops), use `CachedRegistry` to avoid redundant subprocess calls:

```cyr
// Detect once, cache for 60 seconds
let cache = CachedRegistry::new(60);

loop {
    let registry = cache.get(); // re-detects only after TTL expires
    // ... use registry ...
}
```

Call `cache.invalidate()` to force re-detection (e.g. after a GPU reset).

## Selective detection

If you know which hardware is present, skip unnecessary backends:

```cyr
// Only probe CUDA — skip ROCm, Vulkan, TPU, etc.
let registry = registry_detect_builder()
    .with_cuda()
    .detect();
```

Or disable at compile time with `-D` flags for faster builds and smaller
binaries.

## Security considerations

### Trusted PATH

Detection runs CLI tools by **absolute path** (resolved via `$PATH`). Ensure
your `$PATH` is controlled:

```sh
# Good: explicit PATH in systemd unit
Environment=PATH=/usr/local/bin:/usr/bin:/opt/rocm/bin

# Bad: inheriting an untrusted PATH
```

### Output validation

All parsed values from CLI tools are validated:
- Device IDs: 0--1024
- Memory sizes: 0--16 TiB
- Output capped at 1 MiB stdout, 4 KiB stderr

### Deserialization

If loading a registry from untrusted JSON (e.g. from a network API), the
manual JSON parser rejects unknown fields by default.
Apply your own size limits before passing to `json_parse()`.

See [docs/development/threat-model.md](../development/threat-model.md) for the
full threat model.

## Logging

The binary uses structured logging. Log level is controlled via environment
variable or CLI flag:

```sh
# Production: warn-level only
AI_HWACCEL_LOG=warn build/ai-hwaccel --table

# Debugging: full detection trace
AI_HWACCEL_LOG=debug build/ai-hwaccel --table
```

The CLI binary supports `AI_HWACCEL_LOG` and `--debug` flags.

## Monitoring with the CLI

```sh
# One-shot table
ai-hwaccel --table

# JSON for Prometheus/Grafana ingestion
ai-hwaccel --summary | jq .

# Live monitoring (re-detect every 30s)
ai-hwaccel --watch 30

# Filter to GPUs only, sorted by memory
ai-hwaccel --table --family gpu --sort mem
```

## Container / Kubernetes deployment

### Docker

```dockerfile
FROM cyrius:3.9.0-slim AS builder
WORKDIR /app
COPY . .
RUN cyrius build src/main.cyr build/ai-hwaccel

FROM debian:bookworm-slim
COPY --from=builder /app/build/ai-hwaccel /usr/local/bin/
# Ensure nvidia-smi or other tools are available via the runtime
CMD ["ai-hwaccel", "--table"]
```

### Kubernetes device plugin

Use `ai-hwaccel --summary` as a readiness probe or init container to verify
GPU availability:

```yaml
initContainers:
  - name: gpu-check
    image: your-image
    command: ["ai-hwaccel", "--summary"]
```

### Resource requirements

The detection itself uses:
- **CPU**: negligible (< 10ms for sysfs probing)
- **Memory**: < 1 MB resident
- **I/O**: reads sysfs files + runs CLI tools (5s timeout each)
- **Threads**: one per enabled backend during detection (short-lived)

## C FFI

For non-Cyrius applications, use the C API:

```c
#include "ai_hwaccel.h"

HwAccelRegistry *reg = ai_hwaccel_detect();
printf("Devices: %u\n", ai_hwaccel_device_count(reg));
printf("Has accelerator: %s\n", ai_hwaccel_has_accelerator(reg) ? "yes" : "no");

char *json = ai_hwaccel_json(reg);
printf("%s\n", json);

ai_hwaccel_free_string(json);
ai_hwaccel_free(reg);
```

Build with:

```sh
cyrius build src/ffi.cyr build/libai_hwaccel.so --shared
cc -o myapp myapp.c -L build -lai_hwaccel
```

## Version compatibility

The `schema_version` field in serialized output (currently `1`) allows you to
detect breaking format changes:

```cyr
let reg = registry_from_json(json);
assert(reg.schema_version() == SCHEMA_VERSION);
```
