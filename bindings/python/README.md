# ai-hwaccel (Python bindings)

Python bindings for [ai-hwaccel](https://github.com/MacCracken/ai-hwaccel)
— universal AI hardware accelerator detection (18 families, quantization,
sharding, training-memory estimation).

These bindings are a thin, **dependency-free** wrapper over the compiled
`ai-hwaccel` binary. There is no FFI (the cyrius toolchain emits
executables only); each call shells out to the binary and parses its
JSON (schema v4) into typed dataclasses.

## Install

```bash
pip install ai-hwaccel              # Linux x86_64 / aarch64 wheels (2.3.4)
pip install ai-hwaccel[pandas]      # + DataFrame export
```

Linux wheels (manylinux x86_64 / aarch64) bundle a self-contained binary.
**macOS and Windows wheels are not published yet** — they're gated on
cyrius toolchain support (the toolchain is currently Linux-only); see the
roadmap. On those platforms, supply your own binary: the package locates
it via, in order, an explicit `binary=` argument, the `AI_HWACCEL_BIN`
environment variable, a wheel-bundled binary, or `ai-hwaccel` on `PATH`.

## Usage

```python
import ai_hwaccel

reg = ai_hwaccel.detect()
for p in reg.profiles:
    print(p.accelerator, p.device_name, p.memory_bytes)

print(reg.has_accelerator)
for ic in reg.system_io.interconnects:
    print(ic.kind, ic.bandwidth_bytes_per_sec)

# Planning / estimation (parameterized by model size)
plan = ai_hwaccel.plan("70B", quant="bf16")
print(plan.strategy, plan.est_tokens_per_sec)

mem = ai_hwaccel.training_memory("70B", method="lora")
print(mem.total_gib)

rep = ai_hwaccel.cost("70B")
for r in rep.recommendations:
    print(r.instance, r.provider, r.price_per_hour_usd)

# Optional pandas export
df = reg.to_dataframe()          # requires ai-hwaccel[pandas]
```

## API

| Function | Returns | Notes |
| --- | --- | --- |
| `detect()` | `Registry` | accelerators + `system_io` topology |
| `summary()` | `dict` | counts + totals |
| `plan(model, quant=)` | `ShardingPlan` | sharding recommendation |
| `training_memory(model, method=, quant=)` | `TrainingMemory` | bytes + `*_gib_x1000` |
| `cost(model, quant=)` | `CostReport` | cloud instance recommendations |
| `version()` | `str` | binary's self-reported version |

All accept `binary=<path>` and `timeout=<seconds>`.

## Data files & working directory

`--version` reads `VERSION` and `cost()` reads `data/cloud_pricing.json`.
The binary honors the **`AI_HWACCEL_DATA_DIR`** environment variable to
locate them (falling back to cwd-relative if unset):

- **Bundled wheel binary**: the wrapper sets `AI_HWACCEL_DATA_DIR`
  automatically to the bundled directory, so `version()` and `cost()`
  work from any working directory (2.3.3+).
- **A binary you supply** (via `binary=`, `AI_HWACCEL_BIN`, or `PATH`):
  set `AI_HWACCEL_DATA_DIR` to a directory containing `VERSION` and
  `data/cloud_pricing.json`, or run from a directory that has them.
  Detection (`detect`, `summary`, `plan`, `training_memory`) never needs
  this — only `version()` and `cost()` do.

The package's own version is always available as `ai_hwaccel.__version__`.

## License

GPL-3.0-only (same as the core project).
