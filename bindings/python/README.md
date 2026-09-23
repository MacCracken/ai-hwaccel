# ai-hwaccel (Python bindings)

Python bindings for [ai-hwaccel](https://github.com/MacCracken/ai-hwaccel)
— universal AI hardware accelerator detection (18 families, quantization,
sharding, training-memory estimation).

These bindings are a thin, **dependency-free** wrapper over the compiled
`ai-hwaccel` binary. There is no FFI: each call runs the binary as a
subprocess and parses its JSON (schema v6,
[docs/schema.json](https://github.com/MacCracken/ai-hwaccel/blob/main/docs/schema.json))
into typed dataclasses.

## Install

```bash
pip install ai-hwaccel              # platform wheel (Linux / macOS / Windows)
pip install ai-hwaccel[pandas]      # + DataFrame export
```

Prebuilt wheels ship for Linux (manylinux x86_64 / aarch64), macOS
(arm64), and Windows (x86_64) — each bundles a self-contained binary, so
there is nothing else to install. If no wheel matches your platform, the
package falls back to a binary you supply: it locates one via, in order,
an explicit `binary=` argument, the `AI_HWACCEL_BIN` environment
variable, a wheel-bundled binary, or `ai-hwaccel` on `PATH`.

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

All accept `binary=<path>` and `timeout=<seconds>` (30 by default). The
binary puts no time limit on the vendor tools it runs, so `timeout` is what
bounds a hung `nvidia-smi`. `TrainingMemory`, `ShardingPlan` and
`CostRecommendation` also expose the fixed-point fields as floats (`total_gib`,
`est_tokens_per_sec`, `price_per_hour_usd`).

`summary()`'s `total_memory_bytes` and `accelerator_memory_bytes` count system
RAM once. On Apple Silicon the Metal GPU and Neural Engine use the CPU's RAM,
so a 48 GB Mac totals 48 GiB. Each profile's `shared_memory_bytes` (schema v6)
says how much of its `memory_bytes` is system RAM, and
`AcceleratorProfile.dedicated_memory_bytes` is the rest.

## Data files & working directory

`version()` reads `VERSION` and `cost()` reads `data/cloud_pricing.json`.
The binary looks for them in the directory given by `--data-dir`, else in
**`AI_HWACCEL_DATA_DIR`**, else relative to the working directory:

- **Bundled wheel binary**: unless you set `AI_HWACCEL_DATA_DIR` yourself,
  the wrapper points the binary at the bundled directory (with `--data-dir`),
  so `version()` and `cost()` work from any working directory.
- **A binary you supply** (via `binary=`, `AI_HWACCEL_BIN`, or `PATH`):
  set `AI_HWACCEL_DATA_DIR` to a directory containing `VERSION` and
  `data/cloud_pricing.json`, or run from a directory that has them.
  Detection (`detect`, `summary`, `plan`, `training_memory`) never needs
  this — only `version()` and `cost()` do.

The package's own version is always available as `ai_hwaccel.__version__`.

## License

GPL-3.0-only (same as the core project).
