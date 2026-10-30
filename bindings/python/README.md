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

The package is **not published on PyPI**. Install a wheel file, never the name
`ai-hwaccel` from an index: the project does not own that name there, so
whoever registers it would run code on your machine. The Wheels workflow
(`.github/workflows/wheels.yml`) builds the wheels on every tag and keeps them as
that run's artifacts (`wheels-linux`, `wheels-macos`, `wheels-windows`; GitHub
keeps artifacts for 90 days by default):

```bash
pip install ./ai_hwaccel-<version>-py3-none-<platform>.whl
pip install './ai_hwaccel-<version>-py3-none-<platform>.whl[pandas]'   # + DataFrame export
```

Or build one from a checkout with the pinned cyrius:

```bash
bindings/python/scripts/stage_binary.sh
bindings/python/scripts/build_wheel.sh manylinux2014_x86_64   # the platform tag
```

The Wheels workflow builds wheels for Linux (manylinux x86_64 / aarch64), macOS
(arm64), and Windows (x86_64). Each bundles a self-contained binary, so there
is nothing else to install. Where no wheel matches your platform,
`pip install ./bindings/python` installs the package without a binary, and it
runs one you supply. The package needs Python 3.9 or later (its build backend,
setuptools 77+, does not run on 3.8). It locates the binary via, in order, an explicit
`binary=` argument, the `AI_HWACCEL_BIN` environment variable, a wheel-bundled
binary, or `ai-hwaccel` on `PATH`. A `binary=` or `AI_HWACCEL_BIN` you give is
the only one tried: when it is not an executable file the call raises
`BinaryNotFoundError` instead of running another binary (2.5.5).

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
df = reg.to_dataframe()          # requires pandas
```

## API

| Function | Returns | Notes |
| --- | --- | --- |
| `detect()` | `Registry` | accelerators + `system_io` topology |
| `summary()` | `dict` | counts + totals |
| `plan(model, quant=)` | `ShardingPlan` | sharding recommendation |
| `training_memory(model, method=)` | `TrainingMemory` | bytes + `*_gib_x1000`, for a GPU |
| `cost(model, quant=)` | `CostReport` | cloud instance recommendations |
| `version()` | `str` | binary's self-reported version |

All accept `binary=<path>` and `timeout=<seconds>` (60 by default; 30 through 2.5.4). The
binary kills a vendor tool that prints nothing for 10 s; `timeout` bounds the
whole run. `TrainingMemory`, `ShardingPlan` and
`CostRecommendation` also expose the fixed-point fields as floats (`total_gib`,
`est_tokens_per_sec`, `price_per_hour_usd`).

`training_memory()` assumes bf16 weights unless `method` is `qlora-4bit` or
`qlora-8bit`, and takes a model of 1M parameters or more (the estimate works
in whole millions). Its `quant=` keyword never had an effect (`--train` does not read
`--quant`); since 2.5.5 passing it issues a `DeprecationWarning`, and 2.6.0
removes it.

`summary()`'s `total_memory_bytes` and `accelerator_memory_bytes` count system
RAM once. On Apple Silicon the Metal GPU and Neural Engine use the CPU's RAM,
so a 48 GB Mac totals 48 GiB. Each profile's `shared_memory_bytes` (schema v6)
says how much of its `memory_bytes` is system RAM, and
`AcceleratorProfile.dedicated_memory_bytes` is the rest.

## Errors

- `BinaryNotFoundError`: no binary was found, or the one `binary=` or
  `AI_HWACCEL_BIN` names is not an executable file.
- `CommandError`: the binary exited non-zero, ran past `timeout`, or printed
  output that is not valid JSON. The last two raise the subclasses
  `CommandTimeoutError` (also a `subprocess.TimeoutExpired`) and
  `InvalidOutputError` (also a `json.JSONDecodeError`, with the output in
  `.doc`), so code that caught those before 2.5.5 still catches them. A
  `CommandTimeoutError` keeps what the binary printed before it was killed in
  `.stdout` and `.stderr` (as `subprocess` gives them: bytes on POSIX, str on Windows); the stderr log
  names the detector that was running.

The binary's output is decoded as UTF-8 whatever the locale's code page (a
Windows host's is cp1252, cp936, ...); a byte that is not UTF-8 becomes U+FFFD.

## Data files & working directory

`cost()` reads `data/cloud_pricing.json`. The binary looks for it in the
directory given by `--data-dir`, else in **`AI_HWACCEL_DATA_DIR`**, else
relative to the working directory:

- **Bundled wheel binary**: unless you set `AI_HWACCEL_DATA_DIR` yourself,
  the wrapper points the binary at the bundled directory (with `--data-dir`),
  so `cost()` works from any working directory.
- **A binary you supply** (via `binary=`, `AI_HWACCEL_BIN`, or `PATH`):
  set `AI_HWACCEL_DATA_DIR` to a directory containing
  `data/cloud_pricing.json`, or run from a directory that has it.
  Detection (`detect`, `summary`, `plan`, `training_memory`) never needs
  this — only `cost()` does.

`version()` is the version built into the binary (since 2.5.5) and reads no
file; through 2.5.4 it read `VERSION` the same way and said `unknown` without
it.

The package's own version is always available as `ai_hwaccel.__version__`.

## License

GPL-3.0-only (same as the core project).
