# Framework Integration Guide

`ai-hwaccel` detects hardware and plans model deployment. It does **not** run
inference or training — that's the job of your ML framework. This guide shows
how to bridge the gap. Most ML frameworks are driven from Python, so the
examples use the Python package (`pip install ai-hwaccel`). Any other language
can run the binary and parse its JSON ([docs/schema.json](../schema.json)).

## General pattern

```python
import ai_hwaccel

reg = ai_hwaccel.detect()                 # every accelerator, one profile per device
kinds = {p.accelerator for p in reg.profiles if p.available}
plan = ai_hwaccel.plan("70B", quant="bf16")

# Use kinds, plan.strategy and plan.shards to configure your framework.
```

`p.accelerator` is one of the names in the schema: `"CUDA GPU"`,
`"ROCm GPU"`, `"Metal GPU"`, `"Vulkan GPU"`, `"Windows GPU"`, `"TPU"`,
`"Intel NPU"`, and so on. `p.family` groups them: `"CPU"`, `"GPU"`, `"NPU"`,
`"TPU"`, `"AI ASIC"`.

---

## PyTorch

```python
import torch
import ai_hwaccel

kinds = {p.accelerator for p in ai_hwaccel.detect().profiles if p.available}

if "CUDA GPU" in kinds or "ROCm GPU" in kinds:
    device = torch.device("cuda", 0)   # ROCm builds of PyTorch use "cuda" too
elif "Metal GPU" in kinds:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

A CUDA profile's `device_id` is `nvidia-smi`'s index, which follows PCI bus
order. PyTorch numbers CUDA devices fastest-first unless you set
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, so set it before mapping one onto the other.

## JAX

JAX picks its platform when it initializes, from `JAX_PLATFORMS`:

```python
import os
import ai_hwaccel

kinds = {p.accelerator for p in ai_hwaccel.detect().profiles if p.available}
if "TPU" in kinds:
    os.environ["JAX_PLATFORMS"] = "tpu"
elif "CUDA GPU" in kinds:
    os.environ["JAX_PLATFORMS"] = "cuda"
elif "ROCm GPU" in kinds:
    os.environ["JAX_PLATFORMS"] = "rocm"
else:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax   # after setting JAX_PLATFORMS
```

## ONNX Runtime

Order the execution providers by what is present; ONNX Runtime uses the first
one its build supports:

```python
import onnxruntime as ort
import ai_hwaccel

kinds = {p.accelerator for p in ai_hwaccel.detect().profiles if p.available}
providers = []
if "CUDA GPU" in kinds:
    providers.append("CUDAExecutionProvider")
if "ROCm GPU" in kinds:
    providers.append("ROCMExecutionProvider")
if "Metal GPU" in kinds or "Apple Neural Engine" in kinds:
    providers.append("CoreMLExecutionProvider")
if "Windows GPU" in kinds:
    providers.append("DmlExecutionProvider")          # DirectML
if "Intel NPU" in kinds or "Intel oneAPI GPU" in kinds:
    providers.append("OpenVINOExecutionProvider")
providers.append("CPUExecutionProvider")

session = ort.InferenceSession("model.onnx", providers=providers)
```

---

## Multi-device sharding

For a model that doesn't fit on one device, use the sharding plan:

```python
import ai_hwaccel

plan = ai_hwaccel.plan("70B", quant="bf16")

if plan.strategy == "None":
    shard = plan.shards[0]            # the whole model on one device
elif plan.strategy == "Pipeline Parallel":
    for s in plan.shards:             # layers s.layer_start..s.layer_end on s.device
        print(s.id, s.device, s.device_id, s.layer_start, s.layer_end, s.memory_bytes)
elif plan.strategy == "Tensor Parallel":
    pass                              # split tensors across plan.strategy_count devices
```

`plan.est_tokens_per_sec` is an estimate, or `None` when there is none.

## Training memory budgeting

Before launching a fine-tuning job, check that it fits:

```python
import ai_hwaccel

need = ai_hwaccel.training_memory("7B", method="lora")
have = ai_hwaccel.summary()["accelerator_memory_bytes"]

if need.total_bytes > have:
    qlora = ai_hwaccel.training_memory("7B", method="qlora-4bit")
    print(f"LoRA needs {need.total_gib:.1f} GiB, "
          f"QLoRA 4-bit {qlora.total_gib:.1f} GiB, "
          f"{have / 2**30:.1f} GiB available")
```

The methods are `full`, `lora`, `qlora-4bit`, `qlora-8bit`, `prefix`, `dpo`,
`rlhf` and `distillation`.

---

## Other languages

Run the CLI and parse its JSON: `ai-hwaccel` (the registry), `--summary`,
`--plan 70B`, `--train 7B --method lora` and `--cost 70B --json`.
[docs/schema.json](../schema.json) describes each one.

## Cyrius

Cyrius programs use the library bundle directly (see the README's *Using as a
library*):

```cyrius
include "lib/ai-hwaccel.cyr"

var r = registry_detect();
var best = reg_best_available(r);                  # highest-ranked available profile
var q = reg_suggest_quant(r, 70000000000);         # a QUANT_* level for 70B parameters
var plan = reg_plan_sharding(r, 70000000000, q);
var json = registry_to_json(r);
```
