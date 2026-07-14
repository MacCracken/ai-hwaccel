"""ai-hwaccel — universal AI hardware accelerator detection (Python bindings).

Thin, dependency-free wrapper over the compiled ``ai-hwaccel`` binary.
Each call shells out to the binary and parses its JSON (schema v4) into
typed dataclasses.

    >>> import ai_hwaccel
    >>> reg = ai_hwaccel.detect()
    >>> [p.accelerator for p in reg.profiles]
    ['CPU', 'CUDA GPU', ...]
    >>> ai_hwaccel.plan("70B").strategy
    'Tensor Parallel'
"""

from __future__ import annotations

import warnings
from typing import Optional

from . import _runner
from ._runner import BinaryNotFoundError, CommandError, find_binary
from .models import (
    SCHEMA_VERSION,
    AcceleratorProfile,
    CostRecommendation,
    CostReport,
    Interconnect,
    ModelShard,
    Registry,
    RuntimeEnvironment,
    ShardingPlan,
    StorageDevice,
    SystemIo,
    TrainingMemory,
)

def _resolve_version() -> str:
    """Package version, derived — never hardcoded, so it can't drift.

    Installed (the normal case, incl. every published wheel): read the
    distribution metadata, which setuptools stamped from pyproject's
    ``version`` — itself propagated from the repo-root ``VERSION`` by
    scripts/version-bump.sh. Running uninstalled from the source tree
    (no metadata): fall back to that same ``VERSION`` file, four levels
    up from this module (src/ai_hwaccel/__init__.py). Neither path
    carries a version literal, so this stays in lockstep with VERSION.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        return _dist_version("ai-hwaccel")
    except PackageNotFoundError:
        from pathlib import Path

        version_file = Path(__file__).resolve().parents[4] / "VERSION"
        try:
            return version_file.read_text().strip()
        except OSError:
            return "0.0.0+source"


__version__ = _resolve_version()

__all__ = [
    "detect",
    "summary",
    "plan",
    "training_memory",
    "cost",
    "version",
    "find_binary",
    "BinaryNotFoundError",
    "CommandError",
    "SCHEMA_VERSION",
    "AcceleratorProfile",
    "Interconnect",
    "StorageDevice",
    "RuntimeEnvironment",
    "SystemIo",
    "Registry",
    "ModelShard",
    "ShardingPlan",
    "TrainingMemory",
    "CostRecommendation",
    "CostReport",
]


def detect(*, binary: Optional[str] = None, timeout: float = 30.0) -> Registry:
    """Detect all accelerators + system I/O topology."""
    data = _runner.run_json([], binary=binary, timeout=timeout)
    reg = Registry.from_dict(data)
    if reg.schema_version != SCHEMA_VERSION:
        warnings.warn(
            f"ai-hwaccel binary reports JSON schema v{reg.schema_version}, "
            f"these bindings target v{SCHEMA_VERSION}; some fields may be "
            f"missing or unexpected.",
            stacklevel=2,
        )
    return reg


def summary(*, binary: Optional[str] = None, timeout: float = 30.0) -> dict:
    """Compact registry summary (counts + totals) as a plain dict."""
    return _runner.run_json(["--summary"], binary=binary, timeout=timeout)


def plan(
    model: str,
    *,
    quant: str = "bf16",
    binary: Optional[str] = None,
    timeout: float = 30.0,
) -> ShardingPlan:
    """Recommend a sharding plan for ``model`` (e.g. ``"70B"``)."""
    data = _runner.run_json(
        ["--plan", model, "--quant", quant], binary=binary, timeout=timeout
    )
    return ShardingPlan.from_dict(data)


def training_memory(
    model: str,
    *,
    method: str = "full",
    quant: str = "bf16",
    binary: Optional[str] = None,
    timeout: float = 30.0,
) -> TrainingMemory:
    """Estimate training memory for ``model`` under ``method``.

    ``method`` is one of: full, lora, qlora-4bit, qlora-8bit, prefix,
    dpo, rlhf, distillation.
    """
    data = _runner.run_json(
        ["--train", model, "--method", method, "--quant", quant],
        binary=binary,
        timeout=timeout,
    )
    return TrainingMemory.from_dict(data)


def cost(
    model: str,
    *,
    quant: str = "bf16",
    binary: Optional[str] = None,
    timeout: float = 30.0,
) -> CostReport:
    """Cloud instance cost recommendations for ``model``."""
    data = _runner.run_json(
        ["--cost", model, "--json", "--quant", quant], binary=binary, timeout=timeout
    )
    return CostReport.from_dict(data)


def version(*, binary: Optional[str] = None, timeout: float = 30.0) -> str:
    """Return the version reported by the binary."""
    return _runner.run_text(["--version"], binary=binary, timeout=timeout).strip()
