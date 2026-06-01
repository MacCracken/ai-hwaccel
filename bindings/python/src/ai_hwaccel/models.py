"""Typed model for the ai-hwaccel JSON contract (schema v4).

These dataclasses mirror the JSON emitted by the ``ai-hwaccel`` binary
(see ``src/json_out.cyr`` in the cyrius core). They are intentionally a
thin, lossless mapping: field names match the JSON keys exactly. Derived
conveniences (e.g. fixed-point ``*_x1000`` values surfaced as plain
floats) are exposed as ``@property`` so the raw integers stay available.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

# JSON schema version this model targets. detect() warns on a mismatch.
SCHEMA_VERSION = 4


def _select(cls: type, d: dict) -> dict:
    """Keep only keys that are fields of ``cls`` (ignore unknown keys)."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in names}


# --- Accelerators ----------------------------------------------------


@dataclass
class AcceleratorProfile:
    accelerator: str
    device_id: int
    available: bool
    memory_bytes: int
    family: str
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    device_name: Optional[str] = None
    memory_used_bytes: Optional[int] = None
    memory_free_bytes: Optional[int] = None
    numa_node: Optional[int] = None
    temperature_c: Optional[int] = None
    gpu_utilization_percent: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "AcceleratorProfile":
        return cls(**_select(cls, d))


# --- System I/O topology --------------------------------------------


@dataclass
class Interconnect:
    kind: str
    name: str
    bandwidth_bytes_per_sec: int
    state: int

    @classmethod
    def from_dict(cls, d: dict) -> "Interconnect":
        return cls(**_select(cls, d))


@dataclass
class StorageDevice:
    name: str
    kind: str
    bandwidth_bytes_per_sec: int

    @classmethod
    def from_dict(cls, d: dict) -> "StorageDevice":
        return cls(**_select(cls, d))


@dataclass
class RuntimeEnvironment:
    is_docker: bool = False
    is_k8s: bool = False
    k8s_namespace: Optional[str] = None
    cloud_provider: Optional[str] = None
    instance_type: Optional[str] = None
    region: Optional[str] = None
    k8s_gpu_count: Optional[int] = None
    k8s_gpu_source: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "RuntimeEnvironment":
        return cls(**_select(cls, d))


@dataclass
class SystemIo:
    interconnects: list = field(default_factory=list)
    storage: list = field(default_factory=list)
    environment: Optional[RuntimeEnvironment] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["SystemIo"]:
        if d is None:
            return None
        env = d.get("environment")
        return cls(
            interconnects=[Interconnect.from_dict(x) for x in d.get("interconnects", [])],
            storage=[StorageDevice.from_dict(x) for x in d.get("storage", [])],
            environment=RuntimeEnvironment.from_dict(env) if env else None,
        )


@dataclass
class Registry:
    schema_version: int
    profiles: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    system_io: Optional[SystemIo] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Registry":
        return cls(
            schema_version=d.get("schema_version", 0),
            profiles=[AcceleratorProfile.from_dict(x) for x in d.get("profiles", [])],
            warnings=list(d.get("warnings", [])),
            system_io=SystemIo.from_dict(d.get("system_io")),
        )

    @property
    def has_accelerator(self) -> bool:
        return any(p.family != "CPU" and p.available for p in self.profiles)

    def to_dataframe(self):
        """Profiles as a pandas DataFrame (requires the ``pandas`` extra)."""
        from ._pandas import profiles_to_dataframe

        return profiles_to_dataframe(self.profiles)


# --- Sharding plan ---------------------------------------------------


@dataclass
class ModelShard:
    id: int
    layer_start: int
    layer_end: int
    device: str
    device_id: int
    memory_bytes: int

    @classmethod
    def from_dict(cls, d: dict) -> "ModelShard":
        return cls(**_select(cls, d))


@dataclass
class ShardingPlan:
    strategy: str
    strategy_count: int
    total_memory_bytes: int
    est_tokens_per_sec_x1000: Optional[int] = None
    shards: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ShardingPlan":
        return cls(
            strategy=d.get("strategy", "None"),
            strategy_count=d.get("strategy_count", 0),
            total_memory_bytes=d.get("total_memory_bytes", 0),
            est_tokens_per_sec_x1000=d.get("est_tokens_per_sec_x1000"),
            shards=[ModelShard.from_dict(x) for x in d.get("shards", [])],
        )

    @property
    def est_tokens_per_sec(self) -> Optional[float]:
        if self.est_tokens_per_sec_x1000 is None:
            return None
        return self.est_tokens_per_sec_x1000 / 1000.0


# --- Training memory estimate ---------------------------------------


@dataclass
class TrainingMemory:
    model_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    total_bytes: int
    model_gib_x1000: int
    optimizer_gib_x1000: int
    activation_gib_x1000: int
    total_gib_x1000: int

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingMemory":
        return cls(**_select(cls, d))

    @property
    def total_gib(self) -> float:
        return self.total_gib_x1000 / 1000.0


# --- Cost recommendation --------------------------------------------


@dataclass
class CostRecommendation:
    instance: str
    provider: str
    gpu: str
    gpu_count: int
    total_memory_gb: int
    price_per_hour_usd_x100: int

    @classmethod
    def from_dict(cls, d: dict) -> "CostRecommendation":
        return cls(**_select(cls, d))

    @property
    def price_per_hour_usd(self) -> float:
        return self.price_per_hour_usd_x100 / 100.0


@dataclass
class CostReport:
    model: str
    quantization: str
    memory_required_bytes: int
    recommendations: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "CostReport":
        return cls(
            model=d.get("model", ""),
            quantization=d.get("quantization", ""),
            memory_required_bytes=d.get("memory_required_bytes", 0),
            recommendations=[
                CostRecommendation.from_dict(x) for x in d.get("recommendations", [])
            ],
        )
