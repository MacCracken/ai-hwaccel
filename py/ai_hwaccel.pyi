"""Type stubs for ai_hwaccel — Python bindings for ai-hwaccel.

Universal AI hardware accelerator detection, capability querying,
and workload planning.
"""

from typing import Optional

class Registry:
    """Registry of detected hardware accelerators.

    Use ``detect()`` to create a new registry, or
    ``Registry.from_json(json_str)`` to deserialize one.
    """

    def all_profiles(self) -> list[dict]:
        """Return all accelerator profiles (including unavailable ones).

        Each profile is a dict with keys:
            accelerator, accelerator_str, family, available,
            memory_bytes, memory_gb, compute_capability, driver_version,
            memory_bandwidth_gbps, memory_used_bytes, memory_free_bytes,
            pcie_bandwidth_gbps, numa_node, temperature_c, power_watts,
            gpu_utilization_percent.
        """
        ...

    def available(self) -> list[dict]:
        """Return only available accelerator profiles."""
        ...

    def best_available(self) -> Optional[dict]:
        """Return the highest-ranked available accelerator profile, or None."""
        ...

    def total_memory(self) -> int:
        """Total memory in bytes across all available devices."""
        ...

    def total_accelerator_memory(self) -> int:
        """Total non-CPU accelerator memory in bytes."""
        ...

    def has_accelerator(self) -> bool:
        """Whether any non-CPU accelerator is available."""
        ...

    def suggest_quantization(self, model_params: int) -> str:
        """Suggest a quantization level for the given model parameter count.

        Args:
            model_params: Total number of model parameters
                (e.g. 7_000_000_000 for a 7B model).

        Returns:
            One of "FP32", "FP16", "BF16", "INT8", "INT4".
        """
        ...

    def plan_sharding(self, model_params: int, quantization: str) -> dict:
        """Generate a sharding plan for a model.

        Args:
            model_params: Total number of model parameters.
            quantization: Quantization level string
                ("FP32", "FP16", "BF16", "INT8", "INT4").

        Returns:
            A dict with keys: strategy, strategy_type, total_memory_bytes,
            total_memory_gb, estimated_tokens_per_sec, num_shards, shards.
        """
        ...

    def system_io(self) -> dict:
        """Return system I/O information.

        Returns:
            A dict with keys: interconnects (list of dicts),
            storage (list of dicts), has_interconnect (bool),
            total_interconnect_bandwidth_gbps (float).
        """
        ...

    def to_json(self) -> str:
        """Serialize this registry to a JSON string."""
        ...

    @staticmethod
    def from_json(json: str) -> "Registry":
        """Deserialize a registry from a JSON string.

        Raises:
            ValueError: If the JSON is malformed.
        """
        ...

    def warnings(self) -> list[str]:
        """Return non-fatal detection warnings as a list of strings."""
        ...

    def schema_version(self) -> int:
        """Schema version of this registry."""
        ...

    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...

def detect() -> Registry:
    """Detect all available AI hardware accelerators on this system.

    Probes for CUDA GPUs, ROCm GPUs, Apple Metal/ANE, Intel NPUs,
    AMD XDNA NPUs, Google TPUs, Intel Gaudi, AWS Inferentia/Trainium,
    Qualcomm Cloud AI, Vulkan devices, and more.

    Returns:
        A Registry containing profiles for every detected device.
    """
    ...

def suggest_quantization(
    model_params: int,
    registry: Optional[Registry] = None,
) -> str:
    """Suggest a quantization level for the given model and hardware.

    Args:
        model_params: Total number of model parameters
            (e.g. 7_000_000_000 for a 7B model).
        registry: Optional Registry to use. If not provided,
            hardware is detected automatically.

    Returns:
        One of "FP32", "FP16", "BF16", "INT8", "INT4".
    """
    ...

def plan_sharding(
    model_params: int,
    quantization: str,
    registry: Optional[Registry] = None,
) -> dict:
    """Generate a sharding plan for a model.

    Args:
        model_params: Total number of model parameters.
        quantization: Quantization level string
            ("FP32", "FP16", "BF16", "INT8", "INT4").
        registry: Optional Registry to use. If not provided,
            hardware is detected automatically.

    Returns:
        A dict with keys: strategy, strategy_type, total_memory_bytes,
        total_memory_gb, estimated_tokens_per_sec, num_shards, shards.
        Each shard has: shard_id, layer_range, device, memory_bytes, memory_gb.
    """
    ...

def system_io(registry: Optional[Registry] = None) -> dict:
    """Return system I/O information (interconnects, storage).

    Args:
        registry: Optional Registry to use. If not provided,
            hardware is detected automatically.

    Returns:
        A dict with keys: interconnects, storage, has_interconnect,
        total_interconnect_bandwidth_gbps.
    """
    ...

def estimate_training_memory(
    model_params_millions: int,
    method: str,
    target: str,
) -> dict:
    """Estimate training/fine-tuning memory requirements.

    Args:
        model_params_millions: Model size in millions of parameters
            (e.g. 7000 for a 7B model).
        method: Training method — one of "full", "lora", "qlora",
            "qlora-4bit", "qlora-8bit", "prefix", "dpo", "rlhf",
            "distillation".
        target: Target device — one of "gpu", "tpu", "gaudi", "cpu".

    Returns:
        A dict with keys: model_gb, optimizer_gb, activation_gb, total_gb.
    """
    ...
