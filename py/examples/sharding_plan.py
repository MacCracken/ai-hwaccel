#!/usr/bin/env python3
"""Sharding plan example.

Shows how to generate and inspect model sharding plans for different
model sizes and quantization levels.
"""

import ai_hwaccel


def print_plan(name: str, plan: dict) -> None:
    """Pretty-print a sharding plan."""
    print(f"  Strategy:    {plan['strategy']}")
    print(f"  Memory:      {plan['total_memory_gb']:.1f} GB")
    tps = plan["estimated_tokens_per_sec"]
    if tps is not None:
        print(f"  Throughput:  {tps:.0f} tok/s (estimated)")
    print(f"  Shards:      {plan['num_shards']}")
    for shard in plan["shards"]:
        layers = f"layers {shard['layer_range'][0]}-{shard['layer_range'][1]}"
        print(f"    [{shard['shard_id']}] {shard['device']} — {layers} ({shard['memory_gb']:.1f} GB)")
    print()


def main():
    registry = ai_hwaccel.detect()
    print(f"Detected: {registry}")
    print()

    # Try different model sizes with suggested quantization
    models = [
        ("Llama 7B", 7_000_000_000),
        ("Llama 13B", 13_000_000_000),
        ("Llama 70B", 70_000_000_000),
        ("Mixtral 8x7B", 46_700_000_000),
    ]

    for name, params in models:
        quant = registry.suggest_quantization(params)
        print(f"--- {name} ({params / 1e9:.0f}B params) at {quant} ---")
        plan = registry.plan_sharding(params, quant)
        print_plan(name, plan)

    # Compare quantization levels for a 70B model
    print("=== 70B model across quantization levels ===")
    print()
    for quant in ["FP32", "FP16", "BF16", "INT8", "INT4"]:
        plan = registry.plan_sharding(70_000_000_000, quant)
        print(f"--- {quant} ---")
        print_plan(f"70B @ {quant}", plan)

    # Use the standalone function (auto-detects hardware)
    print("=== Standalone plan_sharding() ===")
    plan = ai_hwaccel.plan_sharding(7_000_000_000, "FP16")
    print(f"7B @ FP16: {plan['strategy']} — {plan['total_memory_gb']:.1f} GB")


if __name__ == "__main__":
    main()
