#!/usr/bin/env python3
"""Basic hardware detection example.

Detects all AI hardware accelerators on the system and prints a summary.
"""

import ai_hwaccel


def main():
    # Detect all available hardware
    registry = ai_hwaccel.detect()
    print(registry)
    print()

    # List all profiles
    print("All detected devices:")
    for profile in registry.all_profiles():
        status = "available" if profile["available"] else "unavailable"
        print(f"  {profile['accelerator_str']} — {profile['memory_gb']:.1f} GB ({status})")
    print()

    # Best device
    best = registry.best_available()
    if best:
        print(f"Best device: {best['accelerator_str']} ({best['memory_gb']:.1f} GB)")
    else:
        print("No devices available")
    print()

    # Memory summary
    total_gb = registry.total_memory() / (1024**3)
    accel_gb = registry.total_accelerator_memory() / (1024**3)
    print(f"Total memory:       {total_gb:.1f} GB")
    print(f"Accelerator memory: {accel_gb:.1f} GB")
    print(f"Has accelerator:    {registry.has_accelerator()}")
    print()

    # Warnings
    warnings = registry.warnings()
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    # Suggest quantization for common model sizes
    print()
    print("Quantization suggestions:")
    for name, params in [("7B", 7_000_000_000), ("13B", 13_000_000_000), ("70B", 70_000_000_000)]:
        quant = registry.suggest_quantization(params)
        print(f"  {name} model: {quant}")

    # Round-trip JSON serialization
    json_str = registry.to_json()
    restored = ai_hwaccel.Registry.from_json(json_str)
    print(f"\nJSON round-trip OK: {len(restored)} profiles, schema v{restored.schema_version()}")


if __name__ == "__main__":
    main()
