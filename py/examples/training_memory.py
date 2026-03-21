#!/usr/bin/env python3
"""Training memory estimation example.

Estimates GPU/TPU/Gaudi memory requirements for fine-tuning
workloads with different methods.
"""

import ai_hwaccel


def print_estimate(model_name: str, method: str, target: str, est: dict) -> None:
    """Pretty-print a memory estimate."""
    print(f"  {model_name} / {method} on {target.upper()}:")
    print(f"    Model weights:  {est['model_gb']:6.1f} GB")
    print(f"    Optimizer:      {est['optimizer_gb']:6.1f} GB")
    print(f"    Activations:    {est['activation_gb']:6.1f} GB")
    print(f"    Total:          {est['total_gb']:6.1f} GB")
    print()


def main():
    # Compare training methods for a 7B model on GPU
    print("=== 7B Model — Training Methods on GPU ===")
    print()
    methods = ["full", "lora", "qlora", "prefix", "dpo", "rlhf", "distillation"]
    for method in methods:
        est = ai_hwaccel.estimate_training_memory(7000, method, "gpu")
        print_estimate("7B", method, "gpu", est)

    # Compare targets for LoRA on a 7B model
    print("=== 7B Model — LoRA across Targets ===")
    print()
    for target in ["gpu", "tpu", "gaudi", "cpu"]:
        est = ai_hwaccel.estimate_training_memory(7000, "lora", target)
        print_estimate("7B", "lora", target, est)

    # Compare model sizes for full fine-tune on GPU
    print("=== Full Fine-Tune on GPU — Model Sizes ===")
    print()
    for name, params_m in [("1.3B", 1300), ("7B", 7000), ("13B", 13000), ("70B", 70000)]:
        est = ai_hwaccel.estimate_training_memory(params_m, "full", "gpu")
        print_estimate(name, "full", "gpu", est)

    # Practical example: check if a 7B LoRA fits on a detected GPU
    registry = ai_hwaccel.detect()
    accel_mem_gb = registry.total_accelerator_memory() / (1024**3)
    est = ai_hwaccel.estimate_training_memory(7000, "lora", "gpu")

    print("=== Feasibility Check ===")
    print()
    print(f"  Available accelerator memory: {accel_mem_gb:.1f} GB")
    print(f"  7B LoRA requirement:          {est['total_gb']:.1f} GB")
    if accel_mem_gb >= est["total_gb"]:
        print("  Result: Fits! You can run 7B LoRA fine-tuning.")
    else:
        print("  Result: Does not fit. Consider QLoRA or a smaller model.")


if __name__ == "__main__":
    main()
