# Troubleshooting

Common issues and their solutions.

---

## "nvidia-smi not found"

**Cause**: NVIDIA drivers are not installed, or `nvidia-smi` is not on `$PATH`.

**Fix**:
- Install the NVIDIA driver package for your distribution.
- Verify: `which nvidia-smi` should return a path.
- In containers, ensure the NVIDIA runtime is configured (`--gpus all` for Docker).

---

## "Detection returns CPU only"

**Cause**: Backend tools are not installed or backend features are disabled.

**Checklist**:
1. Are the detection tools installed? (`nvidia-smi`, `vulkaninfo`, `hl-smi`, etc.)
2. Is the kernel driver loaded? (`lsmod | grep nvidia`, `lsmod | grep amdgpu`)
3. Are sysfs paths accessible? (`ls /sys/class/drm/card*`)
4. Was the binary built with the right `-D` flags? Default builds all backends.

If using selective backends:

```sh
cyrius build src/main.cyr build/ai-hwaccel -DCUDA
```

---

## "Vulkan GPU listed instead of CUDA/ROCm GPU"

**Expected behavior**: When both Vulkan and a dedicated backend (CUDA or ROCm)
detect the same GPU, the Vulkan duplicate is automatically removed. If you only
see Vulkan, it means the dedicated backend failed.

**Fix**: Check that `nvidia-smi` or `/sys/class/drm/card*/device/driver` is
working. Run with `--debug` to see detection diagnostics:

```sh
AI_HWACCEL_LOG=debug ai-hwaccel --table
```

---

## "Memory values are zero or wrong"

**Cause**: Sysfs files may have restrictive permissions, or the driver may not
expose memory info on older versions.

**Fix**:
- Check permissions: `cat /sys/class/drm/card0/device/mem_info_vram_total`
- Update GPU drivers to a recent version.
- For NVIDIA, ensure `nvidia-smi` works without errors.

---

## "Detection is slow"

**Cause**: CLI tools have high startup overhead, or a tool is hanging.

**Fix**:
- Use `DetectBuilder` to probe only the backends you need:
  ```cyr
  let registry = registry_detect_builder()
      .with_cuda()
      .detect();
  ```
- Use `CachedRegistry` to avoid re-detecting on every call:
  ```cyr
  let cache = CachedRegistry::new(60);
  let registry = cache.get();
  ```
- Check for hanging tools: run with `--debug` and look for `Timeout` warnings.

---

## "Schema version mismatch"

**Cause**: A cached registry file was created with an older version of ai-hwaccel.

**Fix**:
- Call `CachedRegistry::invalidate()` to force a fresh detection.
- Delete cached JSON files and re-detect.
- The manual JSON parser handles version differences gracefully —
  old JSON will deserialize with new fields set to `nil`/empty.

---

## "Timeout warnings for a specific tool"

**Cause**: A detection tool (`nvidia-smi`, `vulkaninfo`, etc.) took longer than
the 5-second default timeout.

**Note**: Timeouts now return `DetectionError::Timeout` (not `ToolFailed`),
making them easy to identify and retry programmatically.

---

## "PCIe bandwidth shows as `-`"

**Cause**: The sysfs PCI device files `current_link_width` and
`current_link_speed` are not readable, or the device is not a PCI device.

**Fix**:
- Check: `cat /sys/bus/pci/devices/*/current_link_speed`
- Some devices (integrated GPUs, Apple Silicon) don't have PCIe links.

---

## "NUMA node shows as `-`"

**Cause**: The system doesn't have NUMA topology, or the sysfs `numa_node`
file returns `-1` (meaning "no NUMA affinity").

This is normal for single-socket systems.
