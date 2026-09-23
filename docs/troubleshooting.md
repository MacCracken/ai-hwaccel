# Troubleshooting

Common issues and their solutions.

---

## "nvidia-smi not found"

**Cause**: NVIDIA drivers are not installed, or `nvidia-smi` is not on `$PATH`.

**Fix**:
- Install the NVIDIA driver package for your distribution.
- Verify: `which nvidia-smi` should return a path.
- In containers, ensure the NVIDIA runtime is configured (`--gpus all` for Docker).

On Windows no vendor tool runs yet (see the roadmap), so the warnings always
name them. The GPUs are still found, as Windows GPU profiles, through DXGI.

The `warnings` array names each tool that was missing or failed, once per
attempt. So `nvidia-smi` can appear twice (the CUDA backend and the
interconnect pass). On Linux it also lists `system_profiler`, the Apple
backend's macOS fallback. Neither is a problem.

---

## "Detection returns CPU only"

**Cause**: Backend tools are not installed, the kernel driver is not loaded,
or the backend was not asked to run.

**Checklist**:
1. Are the detection tools installed? (`nvidia-smi`, `vulkaninfo`, `hl-smi`, etc.)
2. Is the kernel driver loaded? (`lsmod | grep nvidia`, `lsmod | grep amdgpu`)
3. Are sysfs paths accessible? (`ls /sys/class/drm/card*`)
4. Did the caller mask the backend out? `registry_detect_no_exec()` skips the
   tool-based backends (CUDA, Gaudi, Neuron, oneAPI, Cerebras, Graphcore), and
   `registry_detect_with(mask)` runs only the backends in `mask`.

---

## "Vulkan GPU listed instead of a CUDA/ROCm GPU"

**Expected behavior**: a GPU that both Vulkan and a dedicated backend (CUDA or
ROCm) detect is listed once, as the CUDA or ROCm profile, which has the
driver, temperature and utilization. The two are matched on the card's PCI
vendor and device ID (`nvidia-smi`'s `pci.device_id`, sysfs, and
`vulkaninfo`'s `vendorID`/`deviceID`). A Vulkan GPU no dedicated backend
reports, such as an Intel iGPU next to an NVIDIA card, stays. If you see only
the Vulkan profile, the dedicated backend failed.

**Fix**: Check that `nvidia-smi` or `/sys/class/drm/card*/device/driver` is
working. Run with `--log-level debug` (or `AI_HWACCEL_LOG=debug`) to see
detection diagnostics, including a `dedup:` line for each Vulkan view that was
dropped:

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
- A CPU profile of exactly 17 179 869 184 bytes (16 GiB) is the fallback used
  when total RAM could not be read (`/proc/meminfo` on Linux, `sysctl
  hw.memsize` on macOS, `GlobalMemoryStatusEx` on Windows).

Totals count system RAM once. A GPU or NPU that uses system RAM (Apple
Silicon, laptop NPUs, integrated GPUs) carries `shared_memory_bytes`, and
`total_memory_bytes` adds that part only once.

---

## "Detection is slow or hangs"

**Cause**: Vendor tools have startup overhead, and they run without a time
limit (bounding them is on the roadmap). A tool that hangs, such as
`nvidia-smi` on a GPU in an error state, blocks detection until it exits.

**Fix**:
- Find the tool: at `-vv` the log shows `[ENTER] detect` with no matching
  `[EXIT]`. Run the vendor tools by hand (`timeout 10 nvidia-smi`, and so on).
- Probe only the backends you need:
  ```cyrius
  var r = registry_detect_with(builder_with(builder_none(), BACKEND_CUDA));
  ```
- Cache the registry instead of re-detecting on every call:
  ```cyrius
  var c = cached_registry_new(60);
  var r = cached_get(c);
  ```
- Where a hang is unacceptable, use `registry_detect_no_exec()`, which runs no
  tool at all.

---

## "Schema version mismatch" (Python)

**Cause**: `ai_hwaccel.detect()` warns "ai-hwaccel binary reports JSON schema
vN, these bindings target vM" when the binary and the Python package come from
different releases, typically when `AI_HWACCEL_BIN` or `binary=` points at
another build.

**Fix**: use the binary the wheel bundles, or a build of the same release. The
JSON's `schema_version` (6 since 2.3.28) changes whenever keys are added or
changed; [docs/schema.json](schema.json) describes the current format.

---

## "`pcie_bandwidth_x1000` is missing"

**Cause**: The sysfs PCI device files `current_link_width` and
`current_link_speed` are not readable, or the device is not a PCI device.
Only CUDA and ROCm profiles get PCIe enrichment.

**Fix**:
- Check: `cat /sys/bus/pci/devices/*/current_link_speed`
- Some devices (integrated GPUs, Apple Silicon) don't have PCIe links.

---

## "`numa_node` is missing"

**Cause**: The system doesn't have NUMA topology, or the sysfs `numa_node`
file returns `-1` (meaning "no NUMA affinity").

This is normal for single-socket systems.
