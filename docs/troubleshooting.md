# Troubleshooting

Common issues and their solutions.

---

## "nvidia-smi not found"

**Cause**: NVIDIA drivers are not installed, or `nvidia-smi` is not on `$PATH`.

**Fix**:
- Install the NVIDIA driver package for your distribution.
- Verify: `which nvidia-smi` should return a path.
- In containers, ensure the NVIDIA runtime is configured (`--gpus all` for Docker).

On Windows, `nvidia-smi.exe` must be on `PATH` (recent drivers put it in
System32; older ones in `C:\Program Files\NVIDIA Corporation\NVSMI`). Through
2.5.2 no vendor tool ran on Windows, whatever `PATH` held. The GPUs are found
through DXGI either way, as Windows GPU profiles.

The `warnings` array names each missing tool once (2.5.1), and each run of a
tool that failed, timed out, or exited with a non-zero status (2.5.4: the
warning's message says "exited with status N" or "killed by signal N").
Through 2.5.3 a tool that ran and failed was not listed. A failed
`vulkaninfo` that named no device now falls back to the sysfs scan, as a
missing one does, and a failed `nvidia-smi` no longer adds a `cuda` parse
warning for its error text. The interconnect pass looks for `nvidia-smi`
only when an NVIDIA GPU was found (2.5.2), and `system_profiler`, the Apple
backend's fallback, is looked for on macOS only (2.5.1). Through 2.5.0
`nvidia-smi` could appear twice and Linux runs listed `system_profiler`.

---

## "Detection returns CPU only"

**Cause**: Backend tools are not installed, the kernel driver is not loaded,
or the backend was not asked to run.

**Checklist**:
1. Are the detection tools installed? (`nvidia-smi`, `vulkaninfo`, `hl-smi`, etc.)
2. Is the kernel driver loaded? (`lsmod | grep nvidia`, `lsmod | grep amdgpu`)
3. Are sysfs paths accessible? (`ls /sys/class/drm/card*`)
4. Did the caller mask the backend out? `registry_detect_no_exec()` drops
   CUDA and oneAPI, and finds Gaudi, Neuron, Cerebras and Graphcore through
   their device nodes only (2.5.5; it skipped them before), and
   `registry_detect_with(mask)` runs only the backends in `mask`. A mask
   without a tool-based backend (`backend_uses_exec`) runs no tool at all
   (2.5.1), so Vulkan falls back to its sysfs scan.
5. Did the tool time out? A tool that prints nothing for 10 s is cut and its
   name appears in `warnings` (a timeout, `HWA_ERR_TIMEOUT`); `-vv` or the
   default log level shows "<tool>: timed out after 10s". A tool that ran and
   failed is listed too, and `-v` shows its exit status.

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

**Cause**: Vendor tools have startup overhead. A tool that hangs, such as
`nvidia-smi` on a GPU in an error state, is cut once it has printed nothing
for 10 s (2.5.1) and killed about 10 s later, and is listed in `warnings` as a
timeout; budget about 20 s per hung tool. Before 2.5.1 it blocked detection
until it exited.

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
- Where that is too long, use `registry_detect_no_exec()`, which runs no tool
  at all.

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

## "`AI_HWACCEL_BIN=... is not an executable file`" (Python)

**Cause**: since 2.5.5 a `binary=` argument or an `AI_HWACCEL_BIN` that names a
missing or non-executable file raises `BinaryNotFoundError`. Through 2.5.4 the
package ran the wheel-bundled binary or the one on `PATH` instead, without a
word, so the results came from a build the caller had not chosen.

**Fix**: point it at the binary you built (and `chmod +x` it), or unset
`AI_HWACCEL_BIN` to use the bundled one.

---

## "`pcie_bandwidth_x1000` is missing"

**Cause**: The profile has no PCI address (`pci_bdf`: `nvidia-smi`'s
`pci.bus_id`, or the ROCm card's device link), or the device's
`max_link_width` / `max_link_speed` (falling back to `current_link_*`) cannot
be read, or it is not a PCI device. Since 2.5.5 the value is the link
capability, clipped to the slot (past AMD's on-card switch), not the
power-managed current link. Only CUDA and ROCm profiles get PCIe enrichment.

**Fix**:
- Check: `cat /sys/bus/pci/devices/<bdf>/max_link_speed`
- Some devices (integrated GPUs, Apple Silicon) don't have PCIe links.

---

## "`numa_node` is missing"

**Cause**: The system doesn't have NUMA topology, or the sysfs `numa_node`
file returns `-1` (meaning "no NUMA affinity").

This is normal for single-socket systems.
