# Roadmap

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## Remaining pre-v1

- [ ] Publish to crates.io.

---

## Next release: System I/O and monitoring

- [ ] **VRAM read/write bandwidth probing** -- benchmark actual memory throughput
  per device using vendor tools (`nvidia-smi -q -d MEMORY`, `rocm-smi
  --showmeminfo`). Expose as `AcceleratorProfile::memory_bandwidth_gbps`.
- [ ] **Network topology detection** -- detect InfiniBand, RoCE, NVLink
  interconnects for multi-node training planning. Parse `ibstat`, `ibv_devinfo`,
  or `nvidia-smi nvlink -s`.
- [ ] **Disk I/O throughput** -- estimate data loading bottlenecks by probing
  `/sys/block/*/queue/` for NVMe vs HDD and detecting RAID configurations.
- [ ] **NUMA topology** -- parse `/sys/devices/system/node/` to map which GPUs
  are on which NUMA node for optimal CPU-GPU data placement.
- [ ] **PCIe bandwidth detection** -- read `lspci -vv` for link width/speed to
  estimate host-to-device transfer rates.
- [ ] **Runtime memory monitoring** -- expose current VRAM usage (not just
  total) via `nvidia-smi -q -d MEMORY` for live capacity planning.
- [ ] **Network ingestion estimation** -- given a dataset size and detected
  network bandwidth, estimate data loading time for training jobs.

---

## Hardware-dependent (no deadline -- based on access)

These items require real hardware or cloud instances. They will be addressed
as hardware becomes available, in any order.

### Detection accuracy

- [ ] **AMD ROCm**: query `rocm-smi` for clock speeds, firmware, XGMI topology.
- [ ] **Google TPU**: validate on real GCE VMs; multi-host pod slices.
- [ ] **Intel Gaudi**: test on Gaudi 3 hardware; parse firmware version.
- [ ] **AWS Neuron**: mixed Inferentia + Trainium in the same instance.
- [ ] **Intel oneAPI**: Data Center GPU Max HBM vs DDR memory tiers.
- [ ] **Vulkan**: compute queue family parsing from `vulkaninfo`.

### New backends

- [ ] **AMD MI300X** -- CXL-attached memory, unified HBM pool.
- [ ] **Cerebras WSE** -- `/dev/cerebras*` or SDK introspection.
- [ ] **Graphcore IPU** -- `gc-info` CLI parsing.
- [ ] **Groq LPU** -- when Linux driver is publicly available.
- [ ] **Samsung NPU** -- Exynos AI accelerator sysfs.
- [ ] **MediaTek APU** -- Android/Linux APU sysfs.

### Platform support

- [ ] **macOS ANE via IOKit** -- native ANE detection beyond `system_profiler`.
- [ ] **Windows** -- WMI queries, DirectML enumeration.
- [ ] **Android** -- HAL `hwbinder` for NNAPI accelerator list.
- [ ] **FreeBSD** -- DRM sysctl equivalents.

---

## Post-v1 features (no deadline)

These are longer-term items that don't block any release.

- [ ] **Topology-aware sharding** -- NVLink/NVSwitch/XGMI/ICI topology for
  optimal tensor placement.
- [ ] **NVLink/NVSwitch detection** -- parse `nvidia-smi topo -m`.
- [ ] **Power and thermal monitoring** -- current power draw, temperature,
  throttling state.
- [ ] **Hot-plug support** -- `udev`/`inotify` watch for device add/remove.
- [ ] **Cost-aware planning** -- cloud pricing data for cheapest config.
- [ ] **WASM target** -- browser-based ML dashboards with stubbed detection.
- [ ] **Full Python bindings** -- complete PyO3 package for the ML ecosystem.
- [ ] **Reduce allocations** -- `Cow<str>` for device names, `SmallVec` for
  profiles.
- [ ] **Lazy detection** -- detect only backends the caller queries.

---

## Non-goals

- **Runtime execution** -- detection and planning only, not inference/training.
- **Kernel driver management** -- no installing or configuring drivers.
- **Cloud provisioning** -- detect what's present, not what could be spun up.
