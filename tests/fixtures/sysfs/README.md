# sysfs captures

Real sysfs files, used by `tests/tcyr/topology_test.tcyr` to run
`detect_rocm_at`, `enrich_pcie_at` and `enrich_numa_at` against a fake tree.
The test builds the tree under `/tmp` at run time, because the parts that
matter are symlinks (a DRM card's `device` link names its PCI address, its
`driver` link the driver), and copies these files into it. Tests read them
relative to the repository root, which is where CI runs them.

| file | sysfs path |
|---|---|
| `renoir/kfd-node0-properties.txt` | `/sys/class/kfd/kfd/topology/nodes/0/properties` (the CPU node) |
| `renoir/kfd-node1-properties.txt` | `/sys/class/kfd/kfd/topology/nodes/1/properties` (the GPU node) |
| `renoir/kfd-node1-mem-bank0-properties.txt` | `/sys/class/kfd/kfd/topology/nodes/1/mem_banks/0/properties` |
| `renoir/pp_dpm_mclk.txt` | `/sys/class/drm/card1/device/pp_dpm_mclk` |

Captured on 2026-10-29 on the Linux dev host (Arch, kernel 7.2.11, AMD Ryzen
7 5800H): an integrated Cezanne GPU (vendor 0x1002, device 0x1638) at
0000:04:00.0, bound to amdgpu, the host's only DRM card, card1.

What they pin down:

- `pp_dpm_mclk` lines are "<level>: <MHz>Mhz", "*" on the current one: the
  clock is the number after the colon (1600), not the level before it (3).
- The GPU node's `gfx_target_version` is 90012, gfx90c; its `location_id`
  1024 and `domain` 0 are the card's PCI address, 0000:04:00.0. The PCI
  revision (0xc5), which 2.5.4 reported as the compute capability, is not
  the target.
- The memory bank's `width` is the bus width, 128 bits of dual-channel
  DDR4-3200: with the 1600 MHz clock, 51.2 GB/s.
