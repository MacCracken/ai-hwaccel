# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [semantic versioning](https://semver.org/) as of v0.19.3.

## [2.4.0] — 2026-09-23 — one physical device, one profile

A GPU that two backends report is now listed once. Vulkan sees every card that
CUDA and ROCm see, and ai-hwaccel listed both views. So the dev host's single
AMD APU appeared twice, and its 8 GiB counted twice. A new post-pass in every
detection entry point drops the Vulkan view when a CUDA or ROCm profile has
the same PCI vendor and device ID. It also drops a Vulkan view of Apple's Metal
GPU (MoltenVK on macOS, Honeykrisp on Asahi Linux). The dedicated backend's
profile survives with its own memory and fields. Profile counts and totals
change on any host with such a GPU and `vulkan-tools` installed, hence 2.4.0.
The JSON schema stays v6: no key is added or removed.

### Fixed

- **A GPU seen by Vulkan and by CUDA or ROCm was listed and counted twice.**
  Profiles now carry an internal `pci_id`, the PCI vendor and device ID:
  - from `nvidia-smi`'s `pci.device_id`, a twelfth query field;
  - from sysfs for ROCm and for Vulkan's sysfs scan;
  - from `vulkaninfo`'s `vendorID` and `deviceID`.

  `profiles_dedup` (`src/registry.cyr`) runs first in `registry_post_passes`,
  and in the lazy registry after each family's profiles are merged. It drops:
  - every Vulkan profile whose `pci_id` matches a CUDA or ROCm profile's. A GPU
    listed by two Vulkan drivers (RADV and AMDVLK) is dropped twice over. A
    GPU no dedicated backend reports stays, such as an Intel iGPU next to an
    NVIDIA card. The Rust releases dropped every Vulkan GPU whenever any CUDA
    or ROCm GPU was found, that iGPU included.
  - an integrated Vulkan GPU on system RAM when an Apple Metal profile exists.

  The key is the ID, not the bus address: `vulkaninfo --summary` prints no bus
  address, and identical cards need none to be matched. On the dev host, with
  real `vulkaninfo` and sysfs:

  | | 2.3.29 | 2.4.0 |
  |---|---|---|
  | GPU profiles | ROCm "AMD Radeon (PCI 0x1002:0x1638)" 8 GiB; Vulkan "AMD Radeon Graphics (RADV RENOIR)" 8 GiB | ROCm "AMD Radeon Graphics (RADV RENOIR)" 8 GiB |
  | `gpu_count` | 2 | 1 |
  | `accelerator_memory_bytes` | 17 179 869 184 | 8 589 934 592 |
  | `total_memory_bytes` | 76 060 057 600 | 67 470 123 008 |

  An end-to-end run on the dev host used stand-in `nvidia-smi` and
  `vulkaninfo` first on `PATH`. The `nvidia-smi` stand-in reported two RTX
  4090s. The `vulkaninfo` stand-in listed both cards, `cass`'s UHD 600, a
  lavapipe device and the host's real APU. 2.3.29 reports 7 GPU profiles,
  total 136 164 433 920. 2.4.0 reports 4 (both CUDA cards, the ROCm APU, the
  UHD 600), total 118 984 564 736. The same happens from
  `registry_detect`, `registry_detect_threaded` and `lazy_into_registry`.
- **Apple Silicon with MoltenVK listed its GPU twice.** On `ecb`, with a
  stand-in `vulkaninfo` that reports what MoltenVK does, 2.3.29 lists the M5
  Pro twice: Metal 48 GiB, and Vulkan 36 GiB with `gpu_count` 2. 2.4.0 lists
  it once. The totals were already right, because both views are shared
  memory (2.3.28, 2.3.29).

### Changed

- **The survivor keeps a better name.** When a dedicated profile's name is a
  placeholder, it takes the dropped Vulkan profile's name. The placeholders are
  ROCm's sysfs fallback "AMD Radeon (PCI vendor:device)" and the Apple
  backend's "Apple Silicon" / "Apple Silicon (Asahi Linux)". So the dev host's
  APU is now "AMD Radeon Graphics (RADV RENOIR)". A real name is never
  replaced.
- **Each drop is logged at debug level:** `dedup: Vulkan GPU … is the ROCm
  GPU …; dropped, its name kept`.
- **The `nvidia-smi` query gains `pci.device_id`.** Fields are read by position,
  so output without it still parses, with the ID unknown. The query was not
  run on real NVIDIA hardware (none here); the parser is tested on its
  documented output, `0x268410DE` for device 0x2684 of vendor 0x10DE.
- **New:**
  - the profile's 22nd field `pci_id` (`PROFILE_SIZE` 176; not serialized) and
    `pci_id_make`;
  - `profiles_dedup` and `profile_name_is_placeholder`;
  - `parse_hex_field`, which is `vulkan.cyr`'s `_vk_hex` moved to `command.cyr`
    for the CUDA parser.
- **Tests: 894 → 950 assertions in 15 units.**
  - `registry_test` 113 → 149. The rules: the dev host's APU, identical
    cards next to an iGPU, two Vulkan drivers, no false matches, Apple (macOS,
    Asahi, a dedicated GPU). Placeholder names. A property test over 300
    pseudo-random profile sets: no duplicate left, nothing else dropped, order
    kept, idempotent. Every entry point, on the host it runs on.
  - `gpu_parser_test` 141 → 148: `pci.device_id` and `pci_id` from the real
    `vulkaninfo` captures.
  - `profile_test` 103 → 111: `pci_id_make`.
  - `json_output_test` 45 → 47: `pci_id` is not serialized, and the schema is
    still v6.
  - `lazy_test` 35 → 38: an NPU query, then a GPU query whose MoltenVK view
    arrives in another partial registry.

  Each of 18 single-point mutations of the new code fails at least one
  assertion. The property test catches a 19th that the hand-written cases miss:
  a duplicate in the first position never dropped.
- **Fuzz:** `fuzz/cuda_parser.fcyr` covers `pci.device_id`, valid, missing and
  malformed.
- **Benchmarks: 17 → 18 rows.** `dedup_17dev` runs the pass on 17 profiles,
  4 of them Vulkan views of CUDA cards, refill included (batch-timed).
- **Docs:**
  - the README's detection list, module tree and test table;
  - the architecture overview's post-passes;
  - `docs/troubleshooting.md`, which also named a nonexistent `--debug` flag,
    as did `docs/guides/testing.md` and `docs/guides/production.md` (now
    `--log-level debug`);
  - the counts in `CLAUDE.md` and `docs/guides/testing.md`.
- **Roadmap:** the duplicate-device item is closed. The Windows item now names
  what remains there: a `pci_id` on DXGI profiles.

### Known, not changed here

- **Windows has nothing to merge yet.** No vendor tool runs there (the roadmap's
  `which()` item), so DXGI is the only GPU backend. DXGI profiles have no
  `pci_id`, and DXGI is not a survivor for Vulkan.
- **Intel oneAPI (`xpu-smi`) profiles have no `pci_id`.** An Intel data-center
  GPU seen by `xpu-smi` and by Vulkan is still listed twice. No host has one,
  and the `--dump` field IDs the parser uses are unverified.
- **Platform GPUs other than Apple's have no PCI ID.** A Jetson seen by CUDA and
  Vulkan is an example. Such a pair is not matched.

### Performance

**17 neutral, 0 regressions** (2.3.29 → 2.4.0). The method is 2.3.28's: 5
code layouts per arm, the three floor-bound rows batch-timed (†), and two
30-round passes, shuffled and pinned to one CPU (60 rounds). The two rows
2.3.29 added are compared too.

| row | 2.3.29 | 2.4.0 | Δ | p | verdict |
|---|---:|---:|---:|---:|---|
| `parse_cuda_8gpu` | 14.13 µs | 14.14 µs | +0.1% | 0.80 | neutral |
| `parse_vulkan_2gpu` | 2.67 µs | 2.60 µs | −2.6% | 0.03 | neutral |
| `parse_neuron_2dev` | 1.67 µs | 1.70 µs | +1.5% | 0.17 | neutral |
| `detect_safetensors` † | 406.8 ns | 399.7 ns | −1.7% | 0.03 | neutral |
| `detect_gguf` † | 38.6 ns | 38.6 ns | +0.1% | 0.85 | neutral |
| `best_available_13dev` † | 180.5 ns | 181.8 ns | +0.7% | 0.50 | neutral |
| `total_memory_13dev` | 105.6 ns | 106.1 ns | +0.5% | 0.33 | neutral |
| `has_accelerator_13dev` | 18.7 ns | 18.6 ns | −0.1% | 0.55 | neutral |
| `plan_70B_bf16_4gpu` | 1.46 µs | 1.46 µs | +0.4% | 0.27 | neutral |
| `count_family_gpu_13dev` | 248.2 ns | 249.7 ns | +0.6% | 0.18 | neutral |
| `json_serialize_13dev` | 23.54 µs | 23.51 µs | −0.1% | 0.57 | neutral |
| `json_summary_13dev` | 3.24 µs | 3.23 µs | −0.1% | 0.81 | neutral |
| `json_system_io` | 5.25 µs | 5.27 µs | +0.4% | 0.24 | neutral |
| `json_plan` | 17.44 µs | 17.44 µs | 0.0% | 0.96 | neutral |
| `json_training` | 2.53 µs | 2.53 µs | 0.0% | 0.95 | neutral |
| `parse_vulkan_summary_renoir` | 15.32 µs | 14.88 µs | −2.9% | <0.01 | neutral (placement) |
| `vulkan_heaps_uhd600` | 634.70 µs | 646.13 µs | +1.8% | 0.10 | neutral |

A verdict needs p < 0.01 and |Δ| > 1%. `parse_vulkan_summary_renoir` crosses
the bar, but it is not a win. The parse does strictly more work than before:
each profile also records its `pci_id` and is 8 bytes larger. Its code moved
when `_vk_hex` left `vulkan.cyr`, and the parse runs on the same input. The
detection path's new cost is the pass: `dedup_17dev` measures 907 ns (mean of
per-layout medians), refill included, next to detection's tens of
milliseconds. The host was running other work (load ~4), and the arms were
interleaved.

In `bench-history.csv`, the `028b1e1-dirty` rows at 18:16:43Z are 2.4.0. The
`028b1e1` rows at 18:16:45Z are 2.3.29, measured from `git archive 028b1e1`
right after them.

| binary | 2.3.29 | 2.4.0 | Δ |
|---|---:|---:|---:|
| x86_64 ELF, `CYRIUS_DCE=1` | 231 960 | 236 120 | +4 160 (text +3 912) |
| x86_64 ELF, no DCE | 457 240 | 461 400 | +4 160 (text +3 608) |
| ELF-aarch64 | 805 224 | 805 288 | +64 (text +4 208) |
| PE, as shipped (no DCE) | 529 920 | 534 016 | +4 096 |
| agnos, `CYRIUS_DCE=1` | 229 624 | 233 792 | +4 168 (text +3 912) |
| Mach-O arm64 (cross-built, as verified) | 721 332 | 737 716 | +16 384 (one 16 KiB page) |

### Verified

- **Tests:** 950 assertions in 15 units pass through CI's loop; fuzz 6/6.
- **CI gates:** fmt, lint (0 warnings in `src/`), vet, raw-offset guard,
  `cyrius.lock` current, DCE build. `dist/ai-hwaccel.cyr` differs from
  2.3.29's only in its version line and the seven changed modules (`profile`,
  `command`, `cuda`, `rocm`, `vulkan`, `registry`, `lazy`), and `distlib` is
  deterministic.
- **Linux CLI output matches 2.3.29** on 18 invocations covering every flag,
  apart from the dedup itself:
  - the APU's Vulkan profile and table row are gone;
  - ROCm takes RADV's name;
  - the counts and totals change;
  - the `-v` log reports 2 profiles, not 3.

  **ELF-aarch64** under `qemu-aarch64` matches x86_64 on 8 invocations, dedup
  included. **agnos** builds.
- **Stand-ins** (the table under *Fixed*): every entry point drops the same
  views, and `registry_detect_no_exec()` runs neither tool.
- **Windows on `cass`:** the EXE exactly as `stage_win_cross.sh` builds it
  passes `windows-smoke` (a)–(d). Its JSON and `--summary` match 2.3.29's EXE:
  there is nothing to merge on Windows.
- **macOS on `ecb`**, cross-built:
  - without a stand-in, the JSON matches 2.3.29;
  - with the MoltenVK stand-in, all five detection paths hold one M5 Pro GPU;
  - the debug log shows the drop.
- **Not verified:** a real NVIDIA host (the `pci.device_id` query), a real
  MoltenVK or Honeykrisp install, and two Vulkan drivers on one AMD card.

## [2.3.29] — 2026-09-23 — integrated GPUs through Vulkan: real size, counted once, found without exec

Every Vulkan GPU carried a 4 GiB estimate and counted as memory of its own.
An integrated GPU's memory is system RAM, apart from an AMD APU's carve-out, so
on a Linux host with an Intel iGPU the totals counted RAM a second time. Now
integrated GPUs report their real memory. An Intel (or any non-AMD) iGPU
reports its largest device-local heap from `vulkaninfo`, all of it shared; an
AMD APU reports its BIOS carve-out from sysfs. Software implementations
(lavapipe) are no longer reported as GPUs. `registry_detect_no_exec()` now
runs the Vulkan backend's sysfs scan, so on Linux it finds the Intel and NVIDIA
GPUs it used to miss. Discrete GPUs are unchanged. Schema stays v6.

### Fixed

- **An integrated GPU seen through Vulkan counted system RAM a second time.**
  `vulkaninfo --summary` prints no memory, so every Vulkan GPU got a 4 GiB
  estimate and a `shared_memory_bytes` of 0. `src/detect/vulkan.cyr` now
  reads each device's `vendorID`, `deviceID` and `deviceType`:
  - An integrated GPU of any vendor but AMD has no memory of its own. Its
    profile is all shared. It is sized from its largest device-local heap in
    the full `vulkaninfo` output, a second run made only when such a GPU is
    present. The full output runs to 60 KB (the UHD 600) and 84 KB (the dev
    host's APU) for a single device, and `run_tool` keeps 1 MiB, so it cannot
    replace `--summary` for naming every device on a many-GPU host.
  - An AMD APU's own memory is its BIOS carve-out, which is outside
    `MemTotal`. It is read from sysfs `mem_info_vram_total`, the figure ROCm
    reports. RADV's heaps on an APU split carve-out plus GTT 2:1, so they do
    not give it: on the dev host the device-local heap is 23.61 GiB for an
    8 GiB carve-out.
  - The exec-less fallback (no `vulkaninfo`) marks an Intel GPU at PCI
    `0000:00:02.0` shared. That is where Intel has put its integrated GPU
    since 2008; Arc cards sit behind a root port.

  Checked end to end on the dev host, with the real detection code and a
  stand-in `vulkaninfo` first on `PATH` that serves `cass`'s real captures
  (Intel UHD 600, Intel's Windows driver):

  | | 2.3.28 | 2.3.29 |
  |---|---:|---:|
  | UHD 600 profile | 4 294 967 296, not shared | 4 202 799 104 (its heap, half of `cass`'s RAM), all shared |
  | `total_memory_bytes` | 71 765 090 304 | 67 470 123 008 (the host's RAM plus its ROCm GPU, nothing twice) |

  `registry_detect`, `registry_detect_threaded` and `lazy_into_registry`
  agree. The stand-in logged two runs per entry point with an integrated GPU
  and one without.
- **Software Vulkan implementations were reported as GPUs.** lavapipe
  (`llvmpipe`) and SwiftShader have `deviceType` CPU. Each was a 4 GiB "Vulkan
  GPU", so a CPU-only host with Mesa's Vulkan drivers installed reported
  `has_accelerator: true`. They are skipped now, as DXGI's Basic Render Driver
  is. With the stand-in serving the UHD 600 plus a lavapipe device, 2.3.28
  reports both (total 76 060 057 600); 2.3.29 reports the GPU alone.
- **`registry_detect_no_exec()` found no Intel or NVIDIA GPU on Linux.** The
  Vulkan backend was classed exec, so no-exec mode dropped it, and nothing
  else reports those GPUs without a subprocess. `BACKEND_VULKAN` is native now,
  as Windows and Apple are. `detect_vulkan_opts(profiles, warnings,
  allow_exec, leave_amdgpu)` runs `vulkaninfo` only with exec and the sysfs
  scan otherwise. In no-exec mode the scan leaves `amdgpu` devices to ROCm,
  which reads the same sysfs nodes, so no-exec mode gains no second profile for
  an AMD GPU. On the dev host the no-exec registry is identical to 2.3.28's,
  and the logging stand-in recorded no `vulkaninfo` run from it.
- **The docs promised a dedup that the Cyrius port does not have.**
  `docs/troubleshooting.md` said a GPU found by both Vulkan and CUDA/ROCm "is
  automatically removed" from Vulkan, and `docs/architecture/overview.md`
  listed a "dedup vulkan vs cuda/rocm" pass. The Rust releases dropped every
  Vulkan GPU whenever any CUDA or ROCm GPU was found, a separate iGPU included.
  The port never did even that: `src/` has no dedup, and the dev host lists its
  one APU twice. Both pages now describe what happens, and the roadmap's
  duplicate-device item records the history.

### Changed

- **An AMD APU's Vulkan profile reports its carve-out, not 4 GiB.** On the dev
  host that is 8 GiB, the figure its ROCm profile has always reported. The same
  iGPU is still listed twice, once by ROCm and once by Vulkan (the 2.4.x
  duplicate-device item). So its `accelerator_memory_bytes` rises from 12 to
  16 GiB, and `total_memory_bytes` from 71 765 090 304 to 76 060 057 600. On an
  APU with the common 512 MB – 2 GiB carve-out, the double count shrinks
  instead.
- **New functions:**
  - `detect_vulkan_opts`. `detect_vulkan` keeps its signature and allows exec.
  - `vulkan_device_type` and `vulkan_uses_system_memory`.
  - `vulkan_local_heap_bytes`, `vulkan_size_shared` and
    `_parse_vulkaninfo_ids`.
  - `pci_slot_is_intel_igpu`.

  The parser's `heapSize` lookahead is gone. `vulkaninfo` has no such key, so
  it never matched.
- **Tests: 791 → 894 assertions in 15 units.**
  - `gpu_parser_test` 44 → 141. The Vulkan parser, the heap scan and the
    sizing run against real `vulkaninfo` captures, now in
    `tests/fixtures/vulkaninfo/`: the dev host's APU under RADV (LF) and
    `cass`'s UHD 600 under Intel's Windows driver (CRLF, kept byte-exact by a
    new `.gitattributes`). Hand-written inputs cover lavapipe, 1.1-era
    `vulkaninfo`, NVIDIA's overlapping BAR heap, a heap above the estimate,
    and nested structs that repeat the identity fields.
  - `registry_test` 107 → 113:
    - totals with the UHD 600 sized from its captures;
    - Vulkan native, and in the no-exec mask;
    - no-exec mode leaving exactly ROCm's devices, with no `vulkaninfo` run.

  Each of 22 single-point mutations of the new code fails at least one
  assertion or the fuzz harness.
- **Fuzz:** `fuzz/vulkan_parser.fcyr` now damages every byte of a summary
  block and a heap block in turn, with each of nine structural bytes, and
  also cuts each input short there. Nothing may crash, and the results must
  stay bounded.
- **Benchmarks: 15 → 17 rows.** `parse_vulkan_summary_renoir` parses the dev
  host's real `--summary` output (batch-timed), and `vulkan_heaps_uhd600` scans
  `cass`'s 60 KB full output. Both read the fixtures from the repository root
  and skip themselves elsewhere.
- **Docs:** the README's backend table, module tree, no-exec section (six exec
  backends, eleven native) and test table; the architecture overview's
  detection flow; `docs/troubleshooting.md`; `docs/performance.md`;
  `docs/guides/testing.md`.
- **Roadmap:** the Vulkan iGPU item is closed. The duplicate-device item now
  names the PCI bus address that full `vulkaninfo` output carries. One item is
  new: no vendor tool ever runs on Windows. `which()` splits `PATH` on `:` and
  looks for the bare name, so `nvidia-smi` and `vulkaninfo` (which `cass` has,
  in `System32`) report "tool not found".

### Known, not changed here

- **Discrete GPUs through Vulkan keep the 4 GiB estimate.** CUDA and ROCm
  report the same cards. Which profile survives, with which memory figure, is
  the 2.4.x duplicate-device item. Giving them real sizes first would make
  every NVIDIA host with `vulkan-tools` count its VRAM twice.
- **The Windows and macOS binaries never run `vulkaninfo`.** Windows cannot
  find it (the `which()` item above), and macOS has no `/sys` for the scan.
  Their output is unchanged.

### Performance

**15 neutral, 0 regressions** (2.3.28 → 2.3.29). The method is 2.3.28's.
Each arm was built at 5 code layouts, and the three floor-bound rows were
batch-timed (†). There were two independent 30-round passes, shuffled and
pinned to one CPU; the table combines them (60 rounds).

| row | 2.3.28 | 2.3.29 | Δ | p | verdict |
|---|---:|---:|---:|---:|---|
| `parse_cuda_8gpu` | 13.65 µs | 13.70 µs | +0.4% | 0.29 | neutral |
| `parse_vulkan_2gpu` | 2.55 µs | 2.55 µs | 0.0% | 1.00 | neutral |
| `parse_neuron_2dev` | 1.58 µs | 1.62 µs | +2.0% | 0.05 | neutral (layout) |
| `detect_safetensors` † | 420.1 ns | 396.2 ns | −5.7% | <0.01 | neutral (placement) |
| `detect_gguf` † | 37.6 ns | 37.5 ns | −0.3% | 0.27 | neutral |
| `best_available_13dev` † | 169.7 ns | 170.2 ns | +0.3% | 0.67 | neutral |
| `total_memory_13dev` | 102.8 ns | 102.8 ns | 0.0% | 1.00 | neutral |
| `has_accelerator_13dev` | 18.2 ns | 18.2 ns | 0.0% | 0.73 | neutral |
| `plan_70B_bf16_4gpu` | 1.43 µs | 1.42 µs | −0.4% | 0.25 | neutral |
| `count_family_gpu_13dev` | 243.0 ns | 241.8 ns | −0.5% | 0.11 | neutral |
| `json_serialize_13dev` | 22.83 µs | 22.83 µs | 0.0% | 0.89 | neutral |
| `json_summary_13dev` | 3.15 µs | 3.16 µs | +0.2% | 0.29 | neutral |
| `json_system_io` | 5.12 µs | 5.12 µs | 0.0% | 0.96 | neutral |
| `json_plan` | 16.99 µs | 16.96 µs | −0.2% | 0.08 | neutral |
| `json_training` | 2.46 µs | 2.46 µs | +0.2% | 0.49 | neutral |

A verdict needs p < 0.01 and |Δ| > 1%. Neither flagged row runs changed code,
and each shift is reproduced by 2.3.28's own code laid out like 2.3.29's:

- **`detect_safetensors`, −5.7%, is not a win.** It runs only
  `model_format.cyr` and the stdlib, and neither changed. The new Vulkan code
  (10 056 bytes) sits in front of `model_format.cyr`. This is 2.3.28's +7% in
  reverse. 2.3.28 plus a never-called function of that size at the end of
  `vulkan.cyr` (within 8 bytes either way) runs it 6.3% and 5.3% faster; the
  candidate runs it 5.6% faster.
- **`parse_neuron_2dev`, +2.0% (p = 0.05), is below the bar and is layout
  too.** The new code also moves every global and string after `vulkan.cyr`:
  its 8 enum constants take 64 bytes of `.bss`, and its strings add 163 bytes
  of `.rodata`. A code-only pad does not reproduce that move, and measured
  ±0.1%. A second control copied the whole layout: 2.3.28 with the same
  `types.cyr` edit and a dead pad of 8 enum constants, a 163-byte string and
  dead code. Its text size and its `.bss` and `.rodata` addresses match the
  candidate's byte for byte. It runs the row 1.2% slower. The candidate runs
  it 1.3% slower with 2.3.28's bench file, and 1.7% slower as shipped.
- **`parse_vulkan_2gpu`, 0.0%:** on the bench's small hand-written input, the
  new parser costs what the old one did.
- **New rows (2.3.29 only)**, as means of per-layout medians:
  `parse_vulkan_summary_renoir` 14.92 µs, `vulkan_heaps_uhd600` 619 µs. The
  heap scan runs once per detection, only on a host with an integrated non-AMD
  GPU, beside a `vulkaninfo` run of about 20 ms.

In `bench-history.csv`, the `e95322b` rows at 17:09:40Z are 2.3.28, measured
on the clean tree before any change. The `e95322b-dirty` rows at 17:32:15Z
are 2.3.29.

| binary | 2.3.28 | 2.3.29 | Δ |
|---|---:|---:|---:|
| x86_64 ELF, `CYRIUS_DCE=1` | 219 480 | 231 960 | +12 480 (text +10 224) |
| x86_64 ELF, no DCE | 444 760 | 457 240 | +12 480 (text +10 128) |
| ELF-aarch64 | 739 496 | 805 224 | +65 728 (text +11 584) |
| PE, as shipped (no DCE) | 518 656 | 529 920 | +11 264 |
| agnos, `CYRIUS_DCE=1` | 221 240 | 229 624 | +8 384 (text +10 280) |
| Mach-O arm64 (cross-built, as verified) | 721 332 | 721 332 | 0 |

The aarch64 text crossed a 64 KiB boundary, so its data segment moved to the
next 64 KiB page.

### Verified

- **Tests:** 894 assertions in 15 units pass through CI's loop; fuzz 6/6.
- **CI gates:** fmt, lint (0 warnings in `src/`), vet, raw-offset guard,
  `cyrius.lock` current, DCE build. `dist/ai-hwaccel.cyr` differs from
  2.3.28's only in its version line and the three changed modules (`types`,
  `vulkan`, `registry`), and `distlib` is deterministic.
- **Linux CLI output identical to 2.3.28** on 18 invocations covering every
  flag, except the dev host's APU: its Vulkan profile's memory (4 → 8 GiB) and
  the summary totals that follow from it. **ELF-aarch64** under
  `qemu-aarch64` matches x86_64 on 8 invocations. **agnos** builds.
- **The dev host's real AMD APU** (real `vulkaninfo`, real sysfs): the Vulkan
  profile carries the 8 GiB `mem_info_vram_total`, not shared, and the
  no-exec registry matches 2.3.28's.
- **Stand-in `vulkaninfo`** (a script first on `PATH` that logs each run and
  serves the fixtures), through the probe of every entry point:
  - `cass`'s UHD 600: sized from its heap and all shared, as in the table under
    *Fixed*;
  - the UHD 600 plus lavapipe: the lavapipe device is dropped;
  - a discrete NVIDIA: unchanged, with one run per entry point;
  - `registry_detect_no_exec()`: no run in any scenario.
- **Windows on `cass`:** the EXE exactly as `stage_win_cross.sh` builds it
  passes `windows-smoke` (a)–(d). Its JSON and `--summary` match 2.3.28's EXE.
- **macOS on `ecb`** (Apple M5 Pro), cross-built as in 2.3.27: JSON and
  `--summary` match 2.3.28. All five detection paths run and hold the Metal GPU
  and Neural Engine. `--version` reports 2.3.29.
- **Not verified:** a Linux host with an Intel iGPU, or with lavapipe, on real
  hardware. The parser and sizing are tested on `cass`'s real Intel output and
  exercised end to end through the stand-in, but ANV (Mesa's Linux Intel
  driver) was not run.

## [2.3.28] — 2026-09-23 — unified memory counted once in the totals

The registry totals counted system RAM again for every profile that reported
it. On a 48 GB Apple M5 Pro, 2.3.27 reported `total_memory_bytes` as 100 GiB
and `accelerator_memory_bytes` as 52 GiB. 2.3.28 reports 48 GiB for both.
Profiles now say how much of their memory is system RAM, in a new
`shared_memory_bytes` key (schema v6). The Python bindings now expect schema v6;
they had warned on every `detect()` since 2.3.15. On a host without a
shared-memory accelerator the totals do not change: the Linux dev host and
`cass` produce the same JSON as 2.3.27 apart from `schema_version`.

### Fixed

- **System RAM was counted once per profile that reported it.**
  `reg_total_memory` and `reg_total_accel_memory` summed `memory_bytes` over
  every available profile. The CPU profile already counts system RAM, and
  several accelerators report some or all of that same RAM:
  - Apple Silicon's Metal GPU and Neural Engine (unified memory);
  - the client NPUs, which have no memory of their own (Intel NPU, AMD XDNA,
    Samsung NPU, MediaTek APU);
  - GH200, whose 576 GiB includes the 480 GiB of Grace LPDDR5X that the CPU
    also reports.

  Each profile now records its shared part (`shared_mem`, `src/profile.cyr`).
  `profile_new` sets it from the type: all of the memory for the types above
  (`accel_uses_system_memory`, `src/types.cyr`), otherwise 0. The CUDA parser
  sets 480 GiB for a GH200. The totals add each profile's own memory, plus the
  largest shared part once (`src/registry.cyr`); a registry describes one host,
  so the shared pool is a maximum, not a sum. A shared part outside
  `[0, memory_bytes]`, such as a foreign `shared_memory_bytes`, is clamped.
  `--summary` on `ecb` (Apple M5 Pro, 48 GB, `hw.memsize` 51 539 607 552):

  | | 2.3.27 | 2.3.28 |
  |---|---:|---:|
  | `total_memory_bytes` | 107 374 182 400 (100 GiB) | 51 539 607 552 (48 GiB) |
  | `accelerator_memory_bytes` | 55 834 574 848 (52 GiB) | 51 539 607 552 (48 GiB) |

  `registry_detect`, `registry_detect_no_exec`, `registry_detect_threaded`,
  `lazy_into_registry`, and a lazy NPU query followed by `lazy_into_registry`,
  all report these totals on `ecb`. Asahi Linux (Metal GPU and Neural Engine
  from the device tree) takes the same path; it is not verified, as there is no
  such host.
- **The Python bindings warned on every `detect()` since 2.3.15.**
  `SCHEMA_VERSION` in `bindings/python/src/ai_hwaccel/models.py` still said 4
  after the binary moved to schema v5 in 2.3.15. So `detect()` warned "binary
  reports JSON schema v5, these bindings target v4" on every call, and the v5
  fields were dropped as unknown keys. It is now 6. `AcceleratorProfile` gains
  the v5 fields (`accel_type_id`, `mem_bandwidth_x1000`,
  `pcie_bandwidth_x1000`, `power_x1000`, `tpu_version`, `tpu_chips`,
  `gaudi_gen`, `neuron_chip`, `neuron_cores`), `shared_memory_bytes`, and a
  `dedicated_memory_bytes` property (0 for the CPU).

### Changed

- **Schema v6: `shared_memory_bytes`.** A non-CPU profile whose memory is partly
  or wholly system RAM carries the shared part, in bytes. The key is omitted on
  the CPU (all of its memory is system RAM) and on devices that share nothing,
  so their JSON is unchanged. `profile_from_json` reads the key. Without the
  key, it keeps what `profile_new` derives from the type, so a v5 document
  still parses. `SCHEMA_VERSION` is 6 (`src/units.cyr`).
- **`compatible_with_registry` budgets against the corrected accelerator
  total.** On `ecb` that is 48 GiB, not 52 GiB. So a model that needs 50.3 GiB
  (45B at INT8, with the 20% activation overhead) no longer fits, and it did in
  2.3.27. The CLI does not call this function; library consumers do.
- **New functions and fields:** the profile's 21st field `shared_mem` (derived
  accessors `profile_shared_mem` / `profile_set_shared_mem`; `PROFILE_SIZE`
  168), `profile_shared_bytes` (the clamped shared part),
  `accel_uses_system_memory`, and `ACCEL_SYSMEM_MASK`, which is the same set as
  a bitmask so that `profile_new` makes no call per device.
- **Docs.** `docs/guides/production.md`'s version-compatibility section said
  `schema_version` was 1, and its example was Rust; it now describes the bump
  rule with a Cyrius example. The bindings README describes schema v6 and the
  totals. `docs/guides/testing.md` had the counts and test layout of an older
  tree. The README profile listing and test counts are updated.
- **Roadmap.** The unified-memory item is closed. One item is new:
  `docs/schema.json` still describes the Rust-era v1 output, so no current
  output validates against it.
- **Tests: 746 → 791 assertions in 15 units.**
  - `foundation_test` 122 → 125: schema v6; `accel_uses_system_memory` for
    every type; `ACCEL_SYSMEM_MASK` agrees with it.
  - `profile_test` 91 → 103: the shared part per type, the clamps, and type
    ids outside the enum (`profile_from_json` accepts any integer).
  - `registry_test` 94 → 107: totals for `ecb`'s profiles, a GH200, an NPU
    laptop, a discrete GPU next to an NPU, no CPU profile, unavailable
    profiles, and out-of-range shared parts.
  - `gpu_parser_test` 40 → 44: the GH200 path of `parse_cuda_output` (its
    first test) marks the 480 GiB shared; an A100 shares nothing.
  - `json_output_test` 40 → 45: the key on Metal but not on the CPU or CUDA,
    and `ecb`'s summary totals.
  - `json_roundtrip_test` 28 → 32: the shared part survives the round trip.
  - `backend_test` 58 → 60 and `model_catalog_test` 6 → 8:
    `apple_emit_silicon`'s profiles are shared, and `compatible_with_registry`
    on `ecb`'s profiles.

  Each of 25 single-point mutations of the final code fails at least one
  assertion:
  - 12 in the two totals;
  - 3 in `profile_new`'s type check;
  - 9 across the clamp, the JSON key, the GH200 path, the type table and mask,
    and the schema version;
  - 1 in `compatible_with_registry`'s budget.

### Known, not changed here

- **Integrated GPUs seen through Vulkan are still counted as memory of their
  own.** `vulkaninfo --summary` gives no heap size, so such a GPU gets the
  4 GiB estimate. Whether that memory is part of the CPU's RAM depends on the
  GPU: an AMD APU's BIOS carve-out is not in `MemTotal`, while an Intel iGPU
  uses system RAM. Telling them apart needs `deviceType` and `vendorID` from
  `vulkaninfo`. It is on the roadmap next to the duplicate-device item, which
  is how the Linux dev host's one AMD iGPU still counts twice (ROCm 8 GiB +
  Vulkan 4 GiB). Windows is not affected: DXGI profiles report dedicated video
  memory, never `SharedSystemMemory`.

### Performance

**13 neutral, 2 slower, both explained below** (2.3.27 → 2.3.28). Method as in
2.3.27: each arm built at 5 code layouts, the three floor-bound rows
batch-timed (†), two independent 30-round passes, shuffled and pinned to one
CPU; the table combines both (60 rounds). The host was running other work, so
absolute times are higher than in 2.3.27's table; the arms were interleaved,
so the deltas hold.

| row | 2.3.27 | 2.3.28 | Δ | p | verdict |
|---|---:|---:|---:|---:|---|
| `parse_cuda_8gpu` | 14.24 µs | 14.31 µs | +0.5% | 0.20 | neutral |
| `parse_vulkan_2gpu` | 2.76 µs | 2.79 µs | +1.0% | 0.03 | neutral |
| `parse_neuron_2dev` | 1.66 µs | 1.65 µs | −1.0% | 0.13 | neutral |
| `detect_safetensors` † | 412.6 ns | 441.4 ns | +7.0% | <0.01 | slower: code placement |
| `detect_gguf` † | 39.2 ns | 39.5 ns | +0.9% | 0.02 | neutral |
| `best_available_13dev` † | 179.4 ns | 179.0 ns | −0.2% | 0.70 | neutral |
| `total_memory_13dev` | 89.0 ns | 108.0 ns | +21.3% | <0.01 | slower: the fix |
| `has_accelerator_13dev` | 19.1 ns | 19.1 ns | +0.1% | 0.31 | neutral |
| `plan_70B_bf16_4gpu` | 1.50 µs | 1.46 µs | −2.8% | 0.04 | neutral |
| `count_family_gpu_13dev` | 255.0 ns | 255.0 ns | 0.0% | 1.00 | neutral |
| `json_serialize_13dev` | 23.74 µs | 23.96 µs | +0.9% | <0.01 | neutral |
| `json_summary_13dev` | 3.27 µs | 3.29 µs | +0.7% | 0.03 | neutral |
| `json_system_io` | 5.36 µs | 5.34 µs | −0.3% | 0.09 | neutral |
| `json_plan` | 17.66 µs | 17.83 µs | +1.0% | <0.01 | neutral |
| `json_training` | 2.57 µs | 2.56 µs | −0.2% | 0.49 | neutral |

A verdict needs p < 0.01 and |Δ| > 1%; `json_plan` is +0.95%.

- **`total_memory_13dev`, +21.3% (+19 ns over 13 profiles): the fix.** The
  function now reads each profile's shared part and folds shared RAM into one
  pool. The first version called `profile_shared_bytes` per profile and cost
  +70%. Inlining the clamp brought that to +31.5%. A path for devices that
  share nothing brought it to +21%: such a device pays one load and one
  compare, then adds its memory as before. That covers 11 of the bench's 13
  profiles. The remaining cost is that load, plus the clamp on the two shared
  profiles (the CPU and the Intel NPU). `json_summary_13dev` calls this
  function and `reg_total_accel_memory`; it went from +3.5% to +0.7%.
- **`detect_safetensors`, +7.0%: code placement, not the change.** It runs
  only `model_format.cyr` and the stdlib, and neither changed. The parsing
  suite includes `types`, `profile` and `cuda`, whose new code (1 016 bytes)
  sits in front of `model_format.cyr`. The layout pads live in `log.cyr`, so
  they move `model_format.cyr` and the bench driver together and do not
  reproduce that shift. Two further arms settled it. Each was 2.3.27's code
  plus a never-called function at the end of `profile.cyr`, sized to within
  8 and 24 bytes of 2.3.28's text:

  | arm (mean of per-layout medians, 60 rounds) | `detect_safetensors` |
  |---|---:|
  | 2.3.27 | 408.0 ns |
  | 2.3.27 + dead pad (within 8 bytes) | 432.9 ns (+6.1%) |
  | 2.3.27 + dead pad (within 24 bytes) | 435.0 ns (+6.6%) |
  | 2.3.28 | 434.9 ns (+6.6%) |

  So 2.3.27's own code is as slow as 2.3.28 once it is laid out like 2.3.28.
  The heap is not the cause either: with the heap aligned to 2 MiB before the
  bench's buffer, the delta was +6.5%. `detect_gguf`, in the same file, is
  unaffected (38.9 / 38.9 / 39.1 / 39.1 ns).
- `json_serialize_13dev` is +0.9% because the output grew: the bench's Intel
  NPU now emits `"shared_memory_bytes":4294967296` (33 more bytes per call).
  `json_plan` runs none of the changed code (`plan_to_json`, `_shard_to_json`);
  it moved behind the larger `profile_to_json`.

`bench-history.csv` holds a single-run pair measured back to back on the loaded
host: `bf70273-dirty` at 16:45:12Z is 2.3.28, and `bf70273` at 16:45:40Z is
2.3.27, measured from `git archive bf70273`.

| binary | 2.3.27 | 2.3.28 | Δ |
|---|---:|---:|---:|
| x86_64 ELF, `CYRIUS_DCE=1` | 219 448 | 219 480 | +32 (text +1 336) |
| x86_64 ELF, no DCE | 444 728 | 444 760 | +32 (text +1 960) |
| ELF-aarch64 | 739 464 | 739 496 | +32 |
| PE, as shipped (no DCE) | 516 608 | 518 656 | +2 048 |
| agnos, `CYRIUS_DCE=1` | 217 120 | 221 240 | +4 120 |
| Mach-O arm64 (cross-built, as verified) | 721 332 | 721 332 | 0 |

The ELF files are page-padded, so they show only the data growth. The agnos
text crossed 128 KiB, so its data segment moved to the next page.

### Verified

- **Tests:** 791 assertions in 15 units pass through CI's loop; fuzz 6/6.
- **CI gates:** fmt, lint (0 warnings), vet, raw-offset guard, `cyrius.lock`
  current, DCE build. `dist/ai-hwaccel.cyr` differs from 2.3.27's only in its
  version line and the six changed modules (`units`, `types`, `profile`,
  `cuda`, `registry`, `json_out`), and `distlib` is deterministic.
- **Linux CLI output identical to 2.3.27** on 18 invocations covering every
  flag, apart from `schema_version` and the version string. **ELF-aarch64**
  under `qemu-aarch64` matches x86_64 on 8 invocations. **agnos** builds.
- **Python bindings:** 24 tests pass, including the 5 that need a staged
  binary. `detect()` against the 2.3.28 binary raises no warning; they ran
  with `UserWarning` as an error.
- **macOS on `ecb`** (Apple M5 Pro, macOS 27.0), cross-built as in 2.3.27:
  - the totals above, from all five detection paths;
  - the JSON matches 2.3.27 apart from `schema_version` and the new key, which
    is on the Metal GPU (51 539 607 552) and the Neural Engine (4 294 967 296);
  - stderr is silent by default and carries the `detect` span with `-vv`;
  - `--version` reports 2.3.28;
  - the Python bindings under the system Python 3.9.6 raise no schema warning
    and report `dedicated_memory_bytes` 0 for all three profiles.
- **Windows on `cass`:** the EXE exactly as `stage_win_cross.sh` builds it
  passes `windows-smoke` (a)–(d). Its JSON and `--summary` match 2.3.27's EXE
  apart from `schema_version`. The Intel UHD 600 reports its 128 MiB of
  dedicated memory, so nothing on `cass` is shared.
- **Not verified:** GH200 and the client NPUs on real hardware (the parser and
  the totals are covered by tests), and Asahi Linux.

## [2.3.27] — 2026-09-23 — macOS: real total RAM, and Apple Silicon without system_profiler

Every macOS build so far reported the CPU at the 16 GiB fallback. It now
reports the machine's real RAM, read from sysctl `hw.memsize`. Apple Silicon's
Metal GPU and Neural Engine now come from sysctl too, not from spawning
`system_profiler`. As a result, `registry_detect_no_exec()` finds them on a Mac
(it returned the CPU alone before), and detection on the M5 Pro test host drops
from 83 ms to 3 ms. Linux and Windows output is unchanged. The roadmap was
reorganized.

### Fixed

- **macOS reported the CPU at the 16 GiB fallback.** `detect_system_memory`
  had no macOS branch. It read `/proc/meminfo`, which macOS does not have, got
  0, and the registry substituted 16 GiB. The tagged 2.3.6 and 2.3.24 sources
  do the same, so every macOS wheel since the first had it. It now asks sysctl
  for `hw.memsize` (`mac_total_phys_bytes`, in `src/detect/platform.cyr`). The
  call is the one `lib/sys.cyr`'s `sys_sysinfo` makes: a two-int MIB
  `{CTL_HW, HW_MEMSIZE}` read at width 8, through the syscalls peer's
  `SYS_SYSCTL` alias. The build already includes that peer, so neither
  ai-hwaccel nor its library consumers gain a stdlib dependency. On `ecb`
  (Apple M5 Pro, 48 GB, `hw.memsize` 51 539 607 552), 2.3.26 reports
  17 179 869 184 from all four detection entry points, and 2.3.27 reports
  51 539 607 552 from all four. That is the macOS twin of the Windows RAM bug
  fixed in 2.3.25.
- **`registry_detect_no_exec()` on a Mac returned the CPU alone.** The Apple
  backend found the Metal GPU and Neural Engine by spawning `system_profiler`,
  so it was classed exec and the no-exec mask dropped it. It now reads the
  same two facts natively (`detect_apple_opts`, `src/detect/apple.cyr`):
  `machdep.cpu.brand_string` names the chip (resolved through sysctl's
  name-to-OID lookup, as `lib/sys.cyr` does), and `hw.memsize` sizes its
  unified memory. `system_profiler` stays as the fallback for when sysctl
  cannot name an Apple chip, and runs only when exec is allowed. The Asahi
  Linux device-tree check needed no subprocess and now runs in no-exec mode
  too. `BACKEND_APPLE` is no longer classed exec. On `ecb` the no-exec registry
  now holds 3 profiles (CPU, Metal GPU, Neural Engine), where it held 1.

### Changed

- **Apple Silicon detection reads sysctl, not `system_profiler`.** The profiles
  are unchanged. The GPU is named after `machdep.cpu.brand_string` ("Apple M5
  Pro", the string `system_profiler` prints as "Chip:") and sized by
  `hw.memsize` ("Memory: 48 GB"). The Neural Engine estimate uses the same
  per-generation table, now `_apple_ane_bytes`. On `ecb` the CLI's JSON is
  byte-identical to 2.3.26 apart from the CPU `memory_bytes`. Detection time
  drops from a median of 83.0 ms to 3.2 ms (15 interleaved runs), because
  `system_profiler` was the one tool that actually ran there.
- **New functions:** `mac_total_phys_bytes` (macOS only), `detect_apple_opts`,
  `apple_is_silicon_brand` and `apple_emit_silicon`. `detect_apple` keeps its
  signature and allows exec.
- **Docs describe native Apple detection.** Updated: the README backend table
  and module tree, `docs/architecture/overview.md`, `docs/guides/testing.md`
  and `docs/performance.md`. The README's no-exec section was stale since
  2.3.25 and is corrected: no-exec mode masks seven exec backends, not eight,
  and runs ten native ones, Windows and Apple included.
- **Tests: 725 → 746 assertions in 15 units.**
  - `backend_test` 37 → 58: the brand test (an Intel Mac and a VM's "Apple M1
    (Virtual)" included), what the native path emits for `ecb`, the Neural
    Engine table, the native path matching the `system_profiler` parser field
    for field, and no `system_profiler` attempt without exec.
  - `registry_test`'s exec classification now has Apple on the native side.
  - The stub units (`lazy_test`, `json_output_test`, `planning_test`,
    `model_catalog_test`) stub `detect_apple_opts`, the function the registry
    calls.

  Each of 5 single-point mutations of the Apple code fails at least one
  assertion.
- **`docs/development/roadmap.md` reorganized** (871 → about 310 lines). It now
  holds open work only, planned by release: 2.3.27 (macOS total RAM), 2.4.x
  (correct output + CI coverage), 2.5.x (platform validation), 2.6.x
  (multi-node and hot-plug, previously 2.4.0), 2.7.x (fleet and scale,
  previously 2.5.0), and a *Later* list. Shipped history is one table; this
  file keeps the details. Every open item was either carried forward or closed.
  Seven were closed as done, obsolete or moot, with the reasons recorded in the
  roadmap's *Closed in the 2026-09-23 review*. The toolchain-blocked items were
  re-tested on cyrius 6.6.6: `cyrius capacity --check` now passes and moves to
  2.4.x; `case` labels still reject enum names; `--target js` cannot build
  `src/`.
- **New roadmap items found during the review:**
  - macOS wheels have always reported the CPU at the 16 GiB fallback (fixed
    in this release, below).
  - The Linux dev host's one AMD iGPU is reported twice (ROCm 8 GiB + Vulkan
    4 GiB).
  - Lazy family queries still spawn `nvidia-smi` through the interconnect
    post-pass.
  - `registry_detect_with(builder_no_exec())` is documented as spawn-free but
    is not.

### Known, not changed here

- **Unified memory is counted twice in `total_memory_bytes`.** On Apple
  Silicon the CPU and the Metal GPU profiles describe the same RAM, and the
  total sums every profile. Now that the CPU is right, `ecb`'s total reads
  100 GiB (48 CPU + 48 GPU + 4 Neural Engine) for 48 GB of RAM. With the old
  fallback it read 68 GiB (16 + 48 + 4). Asahi Linux already summed it this
  way. Added to the 2.4.x roadmap, next to the duplicate-device item.
- **CI still never runs a macOS binary.** Everything above was verified on
  `ecb` with a cross-built binary (next section); the `macos-smoke` job is on
  the 2.4.x roadmap.

### Performance

**15 neutral, 0 regressions** (2.3.26 → 2.3.27). `apple.cyr`, `types.cyr` and
`registry.cyr` are in both bench suites, so this used the layout-controlled
A/B. Each arm was built at 5 code layouts, the three floor-bound rows were
batch-timed (†), and there were two independent 30-round passes, shuffled and
pinned to one CPU. The table combines both passes (60 rounds).

| row | 2.3.26 | 2.3.27 | Δ | p | verdict |
|---|---:|---:|---:|---:|---|
| `parse_cuda_8gpu` | 13.38 µs | 13.38 µs | 0.0% | 0.96 | neutral |
| `parse_vulkan_2gpu` | 2.47 µs | 2.49 µs | +0.8% | 0.18 | neutral |
| `parse_neuron_2dev` | 1.58 µs | 1.57 µs | −0.2% | 0.79 | neutral |
| `detect_safetensors` † | 374.4 ns | 379.0 ns | +1.2% | 0.21 | neutral |
| `detect_gguf` † | 36.4 ns | 36.4 ns | 0.0% | 0.92 | neutral |
| `best_available_13dev` † | 168.5 ns | 167.9 ns | −0.4% | 0.40 | neutral |
| `total_memory_13dev` | 81.0 ns | 81.4 ns | +0.5% | 0.05 | neutral |
| `has_accelerator_13dev` | 17.8 ns | 17.8 ns | 0.0% | 0.94 | neutral |
| `plan_70B_bf16_4gpu` | 1.41 µs | 1.45 µs | +2.8% | 0.04 | neutral (layout) |
| `count_family_gpu_13dev` | 233.1 ns | 233.3 ns | +0.1% | 0.55 | neutral |
| `json_serialize_13dev` | 21.67 µs | 21.70 µs | +0.1% | 0.38 | neutral |
| `json_summary_13dev` | 3.05 µs | 3.03 µs | −0.5% | 0.13 | neutral |
| `json_system_io` | 4.93 µs | 4.93 µs | 0.0% | 0.88 | neutral |
| `json_plan` | 16.27 µs | 16.16 µs | −0.7% | <0.01 | neutral |
| `json_training` | 2.37 µs | 2.38 µs | +0.2% | 0.48 | neutral |

A verdict needs p < 0.01 and |Δ| > 1%. `plan_70B_bf16_4gpu` meets neither, and
its +2.8% comes from two of the five layouts (+5.3% and +6.1%); in the other
three, 2.3.27 is within +0.6% to +1.4% of 2.3.26. The bench calls only
`reg_plan_sharding`, which runs none of the changed code, so this is code
placement. `bench-history.csv` rows `6b5af9b-dirty` at 15:21Z are 2.3.26
(`src/` equal to `HEAD`, only docs dirty); those at 15:33Z are 2.3.27.

| binary | 2.3.26 | 2.3.27 | Δ |
|---|---:|---:|---:|
| x86_64 ELF, `CYRIUS_DCE=1` | 219 448 | 219 448 | 0 (contents differ) |
| x86_64 ELF, no DCE | 444 728 | 444 728 | 0 (contents differ) |
| ELF-aarch64 | 739 456 | 739 464 | +8 |
| PE, as shipped (no DCE) | 516 096 | 516 608 | +512 |
| agnos, `CYRIUS_DCE=1` | 217 112 | 217 120 | +8 |

### Verified

- **Tests:** 746 assertions in 15 units pass through CI's loop; fuzz 6/6.
- **CI gates:** fmt, lint (0 warnings), vet, raw-offset guard, DCE build.
  `dist/ai-hwaccel.cyr` differs from 2.3.26's only in its version line and the
  four changed modules.
- **Linux CLI output identical to 2.3.26** on 18 invocations covering every
  flag. **ELF-aarch64** under `qemu-aarch64` matches x86_64 on 8 invocations.
  **agnos** builds.
- **macOS on `ecb`** (Apple M5 Pro, macOS 27.0), with binaries cross-built by
  the pinned toolchain's `cycc_aarch64` (`CYRIUS_MACHO_ARM=1`) from the input
  `stage_binary.sh` composes on macOS, then ad-hoc signed:
  - all four entry points report `hw.memsize` exactly;
  - `registry_detect_no_exec()` holds the Metal GPU and Neural Engine;
  - the CLI's JSON matches 2.3.26 apart from the CPU memory;
  - stdout carries the JSON, and stderr is silent by default and carries the
    `detect` span with `-vv`;
  - `AI_HWACCEL_DATA_DIR` and `--data-dir` resolve `VERSION` from `/` (open
    since 2.3.21);
  - `registry_detect_threaded()` runs and serializes (its first run on Apple
    Silicon).
- **Windows on `cass`:** the EXE exactly as `stage_win_cross.sh` builds it
  passes `windows-smoke` (a)–(d).
- **Not verified:** the wheel's own `macos-14` build (a native build there,
  never run in CI), Intel Macs (not a shipped target; they fall back to
  `system_profiler` as before), and Asahi Linux.

## [2.3.26] — 2026-09-23 — lazy NPU queries find the Apple Neural Engine

A lazy NPU query on a fresh `lazy` registry missed the Apple Neural Engine
unless GPUs had been queried first. Fixed, and verified on an Apple M5 Pro.
Nothing else changes. The CLI does not use the lazy registry, so its output is
identical to 2.3.25, and the x86_64 Linux wheel binary is byte-identical.

### Fixed

- **`lazy_by_family(lr, FAMILY_NPU)` missed the Apple Neural Engine.**
  `detect_apple` is the one detector whose profiles span two families: it
  emits the Metal GPU (GPU family) and the Neural Engine (NPU family). The lazy
  registry runs a backend only for the families whose mask holds it, and Apple
  was in the GPU mask only. An NPU query on a fresh registry never ran it, so
  the Neural Engine showed up only if GPUs had been queried first. Apple is now
  in both the GPU and NPU masks. The lazy registry also records which
  *backends* it has run (a new `backends` field on the `lazy` struct), so the
  second family's query does not run Apple again and push its profiles twice.

  Verified on `ecb` (Apple M5 Pro, macOS 27.0) with a probe built by the
  pinned 6.6.6 toolchain's Linux-hosted Mach-O cross compiler (`cycc_aarch64`
  with `CYRIUS_MACHO_ARM=1`, from the same compiler input
  `stage_binary.sh` composes on macOS). For an NPU query on a fresh lazy
  registry, 2.3.25 returns `count=0` and 2.3.26 returns the Apple Neural
  Engine. The GPU query that follows returns the Metal GPU. The registry then
  holds 3 profiles (CPU, Metal GPU, Neural Engine), the same as
  `registry_detect()`.

### Changed

- **New test unit `tests/tcyr/lazy_test.tcyr` (35 assertions).** Its stub
  detectors each emit their backend's accelerator type and count their calls;
  `detect_apple` emits both of its profiles. The unit checks that, for every
  family, a lazy query on a fresh registry returns the same profiles by type as
  full detection filtered to that family. It also checks that no backend runs
  twice in either query order, and that a TPU query starts no other backend.
  The family-mask test moved here from `registry_test` (108 → 94 assertions),
  updated for Apple being in two masks. **725 assertions in 15 units.** Run
  against the 2.3.25 `lazy.cyr`, the unit fails 5 assertions; each of 4
  single-point mutations of the fix fails at least one.
- **The `lazy` struct has 4 fields** (`backends` added; 32 bytes), with the
  derived `lazy_backends` / `lazy_set_backends` accessors.
- **README's test-unit table lists all 15 units.** It had 11 rows and a stale
  "13 units, 623 assertions" count.

### Performance

- **Benchmarks: 15 neutral by construction, 0 regressions.** Neither bench
  suite includes `src/lazy.cyr`, so `benches/parsing` and `benches/registry`
  build byte-identical to 2.3.25, plain and with `CYRIUS_DCE=1`
  (sha256-compared). `bench-history.csv` rows: `5a72769` (2.3.25) and
  `5a72769-dirty` (2.3.26).

| binary | 2.3.25 | 2.3.26 | Δ |
|---|---:|---:|---:|
| x86_64 ELF, `CYRIUS_DCE=1` | 219 448 | 219 448 | 0 (byte-identical) |
| x86_64 ELF, no DCE | 444 728 | 444 728 | 0 (padding) |
| ELF-aarch64 | 739 456 | 739 456 | 0 (padding) |
| PE, as shipped (no DCE) | 515 584 | 516 096 | +512 |
| agnos, `CYRIUS_DCE=1` | 217 112 | 217 112 | 0 (byte-identical) |

The DCE builds drop the lazy registry entirely: the CLI never calls it.

### Verified

- **Tests:** 725 assertions in 15 units pass through CI's loop; fuzz 6/6.
- **CI gates:** fmt, lint (0 warnings), vet, raw-offset guard, DCE build.
  `dist/ai-hwaccel.cyr` differs from 2.3.25's only in its version line and
  `lazy.cyr`.
- **Linux CLI output identical to 2.3.25** on 18 invocations covering every
  flag.
- **macOS on `ecb`:** the cross-built 2.3.26 CLI reports the CPU, the Metal GPU
  (Apple M5 Pro) and the Neural Engine. stdout carries the JSON; stderr stays
  silent at the default level and carries the `detect` span with `-vv`. This
  binary is a Linux cross-build, not the wheel's native `macos-14` build, and
  that job still does not run its binary.
- **Windows on `cass`:** the EXE exactly as `stage_win_cross.sh` builds it
  passes `windows-smoke` (a)–(d), with `--version` reporting 2.3.26.

## [2.3.25] — 2026-09-23 — Windows detection without wmic; Windows GPUs on every entry point

Windows detection moves off `wmic`, which Windows 11 24H2+ removes: GPUs now come
from DXGI and total RAM from `GlobalMemoryStatusEx`. Windows GPUs are detected
by every detection entry point; `registry_detect_no_exec()`,
`registry_detect_threaded()` and lazy GPU queries used to skip them. The
threaded and lazy entry points no longer return a corrupted registry. The macOS
wheel job that failed on the 2.3.24 tag builds again. Linux CLI output is
unchanged, and all 15 benchmarks are neutral.

### Fixed

- **The macOS wheel job failed on the 2.3.24 tag** with 46 × `cannot hash the
  pinned snapshot (sha256sum missing?)`. `cyrius build` checks and rewrites
  `cyrius.lock` with `sha256sum`, which it cannot reach on the macOS runner, and
  since 2.3.24 the lock records the toolchain pin, which turns that check on.
  On macOS, `bindings/python/scripts/stage_binary.sh` now runs the build as its
  two halves. First, `cyrius deps --no-lock` vendors `./lib/` with the lock set
  aside; the lock is restored on exit, and CI's Linux job still verifies it.
  Second, the pinned `cycc` compiles the same input `cyrius build` composes.
  That input was checked byte-identical to `cyrius build src/main.cyr` output.
  Linux and the aarch64 cross-build keep using `cyrius build` and stage the same
  binaries as before. Verified by reproducing the runner's exact failure (no
  hasher reachable, `uname -s` = Darwin): 46 refusals before, a clean build
  after. A failed compile still restores the lock. All 47 stdlib hashes in the
  lock match the macOS 6.6.6 release's snapshot.
- **Windows detection no longer needs `wmic`.** Windows 11 24H2+ removes
  `wmic` by default. The Windows backend used it to create its GPU profiles,
  and memory detection used it to read total RAM, so on such a host the
  Windows wheel reported a CPU profile only, with the 16 GiB fallback. On
  `cass` (Windows 11 10.0.26200, 8 GB, Intel UHD Graphics 600) 2.3.23 and
  2.3.24 printed `"memory_bytes":17179869184`, no GPU, and a `"wmic"` warning.
  Now:
  - **GPUs come from DXGI** (`src/detect/windows.cyr`). `CreateDXGIFactory1` →
    `EnumAdapters1(i)` until `DXGI_ERROR_NOT_FOUND` → `GetDesc1` gives one
    `Windows GPU` profile per hardware adapter, named from the descriptor's
    UTF-16 `Description` and sized from `DedicatedVideoMemory`
    (`DedicatedSystemMemory` when a driver reports its carve-out there instead).
    Software adapters are skipped: `DXGI_ADAPTER_FLAG_SOFTWARE`, or the
    Microsoft Basic Render Driver's `1414:008C` IDs as a second check that
    does not rely on the flag. One factory serves the whole enumeration; the
    old enrichment pass created one per adapter. An adapter that fails
    `EnumAdapters1` or `GetDesc1` is reported as a `dxgi` warning and
    enumeration carries on to the next one, so one bad adapter cannot hide the
    rest.
  - **Total RAM comes from `kernel32!GlobalMemoryStatusEx`**
    (`src/detect/platform.cyr`). The PE backend has no reroute for it, so it is
    resolved at run time through the `GetModuleHandleA`/`GetProcAddress`
    reroutes (`syscall` `0xF013`/`0xF014`) and called with `callptr`.
    `ullTotalPhys` on `cass` is 8 405 598 208, exactly
    `Win32_ComputerSystem.TotalPhysicalMemory`.
  - **`wmic` is only a fallback now, and only where exec is allowed.** For
    GPUs it runs when DXGI itself is unavailable (no factory, or the first
    `EnumAdapters1` fails with anything but `DXGI_ERROR_NOT_FOUND`), which is
    always reported as a `dxgi` warning. For RAM it runs if
    `GlobalMemoryStatusEx` fails. A host where DXGI works but lists only
    software adapters has no GPU, and `wmic` is not consulted.

  On `cass` the fixed EXE reports `"memory_bytes":8405598208`, a `Windows GPU`
  profile for the Intel(R) UHD Graphics 600, and no `"wmic"` warning.
  **Behaviour change on hosts that still have `wmic`:** an integrated GPU now
  reports its dedicated carve-out, not the driver's `AdapterRAM`. When `cass`
  still had `wmic`, 2.3.9 reported the UHD 600 at 1 GiB; DXGI reports its
  128 MiB `DedicatedVideoMemory`. A discrete card already reported the larger
  of the two figures, which is DXGI's once VRAM passes `wmic`'s 4 GiB cap.
- **Windows GPUs were skipped by three of the four detection entry points.**
  Only `registry_detect()` (the CLI's path) called the Windows detector.
  - `registry_detect_no_exec()`: `BACKEND_WINDOWS` was classed exec, so the
    no-exec mask dropped DXGI together with its `wmic` fallback. DXGI spawns
    nothing, so the backend is native now. `detect_windows_opts(profiles,
    warnings, allow_exec)` always runs DXGI and runs the `wmic` fallback only
    when `allow_exec` is 1. RAM works the same way through
    `detect_system_memory_opts(allow_exec)`, so on Windows
    `registry_detect_no_exec()` never starts a process.
  - `registry_detect_threaded()` never called the Windows detector at all. It
    now runs it on the main thread while the CLI-tool threads run.
  - `lazy_by_family(lr, FAMILY_GPU)` probed CUDA, ROCm, Vulkan and Apple only.
    Windows was missing, and so was Intel oneAPI, on every platform. oneAPI
    emits GPU-family profiles but sat in the AI_ASIC probe mask, so its GPUs
    appeared in a lazy GPU query only if the AI_ASIC family had already been
    probed. Both are in the GPU mask now, and every backend with a detector
    sits in exactly one family mask, so probing all families never runs a
    backend twice.

  Verified on `cass` with a probe that calls each entry point: all four report
  the UHD 600 and the exact RAM, and the no-exec registry carries no warnings
  at all, because no tool was spawned.
- **`registry_detect_threaded()` and `lazy_into_registry()` returned a
  corrupted registry, on every platform.** Their post-passes handed
  `detect_storage`, `detect_interconnects` and `detect_environment` the
  registry instead of its `system_io`. So interconnects were pushed into
  `profiles`, storage entries into `warnings`, and the environment struct
  replaced `system_io`. `registry_to_json` on either result segfaulted (exit
  139 on the Linux dev host, with the 2.3.24 tree as well). The summary JSON
  never reads those fields, which kept it hidden, and the CLI uses
  `registry_detect()`, which was correct. All three entry points now share one
  `registry_post_passes(r, allow_exec)`, which always passes
  `reg_system_io(r)`.
- **Both `wmic` spawns wrote one byte past their buffer** on an exactly-full
  read: `alloc(N)`, `exec_capture(…, N)`, then `store8(buf + n, 0)`. That is the
  same pattern 2.3.22 fixed in `load_models`. Each now reads at most `N - 1`
  bytes.

### Changed

- **`windows-smoke` checks what detection found, not just that it ran**
  (`wheels.yml`, new step (d)). Steps (a)–(c) passed for 2.3.23/2.3.24 on a
  wmic-less host, because a CPU profile is always present. Step (d) compares
  the EXE with Windows' own CIM view, which `Get-CimInstance` reads without
  `wmic.exe`:
  - the CPU profile's `memory_bytes` must not be the 16 GiB fallback
    (17 179 869 184) unless Windows reports exactly that, and must be within 1%
    of `Win32_ComputerSystem.TotalPhysicalMemory`. On `cass` the two are equal
    to the byte. The 1% tolerance is there because that equality was only
    checked on Windows 11, not on the runner's Windows Server.
  - if Windows lists a PCI display adapter on a vendor driver package, at
    least one `GPU`-family profile must be reported. Vendor packages are the
    ones published as `oemNN.inf` (on `cass`, Intel's `iigd_dch.inf` is
    `oem67.inf`), so the inbox Basic Display driver and a VM's synthetic or
    emulated video adapters are not counted. On a GPU-less runner this half has
    nothing to check; it binds on real hardware.

  The step runs under Windows PowerShell 5.1 (`shell: powershell`), the engine
  it was verified with: `cass` has no `pwsh`. Its body was extracted from the
  YAML and run on `cass` inside a reproduction of GitHub's `shell: powershell`
  wrapper. It passes the 2.3.25 EXE and fails the 2.3.24 EXE at the fallback
  check. With the EXE's output swapped for fixtures, it fails a result with no
  GPU profile and one with RAM more than 1% off, and passes RAM 4 KiB short.
- **`BACKEND_WINDOWS` is no longer classed exec** (`backend_uses_exec` → 0), so
  `builder_no_exec()` includes it. New functions: `detect_windows_opts`,
  `detect_system_memory_opts` and `registry_post_passes`. `detect_windows` and
  `detect_system_memory` keep their signatures and allow exec.
- **Tests: 629 → 704 assertions in 14 units.**
  - `windows_test` 25 → 57: DXGI descriptor fixtures (including the two
    adapters `cass` enumerates, with the field values a probe read there), the
    software-adapter and memory rules, device-id order across a hybrid +
    software list, UTF-16 names (surrogate pairs, lone surrogates → U+FFFD, an
    unterminated 128-WCHAR `Description`), an empty name, and the `wmic`
    fallback rule.
  - `registry_test` 69 → 108: the exec classification, no-exec memory, the lazy
    family masks (disjoint, covering every detector), and all four detection
    entry points run end to end on the host, including serializing each
    registry. No test called a detection entry point before, which is how the
    threaded and lazy corruption went unnoticed.
  - `json_output_test` 36 → 40: `registry_post_passes` hands each system-I/O
    pass `system_io`, and skips interconnects without exec.

  17 single-point mutations of the new code, including re-introducing the
  original post-pass bug in each path, each fail at least one assertion.

### Removed

- **`win_merge_vram`, `win_enrich_dxgi_vram`, `win_dxgi_adapter_vram_bytes`.**
  They reconciled wmic's `AdapterRAM` with DXGI's figure; with DXGI creating
  the profiles there is nothing to reconcile. `win_merge_vram` was ungated and
  shipped in `dist/ai-hwaccel.cyr`; nothing in this repo calls it any more. New
  in its place: `win_dxgi_desc_is_software`, `win_dxgi_desc_vram`,
  `win_dxgi_desc_emit`, `win_utf16_to_cstr` and `win_gpu_use_wmic_fallback`
  (ungated, pure), plus `win_dxgi_enum_adapters` and `win_total_phys_bytes`
  (PE only).

### Performance

**15 neutral, 0 regressions** (2.3.24 → 2.3.25). This release changes code the
benchmarks include (`types.cyr`, `platform.cyr`, `registry.cyr`), so each arm
was measured at 5 code layouts (a never-called function of 0/5/11/17/23
statements before the first function in `src/log.cyr`). The three floor-bound
rows were re-timed in batch windows of 20 000 ops (†). The 20 binaries were run
in 30 rounds, shuffled each round and pinned to one CPU. The toolchain and
`lib/bench.cyr` are the same in both arms.

| row | 2.3.24 | 2.3.25 | Δ | p | verdict | layout spread (2.3.24 / 2.3.25) |
|---|---:|---:|---:|---:|---|---:|
| `parse_cuda_8gpu` | 13.62 µs | 13.63 µs | +0.1% | 0.83 | neutral | 1.3% / 1.3% |
| `parse_vulkan_2gpu` | 2.55 µs | 2.56 µs | +0.2% | 0.73 | neutral | 1.6% / 2.4% |
| `parse_neuron_2dev` | 1.57 µs | 1.58 µs | +0.1% | 0.89 | neutral | 2.9% / 1.6% |
| `detect_safetensors` † | 389.4 ns | 390.1 ns | +0.2% | 0.82 | neutral | 3.0% / 2.6% |
| `detect_gguf` † | 37.5 ns | 37.4 ns | −0.1% | 0.77 | neutral | 1.2% / 1.6% |
| `best_available_13dev` † | 169.2 ns | 170.0 ns | +0.5% | 0.27 | neutral | 1.2% / 1.8% |
| `total_memory_13dev` | 85.0 ns | 85.0 ns | 0.0% | 0.41 | neutral | 0.2% / 0.2% |
| `has_accelerator_13dev` | 18.2 ns | 18.2 ns | 0.0% | 0.88 | neutral | 0.1% / 0.1% |
| `plan_70B_bf16_4gpu` | 1.42 µs | 1.43 µs | +0.2% | 0.76 | neutral | 2.4% / 2.2% |
| `count_family_gpu_13dev` | 243.0 ns | 243.0 ns | 0.0% | 1.00 | neutral | 0.0% / 0.0% |
| `json_serialize_13dev` | 22.58 µs | 22.56 µs | −0.1% | 0.41 | neutral | 0.5% / 0.2% |
| `json_summary_13dev` | 3.08 µs | 3.07 µs | −0.3% | 0.22 | neutral | 1.1% / 0.6% |
| `json_system_io` | 5.12 µs | 5.12 µs | −0.2% | 0.44 | neutral | 0.7% / 0.8% |
| `json_plan` | 16.91 µs | 16.91 µs | 0.0% | 0.97 | neutral | 0.2% / 0.1% |
| `json_training` | 2.45 µs | 2.46 µs | +0.2% | 0.28 | neutral | 0.7% / 0.5% |

Values are means of the 5 per-layout medians. p is Welch's t on those medians,
so layout variation is the error term. A verdict needs p < 0.01 and
|Δ| > 1%. No row moves by more than 0.5%, which is less than the layout spread
alone. `bench-history.csv` rows `a2cb2aa` (2.3.24 source) and `e25e8c0-dirty`
(2.3.25) are single runs; the table above is the evidence.

- **Windows wall-clock: slower on `cass` because detection now does work.**
  The median of 30 interleaved runs goes from 33.6 ms (2.3.24) to 48.4 ms
  (2.3.25) per invocation. The 2.3.24 EXE's `wmic` spawn failed at once, and
  DXGI only ran after a successful `wmic`, so that EXE detected nothing. The
  added time is DXGI enumeration: in an earlier run of the same kind, a probe
  EXE doing only the DXGI walk and `GlobalMemoryStatusEx` took 43.2 ms,
  against 23.5 ms for `--version`, which detects nothing. A host that still has `wmic` used to pay for a `wmic`
  process plus one DXGI factory per adapter. No such host was available, so
  that case is unmeasured.

| binary | 2.3.24 | 2.3.25 | Δ |
|---|---:|---:|---:|
| x86_64 ELF, `CYRIUS_DCE=1` | 219 344 | 219 448 | +104 |
| x86_64 ELF, no DCE | 440 528 | 444 728 | +4 200 |
| ELF-aarch64 | 739 352 | 739 456 | +104 |
| PE, as shipped (no DCE) | 510 976 | 515 584 | +4 608 |
| agnos, `CYRIUS_DCE=1` | 217 008 | 217 112 | +104 |

### Verified

- **Tests:** 704 assertions in 14 units pass through CI's loop; fuzz 6/6.
- **CI gates:** fmt (whole CI file set), lint (0 warnings), vet, raw-offset
  guard, DCE build, distlib fresh and deterministic.
- **Linux CLI output identical to 2.3.24** on 18 invocations covering every
  flag, after normalising live sensor readings and timestamps.
  **ELF-aarch64** under `qemu-aarch64` matches x86_64 on 8 invocations.
  **agnos** builds with and without DCE.
- **PE build:** `stage_win_cross.sh` itself was run; its EXE is byte-identical
  to the one tested on `cass`, and the version bump does not change it. Its
  compiler diagnostics are the same set as 2.3.24's.
- **Windows on `cass`,** with the EXE laid out as the wheel's `_bin/`:
  `windows-smoke` (a)–(c) pass under Git Bash (`--version` reports 2.3.25),
  and (d) passes as described above. `-vv` shows `windows: dxgi hardware
  adapters=1`. The entry-point probe reports the UHD 600 from all four
  entry points.
- **Not verified:** a host where DXGI is unavailable (the `wmic` fallback
  path), a discrete GPU on Windows, and the `windows-latest` runner itself.

## [2.3.24] — 2026-09-22 — cyrius 6.6.6, bayan 1.5.6

A toolchain and dependency bump with **no source change**: cyrius `6.6.2 → 6.6.6`
(four upstream repair releases), bayan `1.5.5 → 1.5.6`, `./lib/` re-vendored, and
`cyrius.cyml` cut back to a manifest. The CLI prints the same output as 2.3.23 on
every invocation compared, on Linux and on Windows, and all 629 assertions pass
unchanged. Two parsers get faster; nothing gets slower.

### Changed

- **cyrius `6.6.2` → `6.6.6`.** `./lib/` was re-vendored from a deleted tree
  (`rm -rf lib && cyrius lib sync && cyrius deps`). 20 of the 46 stdlib files
  change and one is new: `lib/alloc_cx.cyr`, pulled in by `lib/alloc.cyr`'s new
  cx-target arm. That makes 48 files: `lib sync` copies 39 and `cyrius deps`
  adds the same 8 transitive leaves as before, plus bayan. Every vendored file is
  byte-identical to the 6.6.6 snapshot or the bayan 1.5.6 dist.
- **bayan `1.5.5` → `1.5.6`** (`c0500d9` → `fd46aa9`), a pin-only release
  upstream. `dist/bayan-json.cyr` differs from 1.5.5 in its `# Version:` line
  and nothing else.
- **`cyrius.lock` regenerated.** It has 48 hashed entries (was 47). The file is
  now sorted by path, because 6.6.3 fixed the writer that emitted readdir order.
  It also ends with a `cyrius 6.6.6` trailer line, which 6.6.4's resolver uses to
  refuse a stdlib file whose content changes under an unchanged pin. A
  clean-tree `lib sync` + `deps` now reproduces the lock byte-for-byte.
- **`cyrius.cyml` is a manifest again.** Over several releases it had filled up
  with release history: a 2.2.6 note on `json_out.cyr`, the 2.3.19/2.3.20 bayan
  move and its duplicate-definition incident, and 2.3.21's measured failures
  for the two wrong bayan/toolchain pairings. All of it was already recorded in
  this file. What remains is one short comment on `[lib]` (why module order
  matters) and one on `[deps.bayan]` (why it is optional): 127 → 74 lines. A
  stray trailing comma in `[deps].stdlib` is gone. The parsed leaf list is
  unchanged: the same 18 leaves in the same order, as the byte-for-byte
  reproduction of the wrapper's build input in *Performance* confirms.
- **`scripts/bench-history.sh` labels rows measured on uncommitted changes
  `<HEAD>-dirty`.** Before this, a release's baseline and its candidate were
  both stamped with the same parent commit.
- **CI's lock-gate comment corrected** (`ci.yml`, *Verify dep hashes*). It said
  `cyrius deps` silently repairs a stale lock. Under 6.6.4+ it refuses instead,
  naming the file. Measured: one corrupted `lib/vec.cyr` hash makes `deps` exit
  1, `--verify` reports "47 verified, 1 failed", and the lock is left
  untouched. A repointed bayan tag is refused the same way. The sorted
  comparison stays in place for the one case `deps` still rewrites: a pin bump
  committed without its re-stamped lock.

### What the toolchain changes here

Only the upstream changes that reach this repo, each checked against the tree:

- **A failed read no longer looks like a short file** (`lib/io.cyr`, 6.6.6).
  `file_read_all` used to return the bytes read before an error as if they were
  the whole file. It now returns a negative errno. All seven callers in `src/`
  already test `n <= 0` / `n > 0`, so a sysfs or procfs read that errors now
  reads as "absent" rather than as a truncated value. `file_write_all`, which
  the disk cache writes through, now loops until every byte lands instead of
  reporting one short write as success.
- **Probe tools no longer outlive ai-hwaccel on Linux** (`lib/process.cyr`,
  6.6.6). `run_tool` spawns `nvidia-smi`, `rocm-smi`, `vulkaninfo`, `hl-smi`
  and the rest through `exec_capture`, and every child it spawns now gets
  `PR_SET_PDEATHSIG(SIGKILL)`. Killing ai-hwaccel no longer leaves a hung probe
  running under PID 1. No deadline is set, because 6.6.6 makes deadlines
  opt-in via `proc_set_timeout_ms`, so probes are still waited for exactly as
  before.
- **`O_TRUNC` truncates on Windows** (6.6.6). The pre-bump roadmap said the
  Windows wheel had been corrupting the detection cache. It had not. The only
  file writer in `src/` is the disk-cache API (`disk_cached_get`,
  `src/cache.cyr:227`), `src/main.cyr` never calls it, and the wheel runs the
  CLI. The exposure was library consumers of `dist/ai-hwaccel.cyr` who build
  for Windows and use `disk_cached_*`: a shorter rewrite left the old tail in
  the file. ai-hwaccel itself never parses that file back. Measured on `cass`
  (Windows 11) by writing 130 bytes then 34 bytes through `file_write_all`:
  the 6.6.2 build leaves 130 bytes on disk, the 6.6.6 build leaves 34.
- **aarch64 binaries grow about 10%.** 6.6.5 renumbers raw syscalls through a
  per-target translation chain, which upstream prices at about 224 B per call
  site.
- **The `CYRIUS_DCE=1` PE crash** ([our 2026-09-07 issue](docs/development/issues/2026-09-07-cyrius-dce-pe-access-violation.md))
  **was fixed upstream in 6.6.1.** PE builds now decline compaction and
  NOP-fill dead code instead, so the EXE is the same size either way. On
  `cass` the 6.6.6 DCE EXE passes every `windows-smoke` assertion.
  `stage_win_cross.sh` still omits the flag. Restoring it would shrink the
  EXE inside the wheel from 77 534 to 43 684 B compressed (zip's default
  deflate); that step is left open in the roadmap.
- **`-D NAME` after the operands is honoured** (6.6.5 CLI). Nothing here
  depends on it: nothing in `src/` reads the `CUDA`/`ROCM`/`TPU`/`NO_BACKENDS`
  defines that ADR-004's recipes pass. That mismatch predates the bump and is
  on the roadmap.

### Verified

- **Tests:** 629 assertions in 14 units, run both through CI's loop and through
  `cyrius tests`, with the same compiler warnings as 2.3.23. Fuzz: 6/6
  harnesses pass.
- **CI gates:** every step of `ci.yml`'s build job passes on a fresh copy:
  lock drift, vet, raw-offset guard, fmt, lint (0 warnings), distlib fresh and
  deterministic, DCE build, benches. The format gate is real, not a no-op:
  `cyrius fmt <file> --check` fails a misformatted copy.
- **CLI output identical to 2.3.23** on 18 invocations covering every flag.
  Live sensor readings and log timestamps are normalised before comparing.
- **ELF-aarch64 under `qemu-aarch64`:** output identical to x86_64 on 8
  invocations. **agnos** builds with and without DCE.
- **Windows PE on `cass`** (Windows 11 10.0.26200), testing the EXE exactly as
  `stage_win_cross.sh` stages it: all three `windows-smoke` assertions pass for
  2.3.23, 2.3.24 and 2.3.24 with `CYRIUS_DCE=1`, and CLI output is identical
  across the three on 10 invocations. The local wine run was not useful: wine
  faults inside detection for 2.3.23 and 2.3.24 alike.
- **Not verified:** macOS arm64 at runtime. `ecb` has cyrius ≤ 6.6.4, and the
  `macos-14` wheel job builds the binary without running it.

| binary | 2.3.23 (6.6.2) | 2.3.24 (6.6.6) | Δ |
|---|---:|---:|---:|
| x86_64 ELF, `CYRIUS_DCE=1` | 214 600 | 219 344 | +4 744 (+2.2%) |
| x86_64 ELF, no DCE | 431 688 | 440 528 | +8 840 (+2.0%) |
| ELF-aarch64 | 673 200 | 739 352 | +66 152 (+9.8%) |
| PE, as shipped (no DCE) | 500 224 | 510 976 | +10 752 (+2.1%) |
| agnos, `CYRIUS_DCE=1` | 216 488 | 217 008 | +520 (+0.2%) |

### Performance

**2 faster, 13 neutral, 0 regressions.** A plain before/after would have been
misleading here, for three reasons, and each was handled:

1. **The instrument changed under the benchmarks.** 6.6.5 rewrote
   `lib/bench.cyr`: floor calibration, window netting, and how min and max are
   reported. So for the comparison the 2.3.23 arm was built against the 6.6.6
   `bench.cyr` too, by composing the compiler's input by hand and calling each
   arm's own `cycc` directly. Built with the stock `bench.cyr`, the composed
   input reproduces the wrapper's binaries byte-for-byte for both 6.6.2 and
   6.6.6, which also confirms each arm really compiles with its pinned
   toolchain.
2. **Floor-bound rows measure the clock, not the function.**
   `best_available_13dev`, `detect_safetensors` and `detect_gguf` time one op
   per window against a ~1.34 µs hpet clock read. Timing them that way showed
   `best_available` at +4.5%, but that came from the instrument code, which
   each compiler builds differently. Those three were re-timed in batch
   windows (20 000 ops per window) and are reported that way (†).
3. **Layout alone moves some rows several percent.** A never-called function
   inserted early in `src/` moved batch `detect_safetensors` by −4.9% on its
   own (p < 0.0001). A single A/B pair had shown that row at +4.5%. So each
   arm was measured at 4 or 5 code layouts, and the toolchain effect is judged
   with Welch's t on the per-layout medians, which makes layout variation the
   error term. 30 interleaved rounds, shuffled order, pinned to one CPU.

| row | 2.3.23 | 2.3.24 | Δ | p | verdict |
|---|---:|---:|---:|---:|---|
| `parse_cuda_8gpu` | 14.30 µs | 13.68 µs | −4.3% | <0.0001 | **faster** |
| `parse_vulkan_2gpu` | 2.60 µs | 2.52 µs | −3.0% | <0.0001 | **faster** |
| `parse_neuron_2dev` | 1.58 µs | 1.57 µs | −0.7% | 0.53 | neutral |
| `best_available_13dev` † | 172.3 ns | 170.4 ns | −1.1% | 0.11 | neutral |
| `detect_safetensors` † | 402.1 ns | 410.8 ns | +2.2% | 0.36 | neutral |
| `detect_gguf` † | 35.9 ns | 35.9 ns | 0.0% | 0.93 | neutral |
| `total_memory_13dev` | 85.2 ns | 85.2 ns | 0.0% | 0.72 | neutral |
| `has_accelerator_13dev` | 18.2 ns | 18.2 ns | 0.0% | 0.97 | neutral |
| `plan_70B_bf16_4gpu` | 1.42 µs | 1.42 µs | +0.5% | 0.45 | neutral |
| `count_family_gpu_13dev` | 243.9 ns | 243.6 ns | −0.1% | 0.36 | neutral |
| `json_serialize_13dev` | 22.63 µs | 22.65 µs | +0.1% | 0.58 | neutral |
| `json_summary_13dev` | 3.14 µs | 3.14 µs | +0.1% | 0.80 | neutral |
| `json_system_io` | 5.14 µs | 5.14 µs | 0.0% | 0.97 | neutral |
| `json_plan` | 17.02 µs | 17.03 µs | 0.0% | 0.81 | neutral |
| `json_training` | 2.44 µs | 2.44 µs | −0.1% | 0.84 | neutral |

Values are means of per-layout medians of `bench_avg_ns`. † = batch-timed.
Verdict needs p < 0.01 and |Δ| > 1%. Both parser wins hold in batch timing too:
`parse_cuda_8gpu` −3.0% (p = 0.0013), `parse_vulkan_2gpu` −2.4% (p = 0.0001).
An earlier 80-round single-pair run on the stock benches agrees: −3.9% and
−2.6%, with no other row outside the floor-bound three flagged.

### Notes

- **`bench-history.csv` changes meaning at this boundary.** Rows `825f724` are
  2.3.23 on cyrius 6.6.2; rows `825f724-dirty` are this release on 6.6.6. Under
  the 6.6.5 instrument, a per-window row's min only resolves when the clock's
  error (floor plus tick, ~2.3 µs on this host) is at most 1% of the window. No
  window here is that long, so the `min_ns` and `max_ns` columns now repeat the
  mean for those rows. Averages are netted at read time instead of clamped per
  window, which removes the old upward bias on floor-bound rows. Measured with
  the same compiler and code, the instrument swap alone moves rows by up to
  ±5%, and floor-bound `detect_gguf` by −24%. As always, a CSV row is one
  run; the A/B above is the evidence.
- **Lint:** 6.6.6's cyrlint also matches `for now`, so `src/cache.cyr:193` is
  a fourth untracked deferral. CI's lint step is non-gating (`|| true`, no
  `--strict-deferrals`), so nothing fails.
- **Found while verifying, not caused by the bump:** `cass` no longer has
  `wmic`, which Windows 11 24H2+ removes by default. Windows detection shells
  out to it for both GPUs and total memory. There, 2.3.23 and 2.3.24 alike
  report a CPU profile with the 16 GiB fallback on an 8 GB host and miss the
  Intel UHD 600. `windows-smoke` still passes, because a CPU profile is always
  present. Tracked in the roadmap.

## [2.3.23] — 2026-09-11 — cyrius 6.6.2

### Changed

- **Toolchain `6.6.0` → `6.6.2`.** No source change. ai-hwaccel constructs no
  `Result` / `Option` / `Either` — every `Ok(` / `Some(` occurrence in `src/` is
  inside a `#` comment — so the 6.6.0 value form has no surface here. Zero sites
  the compiler rejects, zero fail-open sites. **629 assertions** pass unchanged.

  Its five `callptr` sites were traced to their full function-pointer target sets
  and no reachable target is pair-returning. That check has to be done by hand:
  `callptr` dispatches through a runtime pointer, so the compiler cannot see a
  pair-returning callee and a clean build proves nothing about this class.

  Four candidate arity collisions were examined and all four refuted on
  independent re-derivation — `tag/1` (retired at 6.6.2, not redefined), the 131
  `bayan_json_*` names (the two definitions never co-occur in this compile set),
  16 `sys_*` names (`#ifdef`-guarded per-arch alternates), and the
  `boxed_*`/`tagged_new` set (same arity on both sides).

## [2.3.22] — 2026-09-07 — issue-folder triage: three closed on verification, one really broken

A triage pass over the four issues that predated the 2.3.21 work. **Three were
already fixed and nobody had checked**; the fourth was live, had been shipping a
wrong answer to every downstream consumer since 2.3.15, and had no test. Fixing
it turned up a second hole: part of the test suite could not fail.

### Fixed — `load_models` returned 1 model instead of 26

`load_models` does not parse JSON — it scans for `{`, brace-matches to the
object's end, extracts fields from that slice, and resumes. That is right for a
**top-level array**. `data/models.json` ships as `{"models":[ … ]}`, so the first
`{` was the **wrapper**: brace-matching ran to the last byte of the document, the
whole file became one "object", the first `"name":` in it matched, one profile was
pushed, and `pos` was already at EOF. **26 models in the file, 1 returned, no
error.** hoosh hit this and vendors an unwrapped copy to work around it.

Fixed by **option (a)** from the issue — the loader learned the wrapper, so the
shipped data file did not have to change and both shapes parse. Three adjacent
weaknesses the issue named went with it, plus a fourth it did not:

1. **The wrapper** — the scan starts just past the `[` following a `"models"` key
   when one is present; a top-level array still starts at 0.
2. **Off-by-one** — `alloc(32768)` + `file_read_all(…, 32768)` + `store8(buf + n,
   0)` wrote the NUL one byte past the allocation on an exactly-full read. Now
   reads at most `MODELS_BUF - 1`.
3. **Silent truncation** — a read that fills the buffer now warns instead of
   quietly dropping models.
4. **Wrong path (not in the report)** — it read a bare cwd-relative
   `"data/models.json"` while its sibling loader `cost.cyr:79` already went through
   `data_file_path()`. From a pip-installed wheel it read nothing and returned an
   empty vec. Now uses `data_file_path()` too.

Routing through the real bayan parser (the issue's option 3) was **declined**:
`[deps.bayan]` is `optional = true` and feature-gated, so making `load_models`
depend on it would break every consumer that links ai-hwaccel without bayan. The
hand-rolled scanner stays dependency-free.

### Fixed — the test suite could not fail

`load_models` had **zero callers inside ai-hwaccel** and no test, which is exactly
why the suite stayed green for three releases while the function returned 1 of 26.
`tests/tcyr/model_catalog_test.tcyr` now loads `data/models.json` **as shipped**
and asserts the count, so the loader and the data file are gated together — drift
between the two is what happened. Verified in both directions: against the pre-fix
loader it reports `FAIL: load_models returns every model in data/models.json (got
1, expected 26)`; against the fixed one, 6/6.

Writing it surfaced a worse problem. `json_roundtrip_test.tcyr` ended
`assert_summary(); return 0;` — **discarding the failure count**, so `syscall(60,
r)` always exited 0 and a failing assertion in that unit could never fail `cyrius
tests`, which gates on exit status. Twelve of the fourteen units already
propagated it; that one did not. Both it and the new unit now `return
assert_summary()`. Confirmed by sabotage: with a deliberately broken assertion
`cyrius tests` now exits **1**, where before it exited **0**.

### Closed on verification — three issues that were already fixed

Re-checked against the tree rather than trusting their own notes, then archived:

- **`registry_new` symbol collision** (filed 2026-06-11, vs bote-core) — `grep -c
  '^fn registry_new('` over `src/` is **0**, `hw_registry_new` is at
  `src/registry.cyr:25`, and the only three bare occurrences in
  `dist/ai-hwaccel.cyr` are the explanatory comment at `:3629-3633`.
- **`ERR_TIMEOUT` enum collision** (filed 2026-06-23, vs sakshi) — six `HWA_ERR_*`
  members, no bare `ERR_*` defined anywhere in `src/`, and **zero** bare `ERR_*`
  occurrences in the bundle.
- **Threaded GPU probe blocks the agnos build** (filed 2026-06-12) — **resolved
  upstream, no source change.** The fix this issue asked for (gate
  `thread_create`/`thread_join` behind `#ifndef CYRIUS_TARGET_AGNOS`) is now
  redundant: `lib/thread_agnos.cyr` provides `thread_create`, which runs the body
  serially inline, snapshots and restores the caller's thread-local slots, and
  returns a fake non-zero handle so null-checks and `thread_join` stay valid — the
  exact contract the issue wanted hand-rolled. `lib/sync.cyr` has a matching agnos
  branch making `mutex_*` allocating no-ops, covering the `cache.cyr` / `lazy.cyr`
  sweep it also asked for. Verified rather than assumed:

  ```
  cyrius build --agnos src/main.cyr              -> OK (425 288 bytes)
  CYRIUS_DCE=1 cyrius build --agnos src/main.cyr -> OK (216 392 bytes)
  ```

  `CLONE_VM` now appears only in `lib/syscalls_x86_64_linux.cyr` (not compiled for
  agnos) and in comments. Gating was **deliberately not added** — it would
  duplicate, and probably get wrong, upstream's TLS-isolation contract.

### Performance

ABBA-balanced A/B, same toolchain, n=30 per arm, Mann-Whitney: **0 regressions,
14 neutral, 1 marginal improvement** (`json_plan` −1.0%, p=0.0016 — at the
threshold, treated as neutral).

The change is not reachable from either benchmark — `load_models` has no bench
caller — so this measures layout only. `benches/parsing.bcyr` compiles to a
**byte-identical** binary in both arms and serves as the noise floor in the same
run: those five rows still swing up to **+2.3%** (p ≥ 0.24), which is the scale
below which nothing in this table means anything. `registry.bcyr` moved by 96
bytes of layout. Suite: 623 → 629 assertions across 14 units (the new catalogue
unit adds 6). Binary 214 504 → 214 600 B (+96 B).

### Notes

- **Consumers**: anyone calling `load_models` against the shipped
  `data/models.json` was getting one model. They now get all 26 with no change on
  their side. A consumer that vendored an unwrapped top-level array (hoosh) keeps
  working — both shapes parse. `data/models.json` itself is **unchanged**.
- Still open in `docs/development/issues/`: the four filed during 2.3.21 (all
  marked RESOLVED there, kept as that release's record) and the upstream cyrius
  `CYRIUS_DCE=1` PE defect, which is not ours to close.

## [2.3.21] — 2026-09-07 — cyrius 6.6.0, and the five defects it surfaced

**Toolchain `6.5.36 → 6.6.0` (38 releases) + bayan `1.5.2 → 1.5.5`**, plus the
five defects the bump exposed — four pre-existing portability bugs in this repo,
and one in the toolchain that would have shipped a **Windows wheel whose EXE
crashes on launch**.

Every one of the four repo defects is the same shape: **a host assumption with no
target guard.** They survived this long because the suite only ever ran x86_64
Linux, single-threaded. This release was verified on **all four targets** instead.

### Verified per platform

| target | how | result |
|---|---|---|
| x86_64 Linux | native | 623 assertions / 13 units, 6/6 fuzz, all gates |
| **x86_64 PE (Windows)** | wheel built + run on `cass`, Win 11 | all 3 `windows-smoke` assertions pass; real CPU+GPU detection |
| **arm64 macOS** | native build + run on `ecb` | exit 0, DCE and no-DCE output identical |
| ELF-aarch64 | `qemu-aarch64` | exit 0, output identical to x86_64 |

### Fixed — the toolchain defect that broke the Windows wheel

- **`CYRIUS_DCE=1` produces a PE that dies at startup with `0xC0000005`.** Same
  source, same toolchain, flag the only difference — measured on `cass`:

  ```
  cyrius build --win              -> exit 0, 499 bytes of registry JSON
  CYRIUS_DCE=1 cyrius build --win -> exit -1073741819 (0xC0000005), no output
  ```

  Under Git Bash on `windows-latest` (`defaults.run.shell: bash`) that surfaces
  as **exit 139**, failing `windows-smoke` at its first assertion. It is
  PE-specific: `CYRIUS_DCE=1` is correct on x86_64 ELF (where it now genuinely
  reclaims, −48.8%), on ELF-aarch64, and on Mach-O arm64 — all verified running.
  `bindings/python/scripts/stage_win_cross.sh` therefore drops the flag on the
  `--win` build **only**; it bought nothing there anyway (497,152 bytes either
  way). Upstream defect, filed as
  [2026-09-07-cyrius-dce-pe-access-violation](docs/development/issues/2026-09-07-cyrius-dce-pe-access-violation.md);
  the toolchain is not this repo's to patch. Restore the flag when a cyrius
  release fixes it *and* `windows-smoke` passes with it on.

### Fixed — four portability defects

- **`AI_HWACCEL_DATA_DIR` now works on macOS and Windows** — it was a silent
  no-op on both. `cmd_getenv` was a local `/proc/self/environ` reader; neither
  target has `/proc`, so it returned 0 for **every** name. cyrius 6.6.0 made the
  stdlib `getenv` correct on every target we ship to (init-stack `envp` on macOS,
  the `0xF015` `GetEnvironmentVariableA` reroute on Windows), so `cmd_getenv` is
  now a one-line delegation to it. This also repairs `which()`'s `$PATH` lookup
  and `NVIDIA_VISIBLE_DEVICES` there. **Verified on both:** `--version` resolves
  a probe VERSION through the env var on `cass` and on `ecb`, where it previously
  could not. Issue
  [2026-09-07-cmd-getenv-proc-only](docs/development/issues/2026-09-07-cmd-getenv-proc-only.md).

- **The disk cache works on aarch64** — it silently did nothing there.
  `src/cache.cyr` issued raw **x86_64** `syscall(83)`/`syscall(87)` for
  mkdir/unlink; ELF-aarch64 has neither (only `mkdirat` 34 / `unlinkat` 35) and
  cyrius does not remap those two. Now routed through the per-target `sys_mkdir` /
  `sys_unlink` peers behind the same `#ifdef` pattern `read_symlink` uses (AGNOS's
  peers take an explicit pathlen). Measured under `qemu-aarch64`:

  ```
  aarch64   old syscall(83) = -9 (EBADF)   new sys_mkdir = 0    dir created: NO -> YES
  x86_64    old syscall(83) =  0           new sys_mkdir = 0    (unchanged)
  ```

  `CYRIUS_DCE=1` eliminates the disk-cache API from the CLI, so the shipped
  executable was never affected — this is for library consumers of
  `dist/ai-hwaccel.cyr`. Issue
  [2026-09-07-cache-raw-syscalls-wrong-on-aarch64](docs/development/issues/2026-09-07-cache-raw-syscalls-wrong-on-aarch64.md).

- **Cache TTL expiry is defined on macOS and Windows** — it was an uninitialised
  stack read. `_monotonic_secs()` was a bare `syscall(228, 1, &ts); return
  load64(&ts);`. On macOS the route *returns* the ns count and never touches
  `&ts` (and Darwin's `CLOCK_MONOTONIC` is **6**, not 1); on Windows it returns ms
  from `GetTickCount64` and never touches `&ts`. Now branched per target,
  mirroring `lib/chrono.cyr` rather than re-deriving it, with AGNOS on
  `sys_uptime_ms()`. Issue
  [2026-09-07-monotonic-secs-unguarded-on-macos-windows](docs/development/issues/2026-09-07-monotonic-secs-unguarded-on-macos-windows.md).

- **Detector threads no longer log.** `registry_detect_threaded` spawns six
  threads whose parsers called `hwlog_warn` directly, and `lib/sakshi.cyr`'s
  header states it is *"Single-threaded only, EXCEPT `SK_OUT_ATOMIC_RING`"* —
  which `hwlog_init` does not select. Live on Linux for as long as the threaded
  API has existed; cyrius 6.5.44 extended it to arm64-macOS by making Darwin
  threads real. All four sites **already** pushed the same text into the warnings
  vec, which is merged on the main thread, so emission moved to a new
  `warnings_log_parse()` (`src/error.cyr`) called from the main thread in both
  detection paths, filtered to `HWA_ERR_PARSE`. Verified byte-equivalent: same
  line emitted, same suppression at every level. Issue
  [2026-09-07-threaded-detect-vs-single-threaded-sakshi](docs/development/issues/2026-09-07-threaded-detect-vs-single-threaded-sakshi.md).

### Changed — toolchain and dependency

- **cyrius pin `6.5.36` → `6.6.0`.** `./lib/` re-synced: `cyrius lib sync` copies
  the 38 declared `[deps].stdlib` leaves (16 of them change); `cyrius deps` then
  pulls the transitive **`lib/hashseed.cyr`** (new at 6.5.39, required by the
  declared `hashmap` leaf), taking `./lib/` from 45 to 46 files. **`lib sync`
  alone no longer yields a complete tree** — the `CLAUDE.md` vendoring principle
  was corrected to say so.
- **`cyrius.lock` regenerated** — 45 → 46 hashed entries, 16 stdlib hashes
  rotated, bayan commit `b4cb1d8` → `c0500d9`. Format unchanged.
- **`[deps.bayan]` tag `1.5.2` → `1.5.5` — forced, and atomic with the pin.** The
  arity gate is the *vendored* `lib/result.cyr`, not `cycc`, and the two wrong
  pairings fail in opposite ways (both measured):

  ```
  1.5.2 + 6.6.0 lib/   -> HARD COMPILE ERROR   bayan-json.cyr:811, :820
  1.5.5 + 6.5.36 lib/  -> BUILDS CLEAN, 0 warnings, destructures a box pointer
                          into a (tag, payload) pair. Garbage, silently.
  ```

  Neither half is landable alone.
- **`dist/ai-hwaccel.cyr`** regenerated. `dist/ai-hwaccel.deps` unchanged at 18
  stdlib leaves.

### Fixed — a CI gate that could not fail

- **The `cyrius.lock` drift gate was dead**, and this release is exactly the
  commit it exists to catch. `ci.yml` ran `cyrius deps` and then `cyrius deps
  --verify` — but `deps` **rewrites the lockfile**, so verify only ever checked
  the file it had just written. Measured: corrupt a hash, run them in CI order,
  get `46 verified, 0 failed` and a silently repaired file; run verify alone
  against the pre-bump lock and get `30 verified, 15 failed`, exit 1.

  The step now compares the regenerated lock against `git show HEAD:cyrius.lock`
  — **sorted**, which is not cosmetic: `cyrius deps` emits hash lines in hash-map
  iteration order, stable across repeated runs but **not** across the clean-tree
  `lib sync` + `deps` that CI does (5 successive runs byte-identical; one clean
  rebuild reordered 6 lines). A plain `git diff --exit-code` would have gone red
  on line order alone. Tested both ways: passes on a reordering clean rebuild,
  fails with 35 differing lines on the pre-bump lock. The `|| echo` masking
  `cyrius deps`' exit code is also gone.

### Performance

Inherited from the toolchain: 6.5.71 routes `#derive(accessors)` getters/setters
through the inline-replay path, and every heap struct here is derived.
Per-arm shadow `CYRIUS_HOME` with a drift guard (without it both arms resolve the
same `cycc` and the A/B measures nothing), 9 interleaved build rounds then 40
alternating executions, Mann-Whitney over the distributions:

| benchmark | 6.5.36 | 6.6.0 | Δ | p |
|---|---:|---:|---:|---:|
| `best_available_13dev`   |    364 ns |    161 ns | **−55.8%** | 0.0000 |
| `total_memory_13dev`     |    136 ns |     82 ns | **−39.5%** | 0.0000 |
| `has_accelerator_13dev`  |     27 ns |     17 ns | **−37.0%** | 0.0000 |
| `plan_70B_bf16_4gpu`     |  1 780 ns |  1 382 ns | **−22.4%** | 0.0000 |
| `count_family_gpu_13dev` |    284 ns |    231 ns | **−18.7%** | 0.0000 |
| `json_summary_13dev`     |  3 272 ns |  2 968 ns | **−9.3%**  | 0.0000 |
| `parse_neuron_2dev`      |  1 650 ns |  1 564 ns | **−5.2%**  | 0.0000 |
| `parse_vulkan_2gpu`      |  2 676 ns |  2 572 ns | **−3.9%**  | 0.0001 |
| `json_system_io`         |  4 956 ns |  4 788 ns | **−3.4%**  | 0.0000 |
| `json_serialize_13dev`   | 22 195 ns | 21 474 ns | **−3.3%**  | 0.0000 |
| `json_plan`              | 16 594 ns | 16 131 ns | **−2.8%**  | 0.0016 |
| `parse_cuda_8gpu`        | 14 070 ns | 13 746 ns | **−2.3%**  | 0.0001 |
| `detect_gguf` / `detect_safetensors` / `json_training` | | | neutral | >0.25 |

**12 wins (2.3–55.8%), 3 neutral, 0 regressions** from the toolchain. A control
A/B holding `cycc` at 6.6.0 and swapping only the stdlib and bayan is neutral on
all fifteen, so the win is codegen.

**The four source fixes then cost one layout-attributable move**, reported rather
than rounded away. Their own ABBA-balanced A/B flagged four "regressions", three
of them in code the fixes do not touch. Two controls show why: a
**semantically-null** dead function added to unchanged source produced
`plan_70B_bf16_4gpu` **+2.3%, p=0.0028** and swung `best_available_13dev` −9.1%;
and restoring the three `hwlog_warn` calls in `parse_cuda_output` — returning the
binary to its exact pre-fix size — left `parse_cuda_8gpu` at +1.7% vs +1.9% with
them removed, so the changed function is not the cause. The mechanism is
`parse_csv_line` / `validate_device_id` / `validate_memory_mb` all living in
`src/detect/command.cyr`, where `cmd_getenv` shrank by 56 lines. Net: **0
algorithmic regressions, 1 layout move (`parse_cuda_8gpu` ~+1.8%), 14 within
noise.** Accepted because the alternative is keeping a `/proc`-only environment
reader broken on two of three shipped wheel targets.

Floor-bound rows (`best_available_13dev`, `detect_gguf`, `detect_safetensors`
report `min=0` against a measured 1.34 µs timer floor) swing ±10–24% freely and
are never headlined. CSV audit trail: `bench-history.csv`.

### Binary size

`CYRIUS_DCE=1`, built from the published release tarballs so the cross-compilers
match CI:

| target | 6.5.36 | 6.6.0 | Δ |
|---|---:|---:|---:|
| x86_64 ELF (Linux) | 419 360 B | **214 504 B** | **−48.9%** |
| x86_64 PE (Windows) | 489 984 B | 497 152 B | +7 168 B (+1.5%, and now built without DCE) |
| arm64 Mach-O | — | 691 808 B | DCE reclaims nothing on this backend |
| ELF-aarch64 | 673 160 B | 673 096 B | −64 B |

Only the x86_64 ELF backend actually reclaims. 6.5.72 is the release that turned
DCE from NOP-padding into real elimination — and the same release is the prime
suspect for the PE crash above.

### Fixed — documentation the bump made false

- **`src/detect/command.cyr`** — three comment blocks explained the
  `AI_HWACCEL_DATA_DIR` limitation as "cyrius exposes no `GetEnvironmentVariable`
  reroute" and claimed the variable was "honoured on Linux/macOS". Both wrong,
  and both now moot.
- **`.github/workflows/ci.yml`** — the dep steps described ai-hwaccel as
  "stdlib-only (no external git deps)" with "nothing to verify"; false since
  2.3.19, and the second described a live 46-hash gate as a dormant placeholder.
- **`.github/workflows/wheels.yml`** — step (c)'s note that `AI_HWACCEL_DATA_DIR`
  is unreadable on PE.
- **`.gitignore`** — the `lib/` note named the retired `cyrius deps` stdlib path.
  Also now ignores the bare `/ai-hwaccel` binary that `[build] output` drops in
  the repo root, which has been committed by accident before.
- **`CLAUDE.md`** — the compiler pin (four bumps stale), a second stale pin in the
  vendoring principle, the removed `cc5` alias, and the counts (`590 assertions,
  20 benchmarks` → measured **623 across 13 units, 15 benchmarks in 2 suites**).
- **`README.md`** — binary size (`286 KB`, matching neither arm), compiler
  (`6.0.0`), test counts, `detect/` module count (19 → 20), and a development
  recipe telling fresh clones to run `cyrius deps` to repopulate `lib/`.
- **`CONTRIBUTING.md`, `docs/guides/testing.md`** — `518 assertions (11 test
  phases)` → `623 assertions (13 test units)`, `cyrius test` → `cyrius tests`.

### Notes

- **The `Result` / `Option` / `Either` value form is a no-op for this repo's own
  source.** 6.6.0's headline breaking change touches nothing in `src/`, `tests/`,
  `benches/` or `fuzz/` — the repo calls none of that surface. It reaches
  ai-hwaccel only through bayan, which is why the dep bump is mandatory.
- **Consumers**: `dist/ai-hwaccel.cyr` exports the same symbols with the same
  arities. A consumer that **calls `profile_from_json_str`** must build with a
  6.6.0-class toolchain and supply a value-form bayan (1.5.5) in the same commit;
  one that never calls it is unaffected. A consumer pinned in **cyrius
  6.5.50–6.5.55** should move off that band first — those releases miscompile
  prefix-colliding identifiers with no diagnostic. ai-hwaccel went 6.5.36 → 6.6.0
  and never entered it.
- **Still open:** the tests that would have caught these — a `qemu-aarch64`
  disk-cache test, and exercising the threaded path on real Apple Silicon. The
  suite still covers only x86_64 Linux, single-threaded.

## [2.3.20] — 2026-08-30 — the focused bayan dep no longer leaks into consumers

2.3.19 replaced the 641 KB bayan monolith with the focused 100 KB
`dist/bayan-json.cyr` sublib — a real win, measured at **−290,040 B (−33.7 %)**
in chakshu. But the dep still resolved **transitively**, so it reached every
consumer whether or not that consumer wanted it. This release keeps the win and
stops the leak. No source change: 122 assertions pass, benchmarks show no
regression on any of 15.

### Fixed — `[deps.bayan]` is now optional and feature-gated

A consumer that already links the stdlib bayan monolith got `lib/bayan-json.cyr`
vendored **alongside** `lib/bayan.cyr`, producing **131 duplicate function
definitions** in which the older 1.5.2 copy silently won ("last definition
wins"). Seven of the twelve known consumers are in that position — hoosh,
agnosai, szal, stiva, iam, kavach, ranga. All 131 bodies happen to be identical
today, so this was a supply-chain hazard rather than a live defect, but it is
precisely the failure mode cyrius `docs/ecosystem.md` records for patra pinning
a stale sakshi: a folded module pinning its own dep silently downgrades that
module for every transitive consumer.

The dep is now `optional = true`, activated by a `[features] default = ["bayan"]`
entry. cyrius resolves `[features]` from the **root** manifest only (v6.3.1
lever 2), so:

- **this repo's** builds and tests activate it and are unchanged;
- a **downstream** build that does not name `bayan` skips the dep entirely — no
  clone, no module copy, no include push.

### What consumers need to do

Usually nothing. `dist/ai-hwaccel.cyr` references eleven `bayan_json_v_*` call
sites and **defines none**; they are reachable only through
`profile_from_json_str`. Verified against all twelve known consumers:

- **Nine already supply bayan** — via `"bayan"` in `[deps].stdlib` (agnosai,
  szal, stiva, iam, kavach, ranga, hoosh) or their own `[deps.bayan]` (daimon,
  samay). They keep working, and the seven stdlib ones lose 131 duplicate-fn
  warnings.
- **Three do not link bayan at all** (chakshu, agnodrm, mihi) and never call
  `profile_from_json_str`, so the symbols are unreachable and eliminated. They
  stop vendoring a 100 KB file they never used.

A consumer that *does* call `profile_from_json_str` and links no bayan must now
supply it — `"bayan"` in `[deps].stdlib`, the same focused `[deps.bayan]` block,
or `--features bayan`. That was already the instruction in 2.3.19's *Note for
consumers*; it is now enforced by resolution rather than by documentation.

### Changed

- **Cyrius pin `6.5.35` → `6.5.36`**, matching the installed compiler; `lib/`
  resynced (38 declared modules). Clears the toolchain-drift warning.

### Performance

Interleaved A/B, 3 rounds each, same machine and session: 2.3.19 source with the
6.5.35 stdlib versus this tree with 6.5.36, both compiled by cycc 6.5.36 (the
source is byte-identical, so this measures the stdlib swap alone). **No
regressions** across all 15 benchmarks; every delta is within its own run-to-run
spread. Largest movements: `json_system_io` 4945 → 4821 ns (−2.5 %),
`plan_70B_bf16_4gpu` 1833 → 1795 ns (−2.1 %), `count_family_gpu_13dev` 284 →
278 ns (−2.1 %).

## [2.3.19] — 2026-08-24 — canonical bayan API; consumers no longer forced onto the 641 KB monolith

`src/json_out.cyr` called bayan's **legacy back-compat aliases** (`json_v_obj_get`,
`json_v_int`, `json_v_bool`, `json_v_is_str`, `json_v_str`, `json_v_is_obj`,
`json_v_parse_buf`). Those aliases ship **only in the monolithic `dist/bayan.cyr`** —
bayan's focused `dist/bayan-json.cyr` sublib exports the canonical `bayan_json_v_*` names
only. So every consumer of this bundle was transitively forced to link all 641 KB of
bayan (json + toml + cyml + csv + base64 + bigint + u128 + yaml + pdf) to satisfy seven
JSON calls.

Measured downstream in chakshu, whose lean binary links this bundle for the GPU panel:
substituting the sublib for the monolith takes it from **861,536 B to 571,496 B —
−290,040 B (−33.7%)**, with its full suite still green.

### Changed

- **All 11 `json_v_*` call sites → canonical `bayan_json_v_*`** (`src/json_out.cyr`).
  Behaviour is identical; the aliases and the canonical names are the same functions.
  cyrius `docs/stdlib-modules.md` documents the `bayan_*` form as canonical and the bare
  names as legacy aliases, dating from the v6.1.25 carve of json/toml/csv/… out of stdlib.
- **`bayan` moves from `[deps].stdlib` to a focused git dep** pinned to
  `dist/bayan-json.cyr` (100,309 B, versus the 641,083 B monolith). A clean
  `cyrius deps` now vendors `lib/bayan-json.cyr` and no `lib/bayan.cyr` at all.
- **`dist/ai-hwaccel.deps` drops `bayan`** — 19 → 18 stdlib leaves. Consumers are no
  longer told to pull the monolith on this bundle's behalf.
- **Cyrius pin `6.5.32` → `6.5.35`**, clearing the toolchain-drift warning that
  `cyrius distlib` emitted against the installed cycc.

### Note for consumers

`dist/ai-hwaccel.cyr` references seven `bayan_json_v_*` symbols and defines none — they
are reached only through `profile_from_json_str`. A consumer that calls that function
should add the same focused dep:

```toml
[deps.bayan]
git = "https://github.com/MacCracken/bayan.git"
tag = "1.5.2"
modules = ["dist/bayan-json.cyr"]
```

A consumer that does **not** call it (chakshu and mihi both do not) will see seven
`undefined function 'bayan_json_v_*'` warnings on that unreachable path and can either
add the dep to silence them or ignore them. Either way it is now a **100 KB opt-in**
rather than a 641 KB obligation.

## [2.3.18] — 2026-08-19 — definitive names for three symbols kavach also defined

### Changed — three symbols renamed

| was | now |
|---|---|
| `var BACKEND_COUNT` | `var AIHW_BACKEND_COUNT` |
| `enum Backend` | `enum AiHwBackend` |
| `fn path_exists` | `fn aihw_path_exists` |

All three were also defined by **kavach**, and Cyrius has one flat symbol table with
last-definition-wins. `BACKEND_COUNT` was the damaging one: 18 here, 10 in kavach, and in any binary
linking both this definition won. kavach's `_backend_fp` uses it as the bounds check on a **10-slot**
table, so the guard admitted ids 0–17 and `_backend_slot(17)` read 224 bytes past the end — a wild
function-pointer load, which is exactly what kavach's own comment says the check exists to prevent.

Measured rather than inferred: a probe returning `BACKEND_COUNT` as its exit code printed **18** at
cyrius 6.5.32, in a project linking both libraries through agnosai.

The compiler is **silent** on a duplicate `var`, and `check-symbols.sh` in every sibling scans `src/`
only — so a `lib/`↔`lib/` collision between two dependencies is invisible to every gate in the
ecosystem. Fixed at the source in both libraries rather than worked around downstream; see
kavach 3.11.15.

**Not breaking for the enum rename.** In Cyrius an enum qualifier is cosmetic — `Backend.X` and
`AiHwBackend.X` both resolve to the member `X`, with the type name playing no part in resolution
(verified against agnosai, which builds unchanged). Members are untouched (`BACKEND_CUDA`,
`BACKEND_ROCM`, …).

Which means only `BACKEND_COUNT` and `path_exists` closed real collisions here; the `enum Backend`
rename is defensive — it removes a duplicate symbol-table entry that was never resolved through.

### Changed — Cyrius pin 6.5.27 → 6.5.32

Also clears real drift: the installed toolchain was 6.5.32 while the manifest pinned 6.5.27, so
`lib sync --full` and `deps` had been provisioning from a version the manifest did not name.

### Fixed — five files failing `cyrfmt --check`

`src/{model_format,profile,plan}.cyr` and `tests/tcyr/{requirement,model_format}_test.tcyr`.
**Pre-existing** — verified unformatted at the 2.3.17 tag, so this is drift the rename surfaced
rather than caused. `git diff -w` over the three `src/` files is empty.

**636 assertions green, 0 failed** — identical to the pre-rename baseline. `dist/ai-hwaccel.cyr`
regenerated (6,265 lines), carrying the new names and none of the old. Cross-checked against
`dist/kavach.cyr` 3.11.15: **0 remaining collisions** between the two.

## [2.3.17] — 2026-08-17

### Changed

- **Cyrius pin `6.5.10` -> `6.5.27`** (2026-08-17, ecosystem-wide ML/AI-arc realign ahead of
  the arc reopening). `cyrius lib sync --full` re-vendored the version-matched stdlib snapshot,
  clearing the toolchain-drift warning.

### Fixed

- **`tests/tcyr/foundation_test.tcyr` asserted a backend count that had been wrong since
  `BACKEND_AGNOS_GPU` was added.** `BACKEND_COUNT` is **18** (`src/types.cyr:356`,
  `BACKEND_AGNOS_GPU = 17`) while the test still asserted 17, so `foundation_test` was red —
  **before** the pin bump, not because of it (`120 passed, 1 failed` at the old pin and the
  new one alike). Corrected to 18 and added the missing `backend_name(BACKEND_AGNOS_GPU) ==
  "agnos-gpu"` assertion, so the count and the name table are gated together and the next
  backend cannot land half-tested.
- **Removed `cyrius.lock`.** ai-hwaccel declares **zero git deps** — it is stdlib-only — and its
  own CI comments say so: *"cyrius deps emits no lockfile and there is nothing to verify. The
  check stays in place so it engages automatically the day a git dep gets added."* A lockfile
  was nonetheless present, recording **99** `lib/` hashes from an older full-snapshot sync (33
  of the 99 do not match the 6.5.27 stdlib, so it predates this cycle's pin bump). Because
  `lib/` is gitignored, CI rebuilds it from the declared **19**-module `[deps].stdlib` subset,
  so `cyrius deps --verify` compared 99 recorded hashes against the ~24 files that actually
  exist: **24 verified, 75 failed** — most "cannot hash" (file absent), the rest content
  mismatches against the older stdlib. The lock had nothing legitimate to lock. Deleting it
  restores the documented behaviour and matches the ecosystem's two other stdlib-only,
  gitignored-`lib/` repos (avatara, hadara), neither of which carries one. Verified the state
  is stable, not just quiet: with it gone, `cyrius deps`, `cyrius build` and `cyrius test` all
  run clean and **none of them recreates it**. Deliberately NOT added to `.gitignore` — the
  dormant CI check is supposed to engage on its own the day a git dep is added, and ignoring
  the file would silently disable that.

## [2.3.16] — 2026-07-30

**Toolchain catch-up, and the one call it broke.** cyrius 6.5.0 renamed bayan's cstr+len JSON parse
entry points, which left `profile_from_json_str` calling a symbol that no longer exists. Reported by
hoosh while updating to cyrius 6.5.2; on ai-hwaccel's own tree it is a hard build failure, not a
warning, because `json_roundtrip_test` exercises the function.

### Fixed
- **`profile_from_json_str` called `json_v_parse_str`, removed in bayan 1.3.0 (cyrius 6.5.0)** —
  renamed to `json_v_parse_buf`. The bodies are byte-identical, so this is a pure rename with no
  behaviour change. The rename is not reversible: `X_str` is a **reserved overload slot** in Cyrius
  — a call `X(a, …)` routes to `X_str` whenever `a` is Str-typed at the call site — so a `(ptr, len)`
  form may never occupy that name. A comment at the call site now says so.
  Consumers on cyrius ≥ 6.5.0 saw `warning: undefined function 'json_v_parse_str'` and, if they
  called it, a runtime SIGILL — cycc lowers an undefined call to `ud2`. hoosh 2.5.12 shipped with
  the warning because the function is dead in that binary.

### Changed
- **Toolchain: cyrius `6.4.69` → `6.5.2`.** No other source change was needed. The jump also removes
  the whole `regex_*` surface; ai-hwaccel uses none of it, verified by diffing every public symbol
  across the two lib trees. `CLAUDE.md`'s compiler line said 6.2.11 and now points at the pin
  instead of restating a version that goes stale.

### Performance
- **No regressions across the toolchain jump** — 15 benchmarks compared before and after, per the
  mandatory-benchmarking rule. The JSON paths all improved: `json_summary_13dev` −6%,
  `json_system_io` −6%, `json_training` −5%, `json_serialize_13dev` −4%, `json_plan` −4%. Everything
  else is within ±4% noise.

### Known issues
- **`load_models` returns 1 model instead of 26** — it byte-scans for `{` and brace-matches, which
  makes the `{"models":[…]}` wrapper that `data/models.json` actually ships look like a single
  object. Silent: no error, just a short catalog. Zero callers inside ai-hwaccel and no test, which
  is why the suite stays green; hoosh vendors an unwrapped copy and guards it. Filed as
  [`2026-07-30-load-models-misparses-its-own-data-file.md`](docs/development/issues/2026-07-30-load-models-misparses-its-own-data-file.md)
  rather than fixed here, because the two candidate fixes differ in whether they change a published
  artifact, and that is a compatibility call. The issue also records three adjacent weaknesses in
  the same function: a one-byte overflow at exactly 32768 bytes, silent truncation past that, and
  the fact that a real JSON parser is already linked in.

## [2.3.15] — 2026-07-20

**Profile JSON round-trip: `profile_from_json` + lossless `profile_to_json`.**
Adds the deserialization half that never existed, and fixes the emit's silent
data loss for accelerator-specific profiles. The driver was samay, whose
`NodeCapacity.accel_profiles` is a vec of ai-hwaccel profile pointers that its
M4 milestone must round-trip through JSON — previously impossible, because
`profile_to_json` dropped `tpu_version` / `tpu_chips` / `gaudi_gen` /
`neuron_chip` / `neuron_cores` (and the perf fields), so a rebuilt TPU profile
could not satisfy the very `REQ_TPU{min_chips}` requirement it was registered
for.

#### Added
- `profile_from_json(v)` — reconstruct a profile from a bayan JSON object value
  (`json_v*`); returns `0` on a non-object root. Rebuilds the raw stored state
  directly, so `memory_bytes` and every accelerator-specific field come from the
  document rather than being recomputed by the `profile_cuda`/`profile_tpu`/…
  convenience constructors.
- `profile_from_json_str(js_cstr)` — parse a JSON cstr and reconstruct in one
  step; `0` on malformed JSON or a non-object root.
- `bayan` added to `[deps].stdlib` (the JSON DOM the parser reads); no symbol
  collisions with ai-hwaccel's surface (verified).

#### Changed
- **`profile_to_json` is now lossless** (schema **v4 → v5**). New keys:
  `accel_type_id` (the raw `AcceleratorType` enum — the authoritative type tag
  on parse; the human `accelerator` name and derived `family` are kept but
  ignored on the way back), plus the previously-dropped `mem_bandwidth_x1000`,
  `pcie_bandwidth_x1000`, `power_x1000`, `tpu_version`, `tpu_chips`, `gaudi_gen`,
  `neuron_chip`, `neuron_cores`. All are emitted only when meaningful, so CPU/GPU
  output is unchanged apart from the new `accel_type_id`. Additive and
  backward-compatible: readers that ignore unknown keys are unaffected; the
  `schema_version` bump is the signal that the new fields are available.
- Toolchain pin `6.4.62 → 6.4.69` (the release that landed round-trip-correct
  f64 JSON — Grisu2 emit + correctly-rounded parse — which samay's M4 also needs).

#### Tests
- `tests/tcyr/json_roundtrip_test.tcyr` — 8 groups: CUDA (with strings), TPU
  (chips + version + the `requirement_satisfied` property samay depends on),
  Neuron cores, Gaudi gen, CPU-defaults (no field leakage), explicit perf
  fields, and malformed/non-object input (28 assertions). Full suite
  **607 → 635 assertions**, all green.

## [2.3.14] — 2026-07-13

**`registry_new → hw_registry_new` — the second half of the 2026-06-11
symbol-collision fix.** 2.3.13 namespaced the `DetectionError` enum
because 6.4.62 promoted the `sakshi` collision to an in-tree warning; this
release finishes the job on the sibling the same issue flagged. ai-hwaccel
exports `fn registry_new()` — a **32-byte** profile registry
(`{profiles, warnings, system_io, schema}`). bote-core exports a
**different** `fn registry_new()` — a **24-byte** tool registry. Cyrius
functions are *global* symbols resolved by textual-paste + last-definition-
wins, so any consumer that bundles both `dist/ai-hwaccel.cyr` and
`dist/bote-core.cyr` (szal, mihi, hoosh) gets exactly one `registry_new`,
and every caller of the other silently allocates/interprets the wrong
struct layout. This is **not** a benign warning: ai-hwaccel's own
`registry_detect` / lazy / async paths call `registry_new()` internally,
so if bote's 24-byte constructor wins the link, our code writes the
`system_io`/`schema` fields past the end of a 24-byte blob — memory
corruption, and no include order avoids it. Renaming ai-hwaccel's
constructor to `hw_registry_new` is the recommended upstream fix from the
issue; pulled ahead of its filed 2.4.0 target to sit next to the enum
rename it belongs with.

#### Changed

- **`registry_new` → `hw_registry_new`** (`src/registry.cyr:20` def, plus
  its four internal callers: `src/registry.cyr` `registry_detect` /
  `registry_from_profiles`, `src/lazy.cyr` `lazy_into_registry`,
  `src/async_detect.cyr` `registry_detect_threaded_with`). `hw_` matches
  the existing `reg_*` / hardware-flavored surface — only the bare
  `registry_new` constructor collided. `tests/tcyr/registry_test.tcyr` and
  `benches/registry.bcyr` updated to the new name; `dist/ai-hwaccel.cyr`
  regenerated (no bare `registry_new` code symbol remains).
- **No `registry_new` alias kept** — an alias by that name would
  reintroduce the exact bote-core collision, the same reason 2.3.13 kept
  no bare `ERR_*` aliases. `cached_registry_new` is **unchanged**: it is
  ai-hwaccel-specific and does not collide with any bote symbol.
- **`VERSION`** 2.3.13 → 2.3.14; **`dist/ai-hwaccel.cyr`** regenerated
  (embeds 2.3.14).
- **Python bindings version now derives from `VERSION`.**
  `bindings/python/src/ai_hwaccel/__init__.py` `__version__` was a
  hardcoded literal — it read `2.3.4` while the binary was at 2.3.14. It
  now resolves from the installed wheel metadata (falling back to the
  repo `VERSION` file when run uninstalled), so it carries no version
  literal and cannot drift. `bindings/python/pyproject.toml`'s wheel
  metadata version is still a literal (setuptools can't read the
  out-of-root `VERSION`), but `scripts/version-bump.sh` now propagates
  `VERSION` into it on every bump, and a new **"python bindings version"
  CI gate** asserts the two match — the same drift-catch the distlib gate
  gives `dist/ai-hwaccel.cyr`.

#### Fixed

- **`registry_new` symbol collision with bote-core** (docs/development/
  issues/2026-06-11-registry-new-collision.md) — a consumer linking both
  bundles no longer gets a 24-vs-32-byte layout mismatch on
  `registry_new`. Retires the consumer-side vendoring+sed workaround szal
  carried.
- **Python wheel shipped stale version metadata.** Every wheel was tagged
  `2.3.7` (pyproject) and `pip show`/`__version__` reported `2.3.7` /
  `2.3.4` for a 2.3.14 binary — a packaging-only bug (no binary/behaviour
  change), so the perf table below is unaffected. `bindings/python/README.md`
  also still claimed macOS/Windows wheels were unpublished; corrected (all
  three platforms build in `wheels.yml`).

#### Performance

**Metric:** raw `bench_min_ns`, min-of-8, `CYRIUS_DCE=1`, same machine. A
constructor **rename** is codegen-identical by construction (same call,
same `REGISTRY_SIZE=32` alloc, same accessors — only the symbol name
differs), so this is a no-op on the hot paths and the A/B is a formality:

| benchmark                | 2.3.13 | 2.3.14 | Δ |
|--------------------------|-------:|-------:|---|
| `total_memory_13dev`     | 142 ns | 142 ns | identical (deterministic) |
| `has_accelerator_13dev`  |  28 ns |  28 ns | identical (deterministic) |
| `count_family_gpu_13dev` | 294 ns | 294 ns | identical (deterministic) |
| `plan_70B_bf16_4gpu`     | 1 816 ns | 1 886 ns | neutral (noise) |

**0 wins claimed, 0 regressions** — the deterministic min=max=avg registry
benches are byte-identical, which is the proof a pure rename didn't perturb
codegen. Full gate: **12 units / 0 failed**, 6/6 fuzz, fmt / lint /
raw-offset guard (accessor names unchanged) / distlib determinism clean.

#### Notes

- **Breaking to the exported surface** (per SemVer, a rename of a public
  symbol) — but `registry_new` had no well-behaved consumer: it resolved
  unpredictably the moment bote was co-linked. Consumers that call it
  directly switch to `hw_registry_new`; szal/mihi/hoosh reach the registry
  via `registry_detect` / `cached_registry_new`, which are unaffected.
- **Closes the 2026-06-11 issue** and the sibling half of 2026-06-23. With
  2.3.13 (`HWA_ERR_*`) + 2.3.14 (`hw_registry_new`), ai-hwaccel's bundle no
  longer collides with sakshi/yukti/bote/sigil on any exported symbol.

## [2.3.13] — 2026-07-13

**Toolchain bump to cyrius 6.4.62 + `DetectionError` enum namespacing.**
Moving the pin two minors (6.2.11 → 6.4.62) does two things: it delivers
a broad codegen win on the non-trivial hot paths (parse/plan/JSON, up to
−37%), and — because 6.4.62's linker now *reports* duplicate global
symbols — it surfaces a pre-existing collision that was silent under
6.2.11. `sakshi` (the logging library, present in **every** build via
`src/log.cyr`) defines a bare `ERR_TIMEOUT = 5`; our `DetectionError`
defined a bare `ERR_TIMEOUT = 3`. Cyrius enum members are *global*
constants (the enum name does not namespace them), so the two squatted on
the same symbol under last-definition-wins. This release prefixes the
**entire** enum `ERR_* → HWA_ERR_*`, which is the ai-hwaccel-owned fix
from the 2026-06-23 collision issue — pulled ahead of its filed 2.4.0
target now that 6.4.62 promotes it from a downstream-only warning to an
in-tree one. **Enum values are unchanged** (`HWA_ERR_TIMEOUT` is still
`3`); the bare names already resolved unpredictably whenever sakshi was
linked, so no well-behaved consumer could depend on them — this is a
de-facto bugfix to the symbol surface, not a usable-API break.

#### Changed

- **cyrius pin `6.2.11` → `6.4.62`** (`cyrius.cyml`). `./lib/`
  (gitignored) re-synced from the 6.4.62 stdlib snapshot via
  `cyrius lib sync` — the declared `[deps].stdlib` subset (37 files);
  every `include`d module verified byte-identical to
  `~/.cyrius/versions/6.4.62/lib/`. No drift warning: manifest pin now
  matches the wrapper.
- **`DetectionError` enum `ERR_* → HWA_ERR_*`** (`src/error.cyr`): all six
  members — `HWA_ERR_{NONE,TOOL_NOT_FOUND,TOOL_FAILED,TIMEOUT,PARSE,SYSFS_READ}`
  — plus every internal reference in `warning_new` / `warning_*` /
  `warning_print`, and the three test units that assert on them
  (`foundation_test`, `io_test`, `gpu_parser_test`). Values preserved
  1:1. **No bare aliases kept** — an alias `ERR_TIMEOUT = 3` would
  reintroduce the exact sakshi collision. `dist/ai-hwaccel.cyr`
  regenerated (delta is the rename only; byte-deterministic).
- **`VERSION`** 2.3.12 → 2.3.13; **`dist/ai-hwaccel.cyr`** regenerated
  (embeds 2.3.13).

#### Fixed

- **`scripts/bench-history.sh` CSV parser** — 6.4.x `bench_report` prints
  **decimal** units (`19.460us`, `1.712ms`) where 6.2.x truncated to a
  bare integer (`19us`). The old `grep -oP '\d+(?=us avg)'` captured the
  *fractional* digits (`460`) and ×1000'd them, writing `460000` for a
  ~19.5 µs bench. Replaced with an awk converter that reconstructs exact
  integer nanoseconds from `<int>ns` and `<major>.<3-frac><us|ms|s>`
  (handles the old integer form too). Without this, every µs-scale row in
  the audit trail would have been silent garbage from this version on.
- **`duplicate symbol 'ERR_TIMEOUT'` build warning** on `src/main.cyr`
  under 6.4.62 — resolved by the `HWA_ERR_*` namespacing above. Clean
  build, no warnings.

#### Performance

**Metric:** raw `bench_avg_ns` / `bench_min_ns`, min-of-8 runs, `CYRIUS_DCE=1`,
same machine, identical source (the `HWA_ERR_*` rename is codegen-neutral —
`build/ai-hwaccel` is 330 424 bytes on both 6.2.11 and 6.4.62). A/B is the
pure toolchain delta. `min_ns` shown (least noise); µs-scale figures are
below the old `bench_report` truncation, hence the ns harness.

_Parsing_

| benchmark            | 6.2.11 | 6.4.62 | Δ |
|----------------------|-------:|-------:|---|
| `parse_cuda_8gpu`    | 26 610 ns | 16 832 ns | **−36.7%** |
| `parse_vulkan_2gpu`  |  3 422 ns |  2 794 ns | **−18.4%** |
| `parse_neuron_2dev`  |  2 305 ns |  1 886 ns | **−18.2%** |
| `detect_safetensors` |    838 ns |    907 ns | neutral (≈0.9 µs timer floor) |
| `detect_gguf`        |    907 ns |    907 ns | neutral (floor) |

_Registry_

| benchmark                | 6.2.11 | 6.4.62 | Δ |
|--------------------------|-------:|-------:|---|
| `plan_70B_bf16_4gpu`     |  2 793 ns |  1 816 ns | **−35.0%** |
| `json_training`          |  2 724 ns |  2 375 ns | **−12.8%** |
| `json_system_io`         |  5 098 ns |  4 749 ns | **−6.8%** |
| `json_plan`              | 18 368 ns | 17 530 ns | **−4.6%** |
| `json_serialize_13dev`   | 20 743 ns | 19 974 ns | **−3.7%** |
| `json_summary_13dev`     |  3 213 ns |  3 212 ns | neutral |
| `best_available_13dev`   |    907 ns |    907 ns | neutral (floor) |
| `count_family_gpu_13dev` |    288 ns |    289 ns | neutral (batch, ±1 ns) |
| `total_memory_13dev`     |    135 ns |    135 ns | neutral (batch) |
| `has_accelerator_13dev`  |     27 ns |     27 ns | neutral (batch) |

**8 wins (3.7%–36.7%), 7 neutral, 0 regressions.** The two apparent
regressions seen at 3 runs (`total_memory` 135→140, `json_serialize`) were
run-to-run noise — min-of-8 shows them equal/improved. CSV audit trail:
`bench-history.csv` (commit `4fad379`). Full gate re-run on 6.4.62: **594
assertions / 12 units, 0 failed**; 6/6 fuzz; vet / lint / fmt / raw-offset
guard / distlib determinism all clean.

#### Notes

- **Scope vs. the 2026-06-23 issue.** That issue bundles two renames —
  the `DetectionError` enum *and* `registry_new → hw_registry_new` — under
  a single 2.4.0. Only the **enum** ships here (6.4.62 forced its hand);
  `registry_new → hw_registry_new` follows in **2.3.14** (next release).
  Sibling
  bare `ERR_*` (`HWA_ERR_NONE/PARSE/TOOL_NOT_FOUND/…`) that collide
  downstream with yukti/bote/sigil rather than sakshi are prefixed here
  too, so the whole enum is namespaced in one pass.
- **Consumers** (hoosh, mihi, daimon, Irfan, AgnosAI, murti, tazama): if
  you referenced ai-hwaccel's `ERR_*` constants by name from the bundle,
  switch to `HWA_ERR_*`. Integer values are identical, so any code keyed
  on the numeric warning code is unaffected.

## [2.3.12] — 2026-06-15

**`--data-dir` flag — data-file resolution that works on Windows PE.**
Follow-up to the 2.3.11 finding: the bundled binary's `--version` /
`--cost` resolve `VERSION` and `data/cloud_pricing.json` via
`AI_HWACCEL_DATA_DIR`, but that env var is a **silent no-op on PE** —
`cmd_getenv` reads `/proc/self/environ` (absent on Windows) and cyrius
6.2.11 exposes no `GetEnvironmentVariable` reroute, so on the Windows
wheel those commands fell back to cwd-relative and reported `unknown` /
empty recommendations. The roadmap's planned fix (a `GetEnvironmentVariable`
path in `cmd_getenv`) is **blocked on a cyrius toolchain reroute** we
can't add. Instead this routes the data dir through the **command line**,
which *is* readable on PE (argv comes from `GetCommandLineW` /
`CommandLineToArgvW`, already wired — it's how `--version` itself works on
cass).

#### Added

- **`--data-dir <path>` CLI flag** (`src/detect/command.cyr`). New
  `cmd_data_dir_arg()` scans argv (self-contained — `argv`/`argc`/`streq`,
  no dependency on `main.cyr`'s parser, so it works before main's flag
  parsing and on PE). `data_file_path` resolution is now **flag → env →
  cwd**: the flag is the portable channel; `AI_HWACCEL_DATA_DIR` is kept
  for back-compat (Linux/macOS) but superseded by the flag where both are
  present. Listed in `--help`.
- **Test — `bindings/python/tests/test_bundled.py`**: `--data-dir` resolves
  `VERSION` with the env var stripped and a foreign cwd (the PE-faithful
  path), and the runner is asserted to place `--data-dir` on the argv.

#### Changed

- **Python `_runner._run`** now passes `--data-dir <bundled _bin>` on the
  argv for the bundled binary (still exporting `AI_HWACCEL_DATA_DIR` for
  back-compat), so `version()` / `cost()` work on the Windows wheel — not
  just where the caller's cwd happens to hold the data files. Preserves the
  existing "never override a caller-set `AI_HWACCEL_DATA_DIR`" semantics.
- **`VERSION`** 2.3.11 → 2.3.12; **`dist/ai-hwaccel.cyr`** regenerated.

#### Notes

- **Upstream issue to file (drafted, not filed — cyrius repo is off-limits
  here):** cyrius PE backend exposes no environment-read reroute
  (`GetEnvironmentVariableW` / `GetEnvironmentStringsW`), so neither
  `cmd_getenv` nor stdlib `getenv` can read env on PE. The `--data-dir`
  flag makes ai-hwaccel independent of it; the issue tracks restoring
  env-var parity for any consumer that prefers the env channel. Text in the
  2.3.12 PR description / roadmap.

#### Performance

- **No regression; no hot path touched.** `data_file_path` is called only
  on `--version` / `--cost`, never in `registry_detect` or the JSON/parse
  paths. Baseline vs. changed tree (same machine/iters) within run-to-run
  noise: `total_memory_13dev` 149→140 ns, `count_family_gpu_13dev` 298→287
  ns, `has_accelerator_13dev` 27 ns flat, `json_serialize_13dev` ~23 µs
  flat, `parse_cuda_8gpu` ~32 µs flat. 12/12 cyrius test units + 20/20
  Python tests pass.

## [2.3.11] — 2026-06-15

**Windows PE runtime CI gate.** `wheels.yml` cross-builds the Windows PE
on Linux but previously asserted only the `MZ` magic — a silently-broken
*runtime* feature could ship green. The risk is proven: both 2.3.9
regressions (the DXGI `.rdata` corruption and the dropped PE structured
logging) were caught **only** by manual `ssh cass` smoke, never by CI.
The PE was the one target with no automated runtime coverage. This closes
that gap with a `windows-smoke` job that runs the actual cross-built EXE
on an ephemeral `windows-latest` runner and asserts three runtime
contracts.

#### Added

- **`windows-smoke` job — `.github/workflows/wheels.yml`** (`runs-on:
  windows-latest`, `needs: windows`). Downloads the `wheels-windows`
  artifact, extracts the bundled `_bin/ai-hwaccel.exe` from the wheel, and
  asserts on a real Windows execution:
  - **(a) Detection** — bare `ai-hwaccel.exe` exits 0 and emits
    **well-formed JSON** (`schema_version` + a **non-empty `profiles[]`**;
    a CPU profile is always present even on a GPU-less / `wmic`-less
    runner). Regression gate for the COM/`.rdata` corruption class.
  - **(b) Structured logging** — `ai-hwaccel.exe -vv` writes the
    `[ENTER]`/`[EXIT] detect` span to **stderr** while **stdout stays
    clean**, and the **default level stays silent**. Regression gate for
    the PE var-syscall reroute (sakshi roadmap W1).
  - **(c) VERSION resolution** — `ai-hwaccel.exe --version`, run from the
    `_bin/` dir, reports the **bundled version** (not `unknown`). Confirms
    the wheel layout's `data_file_path` resolution on PE.
- **Venue:** `windows-latest` ephemeral runner — the roadmap's preferred
  option (self-contained, no secrets, no self-hosted wiring). Fallback to
  an `ssh cass` self-hosted leg if the cross-built PE ever fails to *run*
  cleanly on the GH runner.

#### Notes

- The gate matches the **real CLI surface** — flag-based (`ai-hwaccel` =
  full JSON, `-vv` = trace, `--version`), not the `detect <flag>`
  subcommand the roadmap sketch had assumed.
- **PE limitation surfaced by this work (follow-up, not fixed here):**
  `AI_HWACCEL_DATA_DIR` is **unreadable on PE** — `cmd_getenv`
  (`src/detect/command.cyr`) reads `/proc/self/environ`, which doesn't
  exist on Windows — so the bundled binary's `--version`/`--cost` resolve
  data files **cwd-relative** on Windows, *not* via the env var the Python
  `_runner` sets. Detection (the primary path, which spawns `wmic` and
  needs no data files) is unaffected. The gate runs `--version` from
  `_bin/` to match this cwd-relative reality. A Windows
  `GetEnvironmentVariable` path in `cmd_getenv` is filed as a follow-up so
  the documented env-var contract holds on PE too.

#### Performance

- **CI-only change; binary byte-unchanged.** No `src/` touched (workflow +
  `VERSION` + docs only) — `VERSION` is read at runtime, not compiled in —
  so the compiled binary and every hot path are identical to 2.3.10.
  Benchmark suite re-run for the record: within run-to-run noise, **no
  regression.** 12/12 test units pass.

## [2.3.10] — 2026-06-15

**Toolchain bump to cyrius 6.2.11.** Pin update + stdlib re-sync,
proving no regression. The headline of the 6.1.18 → 6.2.11 jump is a
stdlib reorganization: the standalone `lib/json.cyr` module is gone,
folded (with `base64`/`csv`/`u128`/`bigint`/`toml`/`cyml`) into the new
`lib/bayan.cyr` distribution bundle, and its functions are renamed
`json_*` → `bayan_json_*`. ai-hwaccel never called the stdlib JSON
parser — `src/json_out.cyr` is a hand-rolled `str_builder` serializer
and `src/model_format.cyr` does its own byte-level safetensors header
parse — so the `"json"` entry in `[deps] stdlib` was dead weight that
now points at a module that no longer exists. **Dropped it** rather
than re-pointing at `bayan` (which would pull the whole bundle for
functions we don't use; the compiler DCEs it either way, but the dep
list should reflect reality). No `src/` changes.

#### Changed

- **`cyrius.cyml`**: pin 6.1.18 → **6.2.11**. Stdlib re-synced (97
  files). Removed the unused **`"json"`** entry from `[deps] stdlib`
  (module folded into `bayan.cyr` upstream; nothing in `src/` references
  it).
- **`VERSION`** 2.3.9 → 2.3.10; **`dist/ai-hwaccel.cyr`** regenerated.

#### Performance

- **Benchmark-neutral.** Full suite re-run on the 6.2.11 tree against
  the 6.1.18 baseline, same machine/iteration counts. All moves sit
  inside run-to-run noise: the deterministic sub-µs rows that looked
  like a regression on a single pair of runs (`total_memory_13dev`
  141→148 ns) were confirmed as noise across 3× re-runs each — new
  141–157 ns vs baseline 144–149 ns, overlapping ranges;
  `has_accelerator_13dev` and `count_family_gpu_13dev` likewise overlap.
  The µs-resolution rows (`json_serialize_13dev` ~23 µs,
  `json_plan` ~21 µs, `parse_cuda_8gpu` ~35 µs) are flat within their
  max-spike variance. **No regression.** 12/12 test units pass, 6/6
  fuzz harnesses build, binary smoke-tested.

## [2.3.9] — 2026-06-09

**Windows DXGI precise VRAM + structured logging, on cyrius 6.1.18.** ai-hwaccel previously emitted nothing to stderr: every
detector shelled out to `nvidia-smi` / `wmic` / `rocm-smi` and swallowed
failures into the warnings vec, visible only if you parsed the JSON. This
wires the AGNOS `sakshi` structured logger through the whole pipeline
(detectors, planning, cache, async spans) so the tool is observable in the
field — while keeping **stdout byte-clean** for the JSON that consumers
parse. With the **cyrius 6.1.18** bump the logger now reaches stderr on
**Windows PE** as well (previously dropped — see *Changed*), so the feature
ships on all three platforms.

#### Added

- **Structured logging — `src/log.cyr`** over stdlib `sakshi`. All output
  goes to **stderr (fd 2)**; stdout is untouched. Default level **WARN**
  (quiet on success, visible on failure). Raise via the **`AI_HWACCEL_LOG`**
  env var (`off|fatal|error|warn|info|debug|trace`) or the
  **`--log-level <name>`** / **`-v`** (debug) / **`-vv`** (trace) /
  **`-q`** (off) CLI flags. Process tagged with `getpid()` as the sakshi
  trace id for log correlation. `hwlog_*_n` message builders and span
  wrappers are level-guarded so nothing allocates (and spans don't emit)
  below the active level.
- **Pipeline instrumentation** at operation granularity (never inside a
  per-iteration parse loop): `registry_detect_with_opts` and
  `registry_detect_threaded_with` get a `detect` span + a profiles/warnings
  summary (info); `cached_get` / `disk_cached_get` log hit/miss (debug) and
  disk write-failure (warn); `reg_plan_sharding` logs the model size (debug)
  and warns when no accelerator is available; the `cuda` / `gaudi` CSV parse
  failures — previously silent — now emit a `warn`. Default-level runs stay
  silent on success.

#### Changed

- **Toolchain pin 6.1.5 → 6.1.18** (stdlib re-synced, 94 files; sakshi
  v2.2.6 → **v2.2.10**, new `fs_win.cyr` Windows `dir_list` port). The
  **compiler** bump is a codegen no-op — cycc 6.1.15 and 6.1.18 emit a
  byte-identical 370,776 B Linux binary (same as 6.1.5). The synced
  **stdlib** adds +16 B (370,776 → 370,792 B), entirely sakshi 2.2.10 source
  linked through `log.cyr`. 13/13 test units (606 assertions) pass;
  `fmt`/`lint`/`vet`/
  distlib-determinism clean; bench delta within noise (bench-history.csv).
- **Windows PE structured logging fixed (cyrius 6.1.18).** Logging was
  silently dropped on PE: sakshi holds syscall numbers in `var` slots and
  cyrius's PE syscall reroute only fired for compile-time-literal numbers,
  so `syscall(_SK_SYS_WRITE, …)` fell through to a non-functional raw
  `0F 05` (no fault, exit 0). 6.1.18 resolves a `var`-held syscall number to
  its constant, so `n=1` (write) routes to `WriteFile`. **Verified on cass**
  (Windows 10.0.26200): `detect -vv` emits the full `detect` span on stderr;
  default WARN stays silent; stdout stays byte-clean. (`getpid` is still
  unrouted on PE, so the trace-id prefix shows `[0]` — cosmetic; delivery is
  unaffected.)
- **Windows wheel cross-build fixed (CI).** `stage_win_cross.sh` now builds
  the PE with **`cyrius build --win`** (the Linux ELF `cycc` emits PE32+
  natively) instead of piping a hand-synthesized translation unit to the
  standalone `cycc_win`. `cycc_win` is itself a **Windows PE**, so on a Linux
  host it only runs under Wine (the local `DOSWin` binfmt handler); a bare CI
  runner (`ubuntu-latest`, no Wine) failed it with `cannot execute binary
  file: Exec format error` (exit 126). `--win` is the correct Linux-native
  cross path — no Wine, deps resolved by the wrapper. PE re-verified on cass
  (`--version` 2.3.9, DXGI JSON, `-vv` logging).
- **`cyrius.cyml`**: **`sakshi`** added to `[deps] stdlib`; **`src/log.cyr`**
  added to `[lib] modules` (so the consumer bundle carries it).
- **Bundle consumers** (mihi et al.): `dist/ai-hwaccel.cyr` now references
  `sakshi_*`, so consumers must add **`sakshi`** to their own
  `[deps] stdlib`. The bundle's "unresolved symbols" note at distlib time
  is expected — stdlib is consumer-supplied.

#### Performance

- **Logging adds nothing measurable.** Full-instrumentation tree vs
  6.0.70-without-logging, interleaved: `parse_cuda_8gpu` min **28 µs**
  (identical), `total_memory_13dev` 139→140 ns, `has_accelerator_13dev`
  28→28 ns, `count_family_gpu_13dev` 292→281 ns, `json_serialize_13dev`
  ~22–23 µs — **all within noise.** Instrumentation lives at dispatch
  sites and error paths, off the benched hot loops; below-threshold log
  calls build no messages. Binary grows **301 KB → 369 KB** (+68 KB, the
  reachable `sakshi` surface; DCE NOPs 623 unreachable fns). At the 6.1.18
  pin the stripped binary is **370,792 B**; 13/13 test units (606
  assertions) pass.

## [2.3.8] — 2026-06-05

**Toolchain bump to cyrius 6.0.70.** Pin update + stdlib re-sync, proving
no regression. Groundwork for the 2.3.9 Windows DXGI precise-VRAM work:
6.0.70 lands the *foundation* (`callptr`/`IR_CALL_INDIRECT`, the
`dxgi.dll!CreateDXGIFactory1` import, COM-vtable dispatch capability —
`CreateDXGIFactory1` returns `S_OK` on cass), but `callptr` to a *real*
Win64 COM callee (`EnumAdapters`/`GetDesc` over the DXGI vtable — the
actual VRAM read) corrupts the caller frame on cass and is fixed upstream
only in **6.0.71** (cyrius issue
`2026-06-05-windows-com-vtable-real-callee-frame-corruption.md`). DXGI
precise VRAM therefore stays deferred to 2.3.9; Windows GPU VRAM remains
on the WMI `AdapterRAM` path.

#### Changed

- **`cyrius.cyml`**: pin 6.0.54 → **6.0.70**. Stdlib re-synced (89 files).
  **CLAUDE.md** pin → 6.0.70.
- **`VERSION`** 2.3.7 → 2.3.8; **`dist/ai-hwaccel.cyr`** regenerated.

#### Performance

- **cyrius 6.0.64 global allocator spinlock — accepted, justified.**
  6.0.70's `lib/alloc.cyr` serializes the bump pointer behind a CAS
  spinlock (a heap-corruption fix: concurrent `alloc()` across real
  threads was overlap-allocating). This costs ~5–10 % on allocation-heavy
  paths — `parse_cuda_8gpu` min **16→28 µs** is the worst case. **It is not
  a regression to fix but a correctness fix we need**: `async_detect.cyr`
  spawns real threads (`thread_create`/`thread_join`) that all `alloc()`
  while parsing detector output; without the lock that path corrupts the
  heap. Single-threaded callers pay the tax; the threaded path stops being
  unsound. Accepted per the no-regression rule's "explicitly justified"
  clause. Otherwise neutral: `total_memory_13dev` 137→139 ns,
  `has_accelerator_13dev` 27→28 ns, `json_serialize_13dev` ~22 µs flat —
  **no other regression.** 12/12 test units pass.

## [2.3.7] — 2026-06-03

**Windows x86_64 wheel ships with real CPU + GPU detection —
`pip install ai-hwaccel` is now self-contained on Windows.** Closes the
roadmap 2.3.7 gate. The Windows PE blockers were fixed upstream (cyrius
6.0.50 unfroze `cycc_win`; 6.0.51 routed Win32 process creation via
CreateProcessW), and this release implements the Windows-side detection
that turns a running-but-blind binary into a useful one. Verified
end-to-end on `cass` (Windows 11): real total RAM + the actual GPU
(Intel UHD Graphics 600) are detected.

#### Added

- **Windows x86_64 wheel**
  (`ai_hwaccel-2.3.7-py3-none-win_amd64`). **Cross-built on Linux** —
  `cycc_win` is a Linux-hosted compiler that emits PE32+, so no Windows
  runner is needed (cass is used only for runtime smoke). New
  `bindings/python/scripts/stage_win_cross.sh` synthesizes the
  translation unit (`[deps] stdlib` + `src/main.cyr`) and pipes it to
  `cycc_win`; `wheels.yml` `windows` job **enabled**, now on
  `ubuntu-latest`.
- **Windows GPU detection** — `src/detect/windows.cyr` (was a stub) now
  spawns `wmic path win32_VideoController` and emits one **`ACCEL_WIN_GPU`**
  ("Windows GPU") profile per controller with device name + VRAM. New
  `BACKEND_WINDOWS` backend (gated into `registry_detect_with` under
  `#ifdef CYRIUS_TARGET_WIN`). The Unix detectors still run, so
  `nvidia-smi.exe` continues to give precise NVIDIA VRAM where present.
- **Windows CPU RAM detection** — `detect_system_memory()` gained a
  `#ifdef CYRIUS_TARGET_WIN` branch spawning `wmic computersystem get
  TotalPhysicalMemory`, replacing the 16 GiB fallback (`/proc/meminfo`
  doesn't exist on Windows) with real total memory.
- **`tests/tcyr/windows_test.tcyr`** — Linux-hosted fixture tests for the
  pure parsers (`win_parse_videocontrollers`, `win_parse_total_memory`):
  single/multi GPU, CRLF, key-order, empty, absent. The parsers are
  ungated so they're testable without a Windows host. 12 test units total.

#### Changed

- **`cyrius.cyml`**: pin 6.0.43 → **6.0.54**. Stdlib re-synced (87 files).
- **`build_wheel.sh`** clears `build/` before packaging, and
  `stage_*.sh` drop the foreign-platform binary, so a per-platform wheel
  bundles only its own binary (a Linux ELF was leaking into the
  win_amd64 wheel via setuptools' build cache).
- **`pyproject.toml`** 2.3.6 → 2.3.7; **`VERSION`** 2.3.6 → 2.3.7;
  **`dist/ai-hwaccel.cyr`** regenerated; **`CLAUDE.md`** pin → 6.0.54.

#### Performance

No hot-path `.cyr` changed (the new code is Windows-gated + the keyed
`accel_*` functions gained one branch each). 6.0.54 + feature vs 6.0.47
baseline, same box/iters: `total_memory_13dev` 141 vs 143,
`count_family_gpu_13dev` 300 vs 303, `has_accelerator_13dev` 28 vs 28,
`json_serialize_13dev` ~23 µs flat — **neutral, no regression.**

#### Known limits / follow-ups

- **VRAM caps at 4 GiB on non-NVIDIA GPUs** —
  `Win32_VideoController.AdapterRAM` is a 32-bit field. Native DXGI
  (`EnumAdapters1`, 64-bit `DedicatedVideoMemory`, no subprocess) is the
  2.3.8 precision upgrade, gated on cyrius PE COM-vtable + dxgi.dll IAT
  support (filed:
  `cyrius/docs/development/issues/2026-06-03-windows-pe-com-vtable-dxgi-for-gpu-enum.md`).
- On Windows the irrelevant Unix probes (`system_profiler`, etc.) still
  emit "tool not found" warnings — cosmetic; keeping the probes enabled
  is what lets `nvidia-smi.exe` work when present.

## [2.3.6] — 2026-06-02

**macOS arm64 wheel ships — `pip install ai-hwaccel` now self-contained
on Apple silicon.** This closes the roadmap 2.3.6 gate. Two things had
to land together: the toolchain pin moves to **6.0.43**, and the Darwin
build bug that blocked the wheel on 6.0.38 is fixed upstream.

The 2.3.5 notes left macOS gated on "no compiler in the installer."
6.0.38 delivered the arm64 Darwin compiler but surfaced a *new* blocker:
`cyrius build` false-negatived its own install check on a manifest with
`cyrius = "<pin>"` + a `[deps] stdlib` block (our exact shape), claiming
the present snapshot lib was "not installed." Filed as cyrius issue
`2026-06-02-macos-arm64-deps-stdlib-pin-check.md` (three stacked
Darwin-ABI defects: an `is_dir` install-probe false-negative, a getcwd
SIGSYS, and a wrong Darwin `st_size` offset truncating the include
scan). **Fixed across cyrius 6.0.40–6.0.43**; verified end-to-end on
`ecb` (real Apple silicon): the arm64 Mach-O builds, the resolved stdlib
executes, and every ai-hwaccel subcommand runs.

#### Added

- **macOS arm64 wheel** (`ai_hwaccel-2.3.6-py3-none-macosx_11_0_arm64`).
  Built on `ecb` via `bindings/python/scripts/build_remote.sh ecb
  macosx_11_0_arm64`, bundling the Mach-O arm64 binary + `VERSION` +
  `data/cloud_pricing.json` under `ai_hwaccel/_bin/` (subprocess + JSON,
  no FFI — same model as the Linux wheels). Verified: `--version`,
  `--json`, `--summary`, `--cost`, `--plan` all run on Darwin arm64.
- CI `wheels.yml` `macos` job **enabled** (`if: true`) — the `macos-14`
  runner installs pinned 6.0.43 and builds natively.

#### Changed

- **`cyrius.cyml`**: pin 6.0.30 → **6.0.43**. Stdlib re-synced into
  `./lib/` (82 files from the 6.0.43 snapshot); drift warning gone.
- **`bindings/python/scripts/build_remote.sh`**: `cyrius lib sync` is now
  best-effort. It false-negatives on Darwin (the directory-listing /
  `getdents64` surface is still unported there — a separate, still-open
  item in the same cyrius issue), and it is unnecessary: `cyrius build`
  resolves `[deps] stdlib` into `./lib` by name (the path fixed in
  6.0.40+). The build populates its own lib.
- **`bindings/python/scripts/stage_binary.sh`**: fixed an "unbound
  variable" crash on the `macos-14` CI runner — the native staging path
  leaves `EXTRA_ARGS` empty, and bare `"${EXTRA_ARGS[@]}"` is a `set -u`
  error under the runner's bash 3.2. Switched to
  `${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}` (expands to nothing on an empty
  array, no spurious empty argv). Verified both the native and
  `--aarch64` paths.
- **`bindings/python/pyproject.toml`**: 2.3.4 → 2.3.6.
- **`CLAUDE.md`**: pinned-version notes 6.0.0/6.0.30 → 6.0.43.
- **`VERSION`**: 2.3.5 → 2.3.6; **`dist/ai-hwaccel.cyr`** regenerated
  (embeds 2.3.6).

#### Performance

No `.cyr` source changed; any delta is codegen-only via the re-synced
stdlib. Built on 6.0.43 vs the 6.0.38 lib snapshot, same machine and
iteration counts. The deterministic ns rows drift run-to-run on this box
(`total_memory_13dev` 145–163 ns, `count_family_gpu_13dev` 303–354 ns,
`has_accelerator_13dev` 30–42 ns) and the 6.0.38 baseline (159 / 354 /
33 ns) sits inside that band — **neutral, within noise, no regression.**
µs-level JSON/parse paths flat within jitter. (This subsumes the
unreleased 6.0.30 → 6.0.38 pin step, which was also neutral.)

#### Compatibility

- 11/11 test units green on 6.0.43 (Linux). macOS arm64 verified by
  running the staged binary on `ecb`.
- **Known cosmetic warning (macOS only):** the Darwin `[deps]` build
  emits `duplicate fn 'arena_new' (last definition wins)` because both
  `lib/alloc.cyr` and `lib/alloc_macos.cyr` define it and Darwin pulls
  both layers. "Last definition wins" selects the macOS variant —
  correct behavior; the binary is verified working. Filed upstream (P4)
  at `cyrius/docs/development/issues/2026-06-02-macos-alloc-arena-duplicate-fns.md`;
  not a consumer blocker.
- **Still pending:** Windows wheel (2.3.7) — needs the PowerShell build
  flow + a PE `cyrius build` target.

## [2.3.5] — 2026-06-01

**Toolchain pin 6.0.25 → 6.0.30.** Resolves the long-standing wrapper
drift (the local toolchain had auto-advanced well past the pin). Pure
toolchain bump — no `.cyr` source changed. Per the mandatory-benchmark
policy, before/after deltas confirm no regression.

#### Changed

- **`cyrius.cyml`**: pin 6.0.25 → 6.0.30. Stdlib re-synced into `./lib/`
  (82 files from the 6.0.30 snapshot). Build drift warning gone.
- **`VERSION`**: 2.3.4 → 2.3.5.
- **`dist/ai-hwaccel.cyr`** regenerated (embeds 2.3.5).
- **`CLAUDE.md`**: pinned-version note 6.0.25 → 6.0.30.

#### Performance

Before/after (min-of-6 @ 2000 iters, ns; built with the respective
pinned toolchain):

| benchmark              | 6.0.25    | 6.0.30    | delta |
| ---------------------- | --------: | --------: | ----- |
| `json_serialize_13dev` | 25535 ns  | 25245 ns  | −1.1% (noise) |
| `json_summary_13dev`   |  5051 ns  |  5025 ns  | flat (noise) |
| `parse_cuda_8gpu` (min)|    18 µs  |    17 µs  | flat (noise) |

No regression; all within run-to-run noise. The compiled binary is
functionally unchanged.

#### Compatibility

- **Gates** (on 6.0.30): 11 test units, 6 fuzz harnesses, `vet` (37
  deps, 0 untrusted), fmt, lint, raw-offset guard, distlib
  drift+determinism — all green.
- **Note**: macOS toolchain still incomplete — the 6.0.30 installer
  sets up the version layout + `cyriusly` on Darwin but does not yet
  deliver the `cyrius`/`cycc` compiler binaries, so the macOS wheel
  (2.3.6) remains blocked. Windows wheel (2.3.7) still pending the
  PowerShell build flow.

## [2.3.4] — 2026-06-01

**Linux wheels + the wheel-build machinery.** `bindings/python` can now
produce platform-tagged wheels that bundle a prebuilt, statically-linked
`ai-hwaccel` binary (+ `VERSION` + `data/`), so `pip install` gives a
self-contained package (subprocess + JSON; no FFI). Linux x86_64 +
aarch64 ship and are validated. **macOS and Windows are deferred to 2.3.5,
gated on cyrius toolchain support** (see below) — no `.cyr` changed, so
the core binary is identical to 2.3.3.

#### Added

- **Wheel machinery** (`bindings/python/`):
  - `setup.py` — `BinaryDistribution` + a `bdist_wheel` override that
    emits a `py3-none-<platform>` wheel (python-agnostic, platform-
    specific) with the tag overridable via `AIH_WHEEL_PLAT`.
  - `scripts/build_wheel.sh <plat-tag>` — package the staged `_bin/`.
  - `scripts/build_remote.sh <host> <plat-tag>` — ship source to an SSH
    host, `lib sync` + build there, retrieve the binary (for platforms
    with no local cross-compiler, e.g. macOS once supported).
  - `pyproject` package-data now bundles `_bin/{ai-hwaccel[.exe],
    VERSION, data/*}`.
- **`.github/workflows/wheels.yml`** — CI matrix (workflow_dispatch +
  tag): linux (x86_64 + aarch64) active; macOS / Windows jobs scaffolded
  but **gated `if: false`** pending toolchain support.

#### Shipped wheels

- `ai_hwaccel-2.3.4-py3-none-manylinux2014_x86_64.whl` — built,
  installed in a clean venv, and run from an unrelated cwd:
  `version`/`detect`/`cost` all work, bundled binary preserved `+x`
  (0o755), data resolved via `AI_HWACCEL_DATA_DIR`.
- `ai_hwaccel-2.3.4-py3-none-manylinux2014_aarch64.whl` — cross-built
  (static ARM binary) and packaged (run-validated in CI/on-device only).

#### Deferred (gated on cyrius toolchain)

- **macOS arm64 wheel (2.3.5, near-term)** — the Darwin binaries are
  built but not yet integrated into `install.sh` (it still rejects
  darwin). Unblocks the moment the installer fix lands.
- **Windows x86_64 wheel (2.3.6, later in 6.0.x)** — needs a full
  PowerShell (`.ps1`) build flow; the PE backend (`cycc_win`) is frozen
  at `cc5_win 5.11.69` and there's no `cyrius build --win` target.
  Both CI jobs are scaffolded and flip on with a one-line `if:` change
  when their toolchain support lands — same "gate on upstream readiness"
  posture as the WASM/JS roadmap item.

#### Changed

- **`VERSION`**: 2.3.3 → 2.3.4; Python package + `__version__` track it.
- **`dist/ai-hwaccel.cyr`** regenerated (embeds 2.3.4); no `.cyr` source
  changed, so the binary is otherwise identical to 2.3.3.

#### Compatibility

- **cyrius core / CLI**: unchanged. No benchmarked path touched (no
  `.cyr` edits) — bench policy satisfied trivially.

## [2.3.3] — 2026-06-01

**Working-directory-independent data files.** Closes the 2.3.2 known
limitation: the binary read `VERSION` (`--version`) and
`data/cloud_pricing.json` (`--cost`) relative to the current working
directory, so a binary invoked from elsewhere (a pip-installed wheel)
returned `"unknown"` / no cost recommendations. The binary now honors an
`AI_HWACCEL_DATA_DIR` env var to locate them, and the Python wrapper sets
it automatically for the bundled binary. Cross-platform and
language-neutral — any consumer (agnos later) sets one env var.
Multi-platform wheel CI is 2.3.4.

#### Added

- **`AI_HWACCEL_DATA_DIR` resolution** (`src/detect/command.cyr`
  `data_file_path`). When set, `VERSION` and `data/cloud_pricing.json`
  resolve under it (`<dir>/<rel>`); unset → cwd-relative (unchanged
  fallback). Used by `_print_version` (`src/main.cyr`) and
  `load_cloud_instances` (`src/cost.cyr`).
- **`bindings/python` wheel-bundling prep**:
  - `_runner._run` sets `AI_HWACCEL_DATA_DIR` to the bundled `_bin/` dir
    when the bundled binary is used (never overriding a caller's value;
    PATH/explicit binaries are left to the caller's environment).
  - `scripts/stage_binary.sh` — builds the binary with the pinned
    toolchain and stages `_bin/{ai-hwaccel, VERSION, data/cloud_pricing.json}`
    (`--aarch64` supported). Foundation for the 2.3.4 wheel build.
  - `tests/test_bundled.py` — 3 tests proving `version()` / `cost()` work
    from a foreign cwd via the bundled binary (skip if not staged).
    Python suite now 18 tests.

#### Fixed

- `version()` / `cost()` no longer depend on the caller's working
  directory when using the bundled binary (the 2.3.2 known limitation).

#### Changed

- **`VERSION`**: 2.3.2 → 2.3.3; Python package + `__version__` track it.
- **`dist/ai-hwaccel.cyr`** regenerated (embeds 2.3.3) by the now
  dist-aware `scripts/version-bump.sh`.

#### Performance

- No benchmarked path touched — `data_file_path` is on the
  `--version` / `--cost` startup paths only, not the JSON/parse hot
  paths. `json_serialize_13dev` 25433 → 24520 ns (within noise),
  confirming no regression (min-of-5 @ 2000 iters).

#### Compatibility

- **CLI**: additive. With `AI_HWACCEL_DATA_DIR` unset, behavior is
  byte-identical to 2.3.2 (cwd-relative). New env var is opt-in.
- **Next**: 2.3.4 — multi-platform wheel CI (manylinux x86_64/aarch64,
  macOS universal2, Windows) using `stage_binary.sh` per target.

## [2.3.2] — 2026-06-01

**Python bindings (`bindings/python/`).** A thin, dependency-free Python
package over the compiled binary + schema-v4 JSON contract from 2.3.1.
There is no FFI (the cyrius toolchain emits executables only); each call
shells out to the binary and parses its JSON into typed dataclasses.
**No cyrius source changed** — the binary is byte-identical to 2.3.1, so
the core benchmarks are unaffected (verified by rebuild; the
mandatory-benchmark policy is satisfied trivially with no core perf
surface touched). Multi-platform wheels are 2.3.3.

#### Added

- **`bindings/python/` package** (`ai_hwaccel`, GPL-3.0-only,
  zero runtime deps, Python ≥ 3.8):
  - **Typed model** (`models.py`) for the full schema-v4 surface:
    `Registry`, `AcceleratorProfile`, `SystemIo`, `Interconnect`,
    `StorageDevice`, `RuntimeEnvironment`, `ShardingPlan`, `ModelShard`,
    `TrainingMemory`, `CostReport`, `CostRecommendation`. Lossless field
    mapping; fixed-point `*_x1000` values surfaced via convenience
    properties (`est_tokens_per_sec`, `total_gib`, `price_per_hour_usd`).
  - **API**: `detect()`, `summary()`, `plan()`, `training_memory()`,
    `cost()`, `version()` — each accepts `binary=` and `timeout=`.
  - **Binary discovery** (`_runner.py`): explicit arg → `AI_HWACCEL_BIN`
    → bundled `_bin/` (2.3.3) → `PATH`. Clear `BinaryNotFoundError` /
    `CommandError`.
  - **Optional pandas export**: `Registry.to_dataframe()` (extra
    `ai-hwaccel[pandas]`); clean `ImportError` when pandas is absent.
  - **`pyproject.toml`** (setuptools, src layout; `pandas` + `test`
    extras), package **README**, and a **`unittest`** suite: 9
    fixture-based model tests + 6 e2e tests against the real binary
    (15 total, all passing locally).
- **`.gitignore`**: Python artifacts (`__pycache__/`, `*.pyc`,
  `*.egg-info/`, `bindings/python/{dist,build}/`, the build-time
  `_bin/`).

#### Known limitations (tracked for 2.3.3)

- The binary reads `VERSION` and `data/cloud_pricing.json` **relative to
  the current working directory**. Run from outside the repo root,
  `version()` returns `"unknown"` and `cost()` yields no recommendations;
  detection is unaffected. The 2.3.3 packaging work bundles these next to
  the binary and resolves them relative to the executable.
  `ai_hwaccel.__version__` always reflects the package version.

#### Changed

- **`VERSION`**: 2.3.1 → 2.3.2 (Python package version tracks it).
- **`dist/ai-hwaccel.cyr`** regenerated so its embedded `# Version:`
  header matches 2.3.2 (content otherwise unchanged from 2.3.1).

#### Fixed

- **CI "distlib drift" gate on version bumps.** The dist bundle embeds
  the project version in its header, so a `VERSION` bump alone makes it
  stale even with no `.cyr` change — CI regenerates with the new version
  and the diff fails. `scripts/version-bump.sh` now **auto-regenerates
  `dist/ai-hwaccel.cyr`** using the *pinned* toolchain
  (`~/.cyrius/versions/<pin>/bin/cyrius`, matching what CI installs)
  after writing `VERSION`. Confirmed distlib output is byte-identical
  across cyrius 6.0.25/6.0.27, so the local wrapper drift is not a
  factor — only the missing regen was.

#### Compatibility

- **cyrius core / CLI**: unchanged (no `.cyr` edits; the regenerated
  bundle differs from 2.3.1 only in the version header line).
- **Next**: 2.3.3 — multi-platform wheels (manylinux x86_64/aarch64,
  macOS universal2, Windows), bundling the per-target binary + data
  files; CI matrix.

## [2.3.1] — 2026-06-01

**JSON surface extension — the data layer for language bindings.** The
roadmap's "Ecosystem" work (Python bindings + packaging) needs the full
detection surface reachable as JSON; today only `AcceleratorProfile`s
were serialized. This release bumps the JSON schema to **v4** and makes
`SystemIo`, `Interconnect`, `StorageDevice`, runtime environment,
`ShardingPlan`, and the training-memory estimate reachable as JSON, plus
new CLI modes to emit them. The compiled binary + this JSON contract is
the **language-neutral substrate** the Python package (2.3.2) and a
later AgnosAI/agnos-kernel target consume — no consumer-specific logic
in the core. CLI text output is unchanged and backward-compatible.

Per the mandatory benchmarking policy, the before/after deltas are in
the table below and `bench-history.csv`.

#### Added

- **`system_io` in the default registry JSON (schema v3 → v4).**
  `registry_to_json` now appends a `"system_io"` object:
  `interconnects[]` ({kind, name, bandwidth_bytes_per_sec, state}),
  `storage[]` ({name, kind, bandwidth_bytes_per_sec}), and
  `environment` ({is_docker, is_k8s, k8s_namespace, cloud_provider,
  instance_type, region, k8s_gpu_count, k8s_gpu_source} or `null`).
  Bandwidth is normalized to bytes/sec uniformly (interconnects store
  GB/s×1000, storage stores MB/s; both → bytes/sec via ×1e6).
  Serializers: `system_io_to_json` + `_ic_to_json` / `_storage_to_json`
  / `_env_to_json` in `src/json_out.cyr`.
- **`--plan <model>`** → `plan_to_json` of `reg_plan_sharding`:
  {strategy, strategy_count, total_memory_bytes,
  est_tokens_per_sec_x1000 (optional), shards[] {id, layer_start,
  layer_end, device, device_id, memory_bytes}}.
- **`--train <model> [--method <m>]`** → `training_to_json` of
  `estimate_training_memory`: model/optimizer/activation/total emitted
  as both `*_bytes` and the lossless `*_gib_x1000` fixed-point. Default
  method `full`, target GPU. New `training_method_from_str` in
  `src/training.cyr`.
- **`--cost <model> --json`** → `cost_to_json` (`src/cost.cyr`):
  {model, quantization, memory_required_bytes, recommendations[]
  {instance, provider, gpu, gpu_count, total_memory_gb,
  price_per_hour_usd_x100}}. Without `--json`, `--cost` prints the same
  text as before.
- **CLI help** updated for the new flags; shared `_quant_arg` /
  `_parse_model_b` helpers in `src/main.cyr`.
- **Tests**: `tests/tcyr/json_output_test.tcyr` +4 cases (system_io
  populated + null-env, plan, training) — 21 → 36 assertions.
- **Benchmarks**: `benches/registry.bcyr` gains `json_system_io`,
  `json_plan`, `json_training` (ns-resolution, 2000 iters).

#### Performance

Mandatory before/after (min-of-7 @ 2000 iters, ns resolution):

| benchmark              | 2.3.0     | 2.3.1     | delta |
| ---------------------- | --------: | --------: | ----- |
| `json_serialize_13dev` | 24265 ns  | 25433 ns  | +4.8% — inherent cost of the always-present (empty) `system_io` object, not a regression |
| `json_summary_13dev`   |  4947 ns  |  4985 ns  | flat (noise) — summary JSON unchanged |
| `json_system_io` (new) |     —     |  7167 ns  | populated: 2 interconnects + 2 storage + env |
| `json_plan` (new)      |     —     | 21299 ns  | mock 70B sharding plan |
| `json_training` (new)  |     —     |  4069 ns  | 70B full-train estimate |

The `json_serialize_13dev` increase is the new feature's payload
(`system_io` is now always serialized), measured and documented rather
than hidden. All other registry/parsing benches untouched.

#### Changed

- **`VERSION`**: 2.3.0 → 2.3.1.
- **`SCHEMA_VERSION`**: 3 → 4 (`src/units.cyr`); `foundation_test`
  assertion updated.
- **`dist/ai-hwaccel.cyr`** — regenerated (json_out/cost/training in the
  bundle); deterministic.

#### Compatibility

- **CLI binary**: additive. Default JSON gains `system_io` +
  `schema_version: 4`; existing keys unchanged. `--cost` text output is
  byte-identical without `--json`. New flags are opt-in.
- **Library consumers**: the bundle exposes the new serializers; unused
  ones DCE away. Bump `[deps.ai-hwaccel] tag = "2.3.1"`, re-run
  `cyrius lib sync` + `cyrius deps`.
- **Next**: 2.3.2 Python package (`bindings/python/`), 2.3.3
  multi-platform wheels.

#### Aside

- Toolchain pin drift: `cyrius.cyml` pins 6.0.25, wrapper is now 6.0.26.
  Per policy that's its own bench-gated point release; left at 6.0.25
  here (builds run with `CYRIUS_NO_WARN_PIN_DRIFT=1`).

## [2.3.0] — 2026-06-01

**Toolchain modernization + serialization hot-path + a codebase-wide
Str→cstr dedup.** Pins the compiler to cyrius 6.0.25 (was 6.0.0),
re-syncs the vendored stdlib snapshot, and lands two pieces of audit
work: the JSON serializer now emits single-byte structural punctuation
via `str_builder_putc` (a real, benchmarked win on the per-profile
path), and the `alloc + memcpy + NUL` "copy a `Str` slice into an owned
C string" idiom that had been hand-inlined across 11 detectors is
consolidated onto the stdlib `str_cstr`. No CLI behavior change; JSON
output is byte-identical (all 21 `json_output_test` assertions pass
unchanged).

Per the development loop, every claim below is backed by a
before/after benchmark — see the delta table.

#### Changed

- **Toolchain pin: cyrius 6.0.0 → 6.0.25** (`cyrius.cyml`). The wrapper
  was already 6.0.25 (the manifest pin had drifted); this aligns the
  manifest, silences the `toolchain drift` build warning, and makes
  `cyrius lib sync` resolve against the installed snapshot. Stdlib
  re-synced into `./lib/` (82 files) from
  `~/.cyrius/versions/6.0.25/lib/`.
- **`VERSION`**: 2.2.6 → 2.3.0.
- **`dist/ai-hwaccel.cyr`** — regenerated by `cyrius distlib`. Shrinks
  by 76 lines (the detector dedup below); deterministic across re-runs.

#### Performance

- **JSON serialization hot path** (`src/json_out.cyr`). Structural
  punctuation (`{` `}` `[` `]` `,` `"` `:`) now goes through
  `str_builder_putc` — a single `store8` — instead of
  `str_builder_add_cstr`, which pays a `strlen` + `memcpy` round-trip
  per call even for a one-byte literal. Variable-length pieces (keys,
  values, the multi-byte `,"profiles":[` / `true` / `false` literals)
  stay on `add_cstr`. On the 13-device registry benchmark
  (`benches/registry.bcyr`, 2000 iters, min-of-6 at nanosecond
  resolution):

  | benchmark              | 2.2.6 (add_cstr) | 2.3.0 (putc) | delta   |
  | ---------------------- | ---------------: | -----------: | ------- |
  | `json_serialize_13dev` |        26946 ns  |    24602 ns  | **−8.7%** |
  | `json_summary_13dev`   |         5335 ns  |     5297 ns  | −0.7% (noise) |

  The win concentrates on `json_serialize_13dev` because it walks a
  13-element profile array × ~13 fields each — hundreds of single-byte
  appends. `json_summary_13dev` (8 scalar fields, no nested array) has
  little single-byte punctuation to save, so it lands within run-to-run
  noise — reported honestly rather than rounded up.

#### Refactored

- **`Str` → owned C string consolidated onto `str_cstr`** across 11
  detectors (16 call sites). Every site previously hand-wrote
  `var d = str_data(s); var n = str_len(s); var c = alloc(n + 1);
  memcpy(c, d, n); store8(c + n, 0);` — the exact body of the stdlib
  `str_cstr(s)`. Replaced with a single `str_cstr` call, dropping the
  now-dead `str_data` / `str_len` locals:
  - `src/detect/cuda.cyr` — compute_cap, driver_version, device_name (×3)
  - `src/detect/rocm.cyr` — product_name, vbios_version, revision (×3)
  - `src/detect/apple.cyr` — parsed chip name
  - `src/detect/gaudi.cyr` — hl-smi CSV name field (comment updated)
  - `src/detect/vulkan.cyr` — device name
  - `src/detect/intel.cyr` — device name
  - `src/detect/bandwidth.cyr` — sysfs device path
  - `src/detect/interconnect.cyr` — interconnect name + state (×2)
  - `src/detect/environment.cyr` — k8s namespace + AWS instance type (×2)
  - `src/detect/pcie.cyr` — sysfs device path
  - `src/detect/disk.cyr` — block device name
  Behavior is identical (same allocator, same bytes). The DCE build
  shrinks 289584 → 287992 bytes (−1592). Perf-neutral on the parsing
  benchmarks (`parse_cuda_8gpu` min holds at 16–17 µs); confirmed not a
  regression rather than claimed as a win.

#### Tooling

- **`benches/registry.bcyr`** — `json_serialize_13dev` /
  `json_summary_13dev` bumped 100 → 2000 iters and now print an
  explicit `avg_ns=` line. `bench_report`'s human-readable output
  truncates to microseconds, which hides a sub-microsecond delta; the
  raw nanosecond average makes the serialization win measurable.

#### Compatibility

- **CLI binary**: no change — same flags, byte-identical JSON / table /
  summary output.
- **Library consumers (mihi et al.)**: source-compatible. The bundle
  regenerates smaller but exposes the same symbols; consumers DCE
  unused detectors as before. Bump `[deps.ai-hwaccel] tag = "2.3.0"`
  and re-run `cyrius lib sync` + `cyrius deps`.
- **Gates**: 11 test units (518 assertions), 6 fuzz harnesses,
  `cyrius vet` (37 deps, 0 untrusted), raw-offset guard, distlib
  drift + determinism — all green.

## [2.2.6] — 2026-05-19

**Library-consumer follow-ups from mihi 0.4.0 integration.** Three
gaps surfaced when `mihi` shipped its M3 GPU probe against the 2.2.5
no-exec API: four detectors weren't populating `device_name` (so
mihi reported `(unnamed)` for any ROCm / TPU / Gaudi / Neuron card),
and `cache.cyr`'s disk-write path referenced `registry_to_json` from
the bundle-excluded `json_out.cyr`, leaking a persistent linker
warning into every consumer build. This release closes all four
name gaps and brings `json_out.cyr` into the bundle so the dangling
reference resolves. Pure consumer ergonomics — no API changes.

#### Added

- **`src/json_out.cyr` is now in the bundle.** Previously excluded
  with the rationale "library consumers build their own
  serialization", but `cache.cyr::disk_cached_get_or_detect` calls
  `registry_to_json` unconditionally on the write path. Excluding
  the symbol left every library consumer with an `undefined function
  'registry_to_json'` linker warning (DCE elides the call, so the
  binary was always correct — but the noise was real). Library
  consumers that don't need JSON output still DCE the entire
  serializer; binary size on `cyrius build src/main.cyr build/...`
  is unchanged at 2.2.6.
- **README "Architecture" comment**: `json_out.cyr` reclassified
  from "CLI-only" to "available to library consumers".

#### Fixed

- **`detect_rocm` populates `profile_device_name`** (`src/detect/rocm.cyr`).
  Tries `/sys/class/drm/cardN/device/product_name` first (newer
  amdgpu kernels expose this on some discrete cards); falls back to
  synthesizing `"AMD Radeon (PCI 0x<vendor>:0x<device>)"` from the
  always-present `vendor` + `device` sysfs files. Worst case (both
  unreadable) the string is `"AMD Radeon"`. On archaemenid (Ryzen
  5800H iGPU, PCI 1002:1638) the CLI now reports
  `AMD Radeon (PCI 0x1002:0x1638)` where 2.2.5 reported nothing.
- **`detect_tpu` populates `profile_device_name`** (`src/detect/tpu.cyr`).
  Builds `"Google TPU <version>"` (v4 / v5e / v5p) via the existing
  `tpu_version_name` helper.
- **`detect_gaudi` populates `profile_device_name`** (`src/detect/gaudi.cyr`).
  Prefers the hl-smi CSV `name` field (`HL-225` / `HL-325`) when
  present; falls back to `"Intel Gaudi2"` / `"Intel Gaudi3"` via
  `gaudi_gen_name`. CSV field is copied to a null-terminated buffer
  (same `alloc + memcpy + NUL` pattern apple.cyr already uses).
- **`detect_neuron` populates `profile_device_name`** (`src/detect/neuron.cyr`).
  Both the `neuron-ls --json-output` parser and the `/dev/neuron*`
  sysfs fallback now set `"AWS Inferentia"` / `"AWS Trainium"` via
  `neuron_chip_name`.

#### Changed

- **`VERSION`**: 2.2.5 → 2.2.6.
- **`dist/ai-hwaccel.cyr`** — regenerated by `cyrius distlib`.
  Grows slightly with the bundled `json_out.cyr` (deterministic
  across re-runs).
- **`cyrius.cyml [lib].modules`** — appends `"src/json_out.cyr"`.
  Comment updated to call out the 2.2.6 inclusion + rationale.

#### Compatibility

- **CLI binary**: no change. Same output formatting, same flags;
  the per-detector name strings flow through the existing JSON /
  table / summary paths. All 11 test units (518 assertions) still
  pass.
- **2.2.4 / 2.2.5 binary consumers**: zero impact.
- **2.2.5 library consumers (mihi 0.4.0)**: source-compatible. After
  bumping `[deps.ai-hwaccel] tag = "2.2.6"` and re-running
  `cyrius deps`, mihi's `mihi_gpu_name(0)` returns the populated
  string for ROCm/TPU/Gaudi/Neuron where it returned 0 (null) on
  2.2.5. The `undefined function 'registry_to_json'` warning is
  gone.

## [2.2.5] — 2026-05-19

**No-exec detection contract — `mihi`-shaped consumers can call the
detection surface without transitively spawning processes.** 2.2.4
shipped the `[lib]` reshape, but `registry_detect()` still fans out to
eight backends that shell out to vendor CLIs (`nvidia-smi`,
`system_profiler`, `vulkaninfo`, `hl-smi`, `neuron-ls`, `xpu-smi`,
`cerebras_cli`, `gc-info`) plus a `detect_interconnects` post-pass
that calls `ibstat`/`nvidia-smi topo`. Consumers with a no-subprocess
contract — `mihi` first, whose CLAUDE.md forbids `exec_*` from inside
a probe — couldn't safely call the entry point. This release classifies
every backend, adds a `builder_no_exec()` mask, and adds a
`registry_detect_no_exec()` entry point that masks off exec backends
(defense-in-depth even if the caller's mask had them set) and skips
`detect_interconnects`. The eight sysfs/syscall-only backends — ROCm,
Intel NPU, AMD XDNA, TPU, Qualcomm, Groq, Samsung NPU, MediaTek APU —
plus the sysfs post-passes (`enrich_bandwidth/pcie/numa`,
`detect_storage`, `detect_environment`) still run.

#### Added

- **`backend_uses_exec(b)` in `src/types.cyr`** — predicate over the
  `Backend` enum. Returns 1 for CUDA / APPLE / VULKAN / GAUDI /
  NEURON / INTEL_ONEAPI / CEREBRAS / GRAPHCORE; 0 for the remaining
  eight. Source-of-truth grep: any `detect_<backend>` in
  `src/detect/<backend>.cyr` that calls `run_tool*` is EXEC. Note
  `cloud_asic.cyr` is mixed (Cerebras and Graphcore EXEC, Groq sysfs)
  and `intel.cyr` is split across two backend bits (NPU sysfs, oneAPI
  EXEC) — the predicate is correct per-backend, not per-file.
- **`builder_no_exec()` in `src/registry.cyr`** — returns a builder
  mask with only the sysfs/syscall backends enabled. Built by
  iterating `BACKEND_COUNT` and inverting `backend_uses_exec`, so
  the source of truth stays in one place.
- **`registry_detect_no_exec()` in `src/registry.cyr`** — convenience
  entry point. Calls the new `registry_detect_with_opts(builder_no_exec(), 0)`
  internal. Library consumers with a no-subprocess contract (mihi M3,
  any future read-only probe library) include this and never have to
  audit the per-backend exec status themselves.
- **`registry_detect_with_opts(mask, allow_exec)` in `src/registry.cyr`** —
  generalized orchestrator. When `allow_exec == 0` it force-strips the
  mask through `builder_no_exec()` and skips `detect_interconnects`.
  Existing `registry_detect()` / `registry_detect_with()` are now thin
  wrappers that pass `allow_exec = 1`, preserving 2.2.4 behavior exactly.

#### Changed

- **`VERSION`**: 2.2.4 → 2.2.5.
- **`dist/ai-hwaccel.cyr`** — regenerated by `cyrius distlib` to pick
  up the new entry points. Deterministic across runs.
- **README "Using as a library"** — documents `registry_detect_no_exec()`
  as the preferred entry point for read-only consumers; bumps the
  example `tag` to `"2.2.5"`.

#### Compatibility

- **CLI binary**: no change. `registry_detect()` semantics preserved —
  same backends, same post-passes, same warnings. All 11 test units
  (518 assertions) still pass; smoke output byte-identical to 2.2.4.
- **Existing binary consumers**: zero impact (they invoke the CLI,
  which calls `registry_detect()`, unchanged).
- **2.2.4 library consumers**: source-compatible. `registry_detect()`
  and `registry_detect_with(mask)` still resolve and behave identically.
- **New no-exec consumers** (mihi M3): `include "lib/ai-hwaccel.cyr"`,
  call `registry_detect_no_exec()`, walk `reg_profiles(r)`. The no-exec
  contract is now part of the library API rather than tribal knowledge.

## [2.2.4] — 2026-05-19

**`[lib]` reshape — first library-shaped consumer (`mihi`) unblocked.**
ai-hwaccel has been binary-only since v1.0.0; every consumer to date
(hoosh, daimon, Irfan, AgnosAI, murti, tazama) calls the CLI and parses
JSON. `mihi` v0.4.0 (M3 — GPU probe) cannot: its CLAUDE.md forbids
spawning processes from probes, so the GPU detection surface it needs
has to be reachable via `include`, not `exec`. This release adds the
`[lib].modules` surface and the `cyrius distlib`-produced
`dist/ai-hwaccel.cyr` bundle so mihi (and any future library consumer)
can pin against ai-hwaccel from their own `cyrius.cyml`. Mirrors the
agnosys / libro / patra / yukti / mihi pattern.

#### Added

- **`[lib].modules` in `cyrius.cyml`** — 35 modules listed in
  `src/main.cyr` include order. Excluded: `src/main.cyr` (CLI argv
  parsing) and `src/json_out.cyr` (CLI output formatting — library
  consumers serialize on their own terms). Every detection backend,
  the registry/profile surface, the sharding planner, the cost model,
  the training memory estimator, the model-format header parser, and
  the async/cache/lazy registry wrappers all ride along.
- **`dist/ai-hwaccel.cyr`** — single-file bundle produced by
  `cyrius distlib`. 5392 lines, 168 KiB at 2.2.4. Byte-deterministic
  across runs (verified — same SHA-256 from two sequential invocations).
  Consumers pull via `[deps.ai-hwaccel] modules = ["dist/ai-hwaccel.cyr"]`;
  `cyrius deps` drops it as `lib/ai-hwaccel.cyr` for `include`.
- **README "Using as a library" subsection** — copy-pasteable
  `[deps.ai-hwaccel]` block + `include` example. Documents the
  bundle-included surface and the CLI-only exclusions.
- **CI: `distlib drift + determinism` step** — runs `cyrius distlib`,
  diffs against the committed bundle (drift = consumers shipping stale
  code), then re-runs and SHA-256-compares (non-determinism = breaks
  the reproducible-build contract). Mirrors the libro / mihi / yukti
  gates. Sits between `Lint` and `Build (DCE)` in the workflow.

#### Changed

- **`VERSION`**: 2.2.3 → 2.2.4.
- **`docs/development/roadmap.md`** — 2.2.4 deliverable checkboxes
  marked complete for the in-repo work (manifest, bundle, README,
  no-regression, determinism guard); the `mihi-side smoke` remains
  unchecked pending external mihi M3 work.

#### Compatibility

- **CLI binary**: no change. `cyrius build src/main.cyr build/ai-hwaccel`
  still produces a 287 KiB ELF; all 11 test units (518 assertions) pass;
  output byte-identical to 2.2.3.
- **Existing binary consumers**: zero impact. They pin to the binary
  release artifacts and invoke `ai-hwaccel` as a subprocess, which is
  untouched.
- **New library consumers**: `[deps.ai-hwaccel] tag = "2.2.4"` resolves
  via `cyrius deps`, then `include "lib/ai-hwaccel.cyr"` exposes the
  detection entry points (`registry_detect`, `registry_detect_async`,
  the family-specific `detect_<backend>` calls, the profile / plan /
  cost / training APIs) directly — no subprocess, no JSON parsing,
  no environment leakage.

## [2.2.3] — 2026-05-19

**Toolchain rename slot — `cycc` canonical, `cc5` only as a legacy
symlink. Push back DXGI feature work; absorb the cyrius 6.0.0 compiler
rename as the 2.2.3 slot.** Cyrius 6.0.0 renames the compiler binary
from `cc5` to `cycc` (with `cc5_aarch64` → `cycc_aarch64`, `cc5_win`
→ `cycc_win`). The legacy `cc5*` names ship as symlinks for now, so
nothing breaks immediately — but every live, forward-looking reference
moves to the canonical name so the codebase stops drifting behind the
toolchain.

#### Changed

- **`cyrius.cyml`**: `cyrius = "5.11.8"` → `cyrius = "6.0.0"`.
- **`VERSION`**: 2.2.2 → 2.2.3.
- **`CLAUDE.md`**:
  - Compiler line: "Cyrius cc5 5.11.8" → "Cyrius cycc 6.0.0", with
    a note that `cc5` remains as a legacy symlink in `~/.cyrius/bin/`.
  - `#derive(accessors)` version note: `cc5 v3.7.1+` → `cycc v3.7.1+`.
- **`README.md`** Key Numbers row: `Cyrius cc5 5.10.34` → `Cyrius cycc 6.0.0`.
- **`src/detect/windows.cyr`** header comments: forward-looking
  references to `cc5_win` → `cycc_win` (with the legacy-symlink note
  inline). The 5.11.5 PE emit blocker note is preserved as a status
  question against 6.0.0 rather than as a hard block.
- **`docs/development/roadmap.md`**:
  - 2.2.3 CI cross-build entry — reframed from "upstream-blocked on
    cc5_win 5.11.5" to "re-verify against `cycc_win` 6.0.0; smoke
    probe required before declaring it unblocked."
  - 2.3.0 Python-bindings entry: "cyrius cc5 emits ELF/Mach-O/PE" →
    "cyrius cycc emits ELF/Mach-O/PE."
  - 2.4.0 hot-plug entry: "cc5's defer" → "cycc's defer."
- **`.github/workflows/release.yml`**: aarch64 gate uses
  `~/.cyrius/bin/cycc_aarch64` (warning text updated to match).
- **`memory/reference_windows_host.md`** forward guidance — `cc5_win`
  / `cc5_win_cross` → `cycc_win` / `cycc_win_cross`.
- **`memory/MEMORY.md`** index entry phrasing updated.
- **CI stdlib resolution (`.github/workflows/ci.yml`,
  `.github/workflows/release.yml`)**: insert a "Sync stdlib into
  `./lib/`" step that runs `cyrius lib sync` before the existing
  "Resolve non-stdlib deps" (`cyrius deps`) step. Pre-6.0.0 the
  single `cyrius deps` invocation handled both; in 6.0.0 the
  responsibilities split, so CI now mirrors the local workflow.
  Without this change, CI fails with 18 × "cannot read
  ./lib/<name>.cyr" because `cyrius deps` no longer fills the
  stdlib subset.

#### Not changed (intentional)

- Shipped CHANGELOG entries (`cc5 adoption arc`, all 2.1.x sections,
  the 2.2.1 toolchain-bump entry, the 2.2.2 cc5_win 5.11.5 blocker
  filing) — historical record using the names in use at the time.
- `memory/feedback_cc5_win_exit_propagation.md` filename — preserved
  to keep cross-refs from the cyrius issue tree intact.
- Source files outside `src/detect/windows.cyr` — no other source
  references the compiler name; nothing to update.

#### Pushed to 2.2.4

- **DXGI COM binding + `DXGI_ADAPTER_DESC1` parser** (was 2.2.3 plan).
- **Linux-hosted fixture tests under `tests/fixtures/windows/`**.

#### Verification

- Linux build (`CYRIUS_DCE=1 cyrius build src/main.cyr build/ai-hwaccel`)
  against cycc 6.0.0: **287,096 bytes** (was 286,152 at 2.2.2, +944 bytes).
  The size change comes from the 6.0.0 stdlib snapshot, not project source —
  the stdlib added new modules (`atomic`, `async`, `mmap`, per-arch
  `syscalls_*_linux`, …) that the include graph now pulls in. 638
  unreachable fns NOPed by DCE (was 629 at 2.2.2).
- Test suite: 11 units, **520 assertions, 0 failures** (was 518 at 2.2.2;
  +2 from the new stdlib).
- fmt sweep: clean across `src/`, `tests/tcyr/`, `fuzz/`, `benches/`.

#### Stdlib repopulation note (workflow change)

- **`cyrius deps` no longer repopulates the stdlib in 6.0.0.** Use
  `cyrius lib sync` instead — it copies the snapshot at
  `~/.cyrius/versions/<pin>/lib/*.cyr` into `./lib/`. After fresh
  clone or cyrius upgrade, run `cyrius lib sync` then `cyrius deps`
  (the latter still resolves non-stdlib `[deps.*]` entries).
  CLAUDE.md updated to reflect this.

## [2.2.2] — 2026-05-11

**Windows backend — source-side skeleton.** `src/detect/windows.cyr`
exists, wired into the include graph behind `#ifdef CYRIUS_TARGET_WIN`,
ready to receive the DXGI binding. **Real Win64 cross-build is not
viable yet** — cc5_win 5.11.5 has a PE emit regression (documented
below) that prevents end-to-end validation. Source structure ships;
cross-build + cross-host smoke gate on a cc5_win patch.

### Added

- **`src/detect/windows.cyr`** — single-file skeleton behind
  `#ifdef CYRIUS_TARGET_WIN`. Stubs `detect_windows(profiles,
  warnings)` matching the `detect_<backend>(profiles, warnings)`
  convention used by the other detectors. Records a
  `warning_tool_not_found("dxgi-binding-pending")` warning so the
  runtime makes the stub state obvious at JSON level. Header
  comments document the planned scope (COM binding, DXGI_ADAPTER_DESC1
  parsing, ACCEL_DXGI / BACKEND_WIN_DXGI additions) for 2.2.3.
- **`src/main.cyr` include**: `include "src/detect/windows.cyr"`
  added between environment.cyr and registry.cyr. On Linux / macOS
  builds the entire file preprocesses out — `build/ai-hwaccel`
  stays **byte-identical at 286,152 bytes**.

### Investigated and recorded as upstream blocker

- **cc5_win 5.11.5 PE emit broken on cass**. Probed end-to-end this
  slot:
  - `syscall(60, 42); ` source → cc5_win builds a 1536-byte PE.
  - PE loads on cass (Win11 26200), process starts.
  - Exit code under `cmd /v /c "exe & echo exit=!errorlevel!"` is
    **0x40001000** (1073745920), not 42.
  - WriteFile output ("hello\n" from `syscall(1, 1, "hello\n", 6);`)
    never reaches stdout.
  - Same byte-identical emit from `cc5_win` (install) and
    `cc5_win_cross` (cyrius source build, v5.10.37) — both versions
    produce the same broken PE.
  - Linux build of the same source works fine.

  Recorded as `memory/feedback_cc5_win_exit_propagation.md` and
  filed in the cyrius tree at
  `docs/development/issues/2026-05-11-ai-hwaccel-cc5-win-pe-exit-propagation.md`
  (pending user-side review + commit on cyrius). ai-hwaccel's Win64
  cross-build + cass smoke gate on the fix. Linux fixture-based tests
  for the DXGI parser (2.2.3) don't depend on this — the parser
  logic can be exercised against synthetic `dxdiag` output without
  needing a Win64 binary to load on cass.

- **Full ai-hwaccel cross-build via cc5_win not yet viable.** A
  full `cc5_win src/main.cyr` invocation runs to exit 0 but emits
  a 1,536-byte stub PE rather than a real binary — cc5_win silently
  fails on something deeper in the transitive include tree (likely
  one of the Linux-only stdlib paths that the PE backend doesn't
  yet reroute). Cyrius's own 5.11.6 release notes call out this
  shape as "compile-path quirk." Skipping the CI cross-build step
  until cc5_win can handle the full tree.

### Verification

- Linux build: **286,152 bytes** (byte-identical to 2.2.1 — the
  windows.cyr include adds zero on non-Win targets).
- 11 test units, **518 assertions, 0 failures**.
- fmt sweep clean.
- `cyrius vet src/main.cyr`: 37 deps (was 36 — `+1` for
  `src/detect/windows.cyr`), 0 untrusted, 0 missing.

### Next slot — 2.2.3

Either:
- **Wait for cc5_win patch** (cyrius-side) and resume cross-build +
  COM binding in one go.
- **Or proceed Linux-side**: implement the DXGI parser against
  synthesized `dxdiag` text fixtures under `tests/fixtures/windows/`.
  Parser logic ships and ride along once the cross-build is viable.

The user can redirect.

## [2.2.1] — 2026-05-11

**Cyrius toolchain bump 5.10.34 → 5.11.8.** Mechanical prerequisite
slot — same shape as 2.0.1 (which bumped 3.10.0 → 5.10.34). Unblocks
the Windows DXGI work queued for 2.2.x by bringing `cc5_win` into the
standard install bundle and folding the v5.11.6 PE exit-code fix into
the pinned toolchain.

### Changed

- **`cyrius.cyml`**: `cyrius = "5.10.34"` → `cyrius = "5.11.8"`.
- **`CLAUDE.md`**: compiler reference line bumped to match.

### Toolchain context

- **5.10.x → 5.11.x** absorbed in one bump (the cycle ran .35 → .50 →
  .0 → .8, 9 patches + the 5.11.0 minor open). Highlights relevant to
  ai-hwaccel:
  - **5.10.49** — premise-debunk: PE exit-code propagation was always
    working; the false-negative was a chat-side wrapper bug
    (`%errorlevel%` vs `!errorlevel!`).
  - **5.11.6** — underlying PE exit-code path fully verified, plain
    `cmd /c "prog.exe & echo exit=%errorlevel%"` shape works again on
    Win64 PE binaries.
  - **5.11.8** — `cc5_win` shipped in the default install bundle at
    `~/.cyrius/bin/cc5_win` (612 KB). No longer needs to be invoked
    from the cyrius source-build `cc5_win_cross` artifact.
- **5.11.0 minor opens** with kavach P1 sandbox syscall wrappers
  (`sys_fchmod`, `sys_setresuid` / `sys_setresgid`, `sys_prctl`,
  `sys_seccomp`, `sys_execveat`) and the stdlib annotation arc. None
  of these affect ai-hwaccel directly — we don't use seccomp or
  unprivileged process management.

### Verification

- Clean rebuild against 5.11.8: `build/ai-hwaccel` byte-identical to
  2.2.0 / 2.1.7 (**286,152 bytes**). The toolchain bump produces the
  same emit for the existing source.
- Test suite: 11 units, **518 assertions, 0 failures**.
- fmt sweep: clean across `src/`, `tests/tcyr/`, `fuzz/`, `benches/`.
- `cc5_win` present at `~/.cyrius/bin/cc5_win` — Windows cross-build
  is now a `cc5_win <source> <output>` invocation away (skeleton
  lands next slot).

### Documentation

- **Memory** (`reference_windows_host.md`) — wrapper-gotcha section
  reorganised: 5.11.8 pin uses plain shell; the `cmd /v /c …
  !errorlevel!"` variant kept as a historical note for any
  5.10.x-pinned consumer that hits the same issue.
- **Roadmap** — Windows DXGI entry no longer carries the "install
  cc5_win" prerequisite; pickup target is now the source-side work
  only.

## [2.2.0] — 2026-05-11

**Test-file rename correction + README refresh.** A docs-and-renames
slot — accepted as a release but not a real Platform Validation
deliverable. The 2.2.x arc's actual feature work (Windows DXGI,
hardware-fixture coverage for the untested backends) is open and
queued for 2.2.1+ pickup, not deferred or blocked.

### Changed

- **6 test files renamed** to match actual content (audit revealed
  the 2.1.0 rename assumed phase numbers → subject 1:1, which several
  didn't):

  | 2.1.0 name (misleading) | Actual content | 2.2.0 name |
  |---|---|---|
  | `cost_model_test.tcyr` | sharding plans + training memory + model | `planning_test.tcyr` |
  | `detect_gaudi_test.tcyr` | CUDA + Gaudi + Neuron parsing | `gpu_parser_test.tcyr` |
  | `detect_neuron_test.tcyr` | Apple + Intel + AMD XDNA + cloud ASIC | `backend_test.tcyr` |
  | `registry_test.tcyr` | which + run_tool + CSV + sysfs | `io_test.tcyr` |
  | `sharding_test.tcyr` | interconnect + bandwidth + PCIe + storage | `topology_test.tcyr` |
  | `system_io_test.tcyr` | registry + builder + suggest_quant | `registry_test.tcyr` |

  The remaining 5 names (`foundation_test`, `profile_test`,
  `requirement_test`, `json_output_test`, `model_format_test`) were
  already accurate and stay unchanged. Git tracks all 6 as renames —
  content byte-identical, all 518 assertions still pass.

- **`README.md` refreshed**:
  - Binary size: 217 KB → 286 KB (cc5 stdlib growth + derived setter
    surface; documented in 2.0.1 / 2.1.x CHANGELOG entries).
  - Compiler line: Cyrius cc5 5.10.34 (was implicit).
  - Test-units table (the 11 `.tcyr` files with one-line descriptions)
    replaces the stale "11 phases" reference.
  - New section: "Pattern: derived struct accessors" with the
    `#derive(accessors)` shape + CI gate pointer.
  - Removed: Compile time + Source LOC rows (out of date and not
    load-bearing; binary size + assertion count carry the same
    signal).
  - Development workflow snippet: now includes `cyrius deps` and the
    `tests/tcyr/*.tcyr` test-loop pattern that matches CI.

- **`docs/development/roadmap.md`**: the 2.1.0 test-rename entry now
  carries the corrected mapping (was misleading documentation
  alongside the bad names).

### Test suite

- 11 units, 518 assertions, 0 failures (unchanged from 2.1.7 — pure
  rename slot).

### Carried forward — **open, queued for 2.2.1+**

These are the real 2.2.x arc items. Reframed from "deferred / access-
blocked" to **open work**, because the source-side path can ship
against synthesized fixtures even before hardware access lands.

- **Windows DXGI backend** — `src/detect/windows.cyr` behind
  `#ifdef CYRIUS_TARGET_WIN64`, COM binding for `IDXGIFactory1::
  EnumAdapters1`, `DXGI_ADAPTER_DESC1` parsing. Toolchain side:
  install `cc5_win` into `~/.cyrius/bin/` (the cross-compiler exists
  at v5.10.37 in cyrius's source build), wire CI cross-build behind
  the same `if [ -x $HOME/.cyrius/bin/cc5_win ]` gate the release
  workflow uses for aarch64. Next pickup target.
- **Hardware-fixture coverage** — move inline sample tool outputs
  in `gpu_parser_test.tcyr` / `backend_test.tcyr` to per-backend
  files under `tests/fixtures/`. Establishes the drop-zone for
  contributor captures from real H100 / MI300X / TPU v5 / Trn1 /
  Gaudi 3 hardware. The parser tests run against fixtures, so the
  source-side work doesn't gate on cloud access.
- **Untested-backend parsers** — Cerebras / Graphcore / Groq /
  Samsung NPU / MediaTek APU. Each one has a documented sysfs or
  CLI output format; the parser can ship against a synthesized
  fixture from vendor docs and graduate to a real-hardware fixture
  as captures come in.

See `docs/development/roadmap.md` § 2.2.x for the per-item plan.

## [2.1.7] — 2026-05-11

**P(-1) scaffold hardening: 8 more structs derived, every heap struct
in the codebase now uses `#derive(accessors)`.** Closes the 2.1.x
minor before 2.2.0's platform-validation arc opens. Baseline +
post-audit benchmarks captured; small perf cost documented.

### Pre-audit baseline

- 518 assertions, 0 failures across 11 test units
- `cyrius lint`: clean (no warnings on any `src/` / `src/detect/` file)
- `cyrius fmt`: clean across `src/`, `tests/tcyr/`, `fuzz/`, `benches/`
- Binary: 281,720 bytes
- Bench hot-paths:
  - `total_memory_13dev`: 136 ns
  - `has_accelerator_13dev`: 27 ns
  - `count_family_gpu_13dev`: 288 ns
  - `json_serialize_13dev`: 24 µs
  - `json_summary_13dev`: 5 µs
  - `parse_cuda_8gpu`: 22 µs

### Changed

Eight more structs converted to `#derive(accessors)`:

- **`env` struct** (`src/system_io.cyr`, was RuntimeEnvironment) — 8
  fields: `is_docker`, `is_k8s`, `k8s_namespace`, `cloud_provider`,
  `instance_type`, `region`, `k8s_gpu_count`, `k8s_gpu_source`. Two
  setters renamed (field-derived): `env_set_docker` →
  `env_set_is_docker`, `env_set_k8s` → `env_set_is_k8s`. 5 caller
  sites updated (4 in `src/detect/environment.cyr`, 1 in
  `tests/tcyr/profile_test.tcyr`).
- **`sio` struct** (`src/system_io.cyr`, was SystemIo) — 3 fields:
  `interconnects`, `storage`, `environment`. Also cleaned up 4
  internal `var ics = load64(sio);` shortcuts → `sio_interconnects(sio)`.
- **`shard` struct** (`src/system_io.cyr`, was ModelShard) — 6
  fields: `id`, `layer_start`, `layer_end`, `device_type`,
  `device_id`, `memory_bytes`. Constructor stays `model_shard_new`.
  Internal helpers `shard_num_layers` / `shard_is_valid` switched
  from `load64(ms + N)` to derived getters.
- **`cloud_inst` struct** (`src/cost.cyr`, was CloudGpuInstance) — 10
  fields. Field name `total_gpu_mem_gb` (layout) → `total_mem_gb` to
  match existing `cloud_inst_total_mem_gb` accessor callers in
  `src/main.cyr:210` and `src/cost.cyr:257`. No external constructor
  (built by `load_cloud_instances` JSON parser via
  `_parse_cloud_field_*` helpers, which take offset as a parameter
  and stay raw inside the defining file).
- **`rec` struct** (`src/cost.cyr`, was InstanceRecommendation) — 3
  fields: `instance`, `mem_required`, `headroom_x100`. Constructor
  stays `inst_rec_new`.
- **`cached` struct** (`src/cache.cyr`, was CachedRegistry) — 4
  fields: `registry`, `last_detect_secs`, `ttl_secs`, `mutex`. Five
  internal helpers (`cached_get`, `cached_invalidate`, `cached_ttl`,
  `cached_is_valid`) switched to derived accessors.
- **`disk_cached` struct** (`src/cache.cyr`, was DiskCachedRegistry)
  — 5 fields adding `cache_path`. Same internal-cleanup pattern.
- **`lazy` struct** (`src/lazy.cyr`, was LazyRegistry) — 3 fields:
  `profiles`, `probed` (i64 bitmask), `mutex`. Seven internal raw
  `load64(lr + N)` sites cleaned to derived getters. Bit-manipulation
  helpers (`_lazy_is_probed`, `_lazy_mark_probed`) use
  `lazy_probed` + `lazy_set_probed` plus inline mask compute.

### CI

- **Raw-offset guard at 15 entries** (10 cross-file + 5 field-count
  bound):
  - Cross-file `check_struct`: `storage` (sd), `ic` (ic), `plan` (sp),
    **`shard` (ms)**, **`sio` (sio)**, `reg` (r), `profile` (p),
    **`cloud_inst` (ci)**, **`rec` (rec)**, **`lazy` (lr)**. Five new.
  - Field-count bound: `est` (e, 4), **`env` (e, 8)** (was `runtime_env`,
    renamed to match the struct name), `meta` (m, 5), `model` (m, 4),
    **`disk_cached` (c, 5)** (covers both `cached` and `disk_cached`
    since both use param `c` in the same file; the larger bound
    accommodates both — see roadmap rationale). Two new.

### Documentation

- `CLAUDE.md` now lists `#derive(accessors)` as a project principle
  with the canonical pattern, the full 16-struct inventory, and a
  pointer to the CI gate.
- `docs/development/roadmap.md` marks the 2.1.x arc as **SHIPPED**
  with a per-slot summary; 2.2.0 is the next arc.

### Post-audit verification

- 518 assertions, 0 failures (unchanged across the slot)
- Binary: **286,152 bytes** (was 281,720) — **+4,432 bytes** (≈+1.6%)
  for derive-generated setters across 8 new structs. cc5's DCE
  inlines the truly-internal setters (constructor-only callers); the
  net cost is from setters that have external consumers and stay as
  callable functions.
- Bench deltas vs baseline:
  - `total_memory_13dev`: 136 → 145 ns (+9 ns, ≈+6.6%)
  - `count_family_gpu_13dev`: 288 → 304 ns (+16 ns, ≈+5.6%)
  - `has_accelerator_13dev`: 27 → 28 ns (+1 ns, within noise)
  - `json_serialize_13dev`: 24 → 25 µs (+1 µs, ≈+4%)
  - `json_summary_13dev`: 5 → 6 µs (+1 µs, ≈+20% but tiny absolute)
  - `parse_cuda_8gpu`: 22 → 20 µs (improved — within noise)

  The sub-µs hot paths regress slightly because derived getters are
  thin function calls (`fn reg_profiles(r) { return load64(r); }`)
  rather than inlined `load64(r + 0)`. The accessor surface offers
  much stronger guarantees (CI-gated, derive-traced, no offset
  arithmetic at call sites) — the trade is real but the absolute
  numbers stay in the sub-microsecond range for everything but
  `json_*`, which were already micro-second.

### Arc closes

- **2.1.x cc5 adoption arc complete.** Seven slots over two days.
  Every heap-allocated struct in the codebase (16 total) now uses
  `#derive(accessors)`. CI gate covers them all. The codebase is
  internally consistent before 2.2.0's external-feature work begins.
- **Next arc: 2.2.0 — Platform Validation** (Windows PE via cc5
  Win64 backend, live cloud hardware validation, untested-backend
  triage).

## [2.1.6] — 2026-05-11

**cc5 adoption arc closes — `profile` converted, 8 structs total on
`#derive(accessors)`.** The biggest struct in the codebase (20 fields,
160 bytes, most-called accessor surface) now uses derived getters and
setters. The cc5 adoption arc that opened with 2.1.0's mechanical
test/CI reorg ends here.

### Changed

- **`profile` struct** (`src/profile.cyr`, was AcceleratorProfile) —
  20 fields: `accel_type`, `device_id`, `available`, `memory_bytes`,
  `compute_cap`, `driver_version`, `device_name`, `mem_bw_x1000`,
  `mem_used`, `mem_free`, `pcie_bw_x1000`, `numa_node`, `temp_c`,
  `power_x1000`, `gpu_util`, `tpu_version`, `tpu_chips`, `gaudi_gen`,
  `neuron_chip`, `neuron_cores`. The 20 hand-rolled getters and 17
  hand-rolled setters are gone; the constructor `profile_new` calls
  the derived `profile_set_*` setters to initialise all 20 slots.
  Struct name `profile` matches the existing `profile_*` accessor
  convention — zero callsite changes for the dozens of detect/* /
  registry / json_out / sharding / training / cost helpers that
  consume the surface.
- **4 cross-file raw `store64(p + 24, …)` writes** (all setting
  `profile.memory_bytes` after hardware-specific detection adjusts the
  value) converted to `profile_set_memory_bytes(p, …)`:
  - `src/detect/gaudi.cyr:46` — hl-smi memory override
  - `src/detect/vulkan.cyr:62` — vulkaninfo heapSize parsing
  - `src/detect/cuda.cyr:136` — GH200 unified memory adjustment
    (`+ 480 GiB`)
  - `src/detect/rocm.cyr:100` — ROCm CXL visible-VRAM total

### CI

- **Raw-offset guard** now registers 9 entries (5 cross-file + 4
  field-count bound):
  - Cross-file `check_struct`: `storage` (sd), `ic` (ic), `plan` (sp),
    `reg` (r), **`profile` (p)** (new).
  - Field-count bound: `est` (e, 4), `runtime_env` (e, 8),
    `meta` (m, 5), `model` (m, 4).

### Binary size

- `build/ai-hwaccel`: **281,720 bytes** (was 281,592 at 2.1.5). +128
  bytes — `profile` had hand-rolled setters already, so the derive
  output mostly replaced existing code rather than adding fresh setter
  bodies. 518 assertions, 0 failures.

### Arc closes

- **2.1.x cc5 adoption arc complete.** Six slots over two days:
  - 2.1.0 — test reorg + CI tighten
  - 2.1.1 — Rust parity audit + gitignore cleanup
  - 2.1.2 — defer audit (clean), chrono investigated and rejected,
    build/ untracked
  - 2.1.3 — `#derive(accessors)` proof of concept (meta + storage),
    first raw-offset CI gate
  - 2.1.4 — three more structs (ic + plan + est), CI gate gains
    libro field-count bound check
  - 2.1.5 — two more structs (reg + model), gate at 8 entries
  - 2.1.6 — `profile` (this slot), gate at 9 entries
- **Next arc: 2.2.0 — Platform Validation.** Live cloud hardware
  validation (NVIDIA H100/A100/GH200, AMD MI300X, Google TPU v5,
  AWS Neuron, Intel Gaudi 3), Windows PE backend (now reachable
  via cc5 5.10.x), and the untested-backend list (Cerebras WSE,
  Graphcore IPU, Groq, Samsung NPU, MediaTek APU).

## [2.1.5] — 2026-05-11

**cc5 adoption arc — `#derive(accessors)` continues. Two more structs
converted (`reg` + `model`), CI gate expanded.** Bundled because both
were medium-complexity and shared the same review surface (registry
internals, model dispatch). Zero external call-site changes.

### Changed

- **`reg` struct** (`src/registry.cyr`, was accelerator_registry) —
  4 fields: `profiles`, `warnings`, `system_io`, `schema`. Struct named
  `reg` to match the existing `reg_*` accessor convention; constructor
  stays `registry_new`. Also cleaned up 9 internal `var profs =
  load64(r);` shortcuts in helpers (`reg_total_memory`,
  `reg_total_accel_memory`, `reg_has_accelerator`, `reg_best_available`,
  `reg_count_by_family`, `reg_by_family`, `reg_suggest_quant`) →
  `var profs = reg_profiles(r);` so the file is internally consistent
  with the new accessor surface.
- **`model` struct** (`src/model.cyr`, was ModelProfile) — 4 fields:
  `name`, `family`, `params_b_x1000`, `dtype`. Param `m` is shared with
  `meta` in `model_format.cyr` — both rely on the libro field-count
  bound CI check rather than a cross-file specific-struct guard. One
  in-place mutation in `_parse_models` line 137 — `store64(m + 16,
  whole * 1000 + frac)` — converted to the derived setter
  `model_set_params_b_x1000(m, …)`.

### CI

- **Raw-offset guard registry expanded** to 8 entries total:
  - Cross-file `check_struct`: `storage` (sd), `ic` (ic), `plan` (sp),
    **`reg` (r)** (new).
  - Field-count bound: `est` (e, 4), `runtime_env` (e, 8), `meta` (m, 5),
    **`model` (m, 4)** (new).

### Binary size

- `build/ai-hwaccel`: **281,592 bytes** (was 280,696 at 2.1.4). +896
  bytes for derive-generated setters across `reg` + `model`. 518
  assertions, 0 failures.

## [2.1.4] — 2026-05-11

**cc5 adoption arc — `#derive(accessors)` continues. Three more structs
converted + CI raw-offset gate expanded with libro's field-count bound
check.** Zero external call-site changes — every struct gets a name that
matches its existing accessor prefix, so derive generates the names the
code already imports.

### Changed

- **`ic` struct** (`src/system_io.cyr`, was Interconnect) — 4 fields:
  `kind`, `name`, `bw_x1000`, `state`. Struct named `ic` to match the
  existing `ic_*` accessor shorthand; constructor stays
  `interconnect_new`. 16 external call sites of `ic_kind` /
  `ic_name` / `ic_bw_x1000` / `ic_state` / `ic_set_state` unchanged.
- **`plan` struct** (`src/system_io.cyr`, was ShardingPlan) — 5 fields:
  `shards`, `strategy`, `strategy_count`, `total_memory`,
  `est_tps_x1000`. Struct named `plan` to match the existing `plan_*`
  accessor convention; constructor stays `sharding_plan_new`. The
  pre-existing manual setters (`plan_set_total_memory`,
  `plan_set_est_tps_x1000`) replaced by derive-generated setters; the
  rest of the surface unchanged.
- **`est` struct** (`src/training.cyr`, was MemoryEstimate) — 4 fields:
  `model_x1000`, `optimizer_x1000`, `activation_x1000`, `total_x1000`.
  Struct named `est` to match the `est_*` accessor convention;
  constructor stays `mem_est_new`. Param `e` is shared with
  `runtime_env` in `system_io.cyr` — see CI gate change below.

### CI

- **Raw-offset guard expanded with `check_offset_bound`** — libro's
  field-count bound check for structs whose canonical param name is
  ambiguous across files. For each `(file, param, struct, field_count)`
  tuple, every raw `load64(<param> + N)` / `store64(<param> + N, ...)`
  site in `<file>` must have `N ≤ (field_count − 1) * 8`. Catches
  off-by-one after a field-count shrink, and accidental access past a
  struct boundary after another struct grows. Currently registered:
  - `est`   (`src/training.cyr`,   param `e`, 4 fields → max +24)
  - `runtime_env` (`src/system_io.cyr`, param `e`, 8 fields → max +56)
  - `meta`  (`src/model_format.cyr`, param `m`, 5 fields → max +32)
- **Cross-file `check_struct` guards added** for the two newly-derived
  structs with unambiguous params:
  - `ic`   in `src/system_io.cyr`, param `ic`
  - `plan` in `src/system_io.cyr`, param `sp`

### Binary size

- `build/ai-hwaccel`: **280,696 bytes** (was 279,656 at 2.1.3). +1,040
  bytes for derive-generated `_set_*` setters across the three new
  structs. 518 assertions, 0 failures.

## [2.1.3] — 2026-05-11

**cc5 adoption arc — `#derive(accessors)` lands, first two structs
converted + raw-offset CI gate.** Proof-of-concept slot establishing
the pattern (and the CI guard rail) before the bigger structs
(`profile`, `accelerator_registry`, `model`) follow in 2.1.4+.

### Changed

- **`meta` struct** (`src/model_format.cyr`) — 5 hand-rolled
  `load64(m + N)` getter functions replaced with
  `#derive(accessors) struct meta { format; param_count; dtype;
  tensor_count; format_version; }`. The constructor `meta_new` stays
  manual (derive only generates accessors) but now calls the derived
  `meta_set_*` setters internally instead of raw `store64(m + N, …)`.
  All 21 external call sites (in `model_format.cyr` itself,
  `tests/tcyr/model_format_test.tcyr`) use the existing `meta_*`
  getter names unchanged — derive generates them under exactly the
  names the code already imports.
- **`storage` struct** (`src/system_io.cyr`) — 3 hand-rolled getters
  replaced with `#derive(accessors) struct storage { name; kind;
  bw_x1000; }`. Same shape as `meta`: constructor stays manual but
  uses derived setters; external call sites
  (`src/detect/disk.cyr:47`, `tests/tcyr/sharding_test.tcyr:147`,
  `tests/tcyr/profile_test.tcyr:195-204`) keep their `storage_*`
  getter names.

### Added

- **Raw-offset CI gate (`Raw-offset guard` step in
  `.github/workflows/ci.yml`)** — for each `#derive(accessors)`
  struct, no file outside its defining file may do raw
  `load64(<param> + N)` / `store64(<param> + N, …)` /
  `load64(<param>)` on it. Mirrors the libro v2.6.x pattern.
  Registers `storage` (param `sd`) — unambiguous across `src/`,
  works with the simple cross-file `check_struct` form. `meta`
  (param `m`) is held back because `src/model.cyr` legitimately
  uses `m` for its own (not-yet-derived) struct; that case is
  documented to use the libro field-count bound check once `model`
  itself moves to derive.

### Binary size

- `build/ai-hwaccel`: **279,656 bytes** (was 278,808 at 2.1.2).
  +848 bytes for derive-generated `_set_*` setters across both
  structs; setters for `meta` aren't called from outside the
  constructor, so stricter DCE passes will reclaim those later.
  Test suite: 518 assertions, 0 failures (unchanged).

### Investigated and rejected this slot

- **Multi-return `(value, error)` in detect/* (the 2.1.0-arc item) —
  doesn't fit.** The detect entry points are
  `detect_<backend>(profiles, warnings)`: both vec OUT-params,
  pushing 0..N profiles and 0..M structured warnings, returning an
  unused `0`. There is no single value to multi-return, and errors
  are already accumulated into `warnings` as structured entries
  rather than collapsed to a sentinel int. Closed in the roadmap;
  the out-param-vec pattern is now noted as canonical.
- **`lib/regex.cyr` for parser output (the 2.1.0-arc item) — no
  fit.** The detect parsers go `run_tool` → `str_split` (lines) →
  `parse_csv_line` (fields) → `str_contains_cstr` (single-token
  substring checks). Substring checks aren't what regex replaces;
  the CSV helpers are already idiomatic. Closed in the roadmap.
- **`lib/test.cyr` adoption (the 2.1.0-arc item) — closed as a
  misread.** `lib/test.cyr` is a `test_each` parameterised-test
  helper, not an alternative assertion framework. The tests already
  use stdlib `lib/assert.cyr`. Nothing to migrate.

## [2.1.2] — 2026-05-11

**cc5 adoption arc — verification slot.** Two roadmap items investigated;
both close cleanly with **no source change**. Documenting the outcomes so
future work doesn't re-tread the same ground.

### Verified

- **Defer-on-all-paths audit (cc4+ defer semantics)** — swept
  `src/system_io.cyr`, `src/cache.cyr`, `src/detect/command.cyr` for
  manual file-descriptor management on paths where an early return could
  leak. The only `file_open` / `file_close` pair in the entire tree is
  `cmd_getenv()` at `src/detect/command.cyr:17-48`; the `file_close` runs
  unconditionally on line 22 before any of the subsequent returns. All
  other I/O goes through `lib/fs.cyr`'s `file_read_all` / `file_write_all`
  atomic wrappers, which handle open/read/close internally. **Zero leaks
  found, no `defer` insertions needed.** Roadmap item closes; the cc4+
  defer-on-all-paths feature stays available for future code paths.

### Investigated and rejected

- **`lib/chrono.cyr` adoption for cache TTL** — attempted replacing
  `cache.cyr`'s `_monotonic_secs()` (4 syscall lines) with
  `clock_now_ms() / 1000` via stdlib chrono. The functional change
  works (all tests pass), but pulling in chrono adds the module to
  `[deps].stdlib` for the sake of saving 3 lines in one helper. **Net
  cost > net win**, so the local `syscall(228, CLOCK_MONOTONIC, &ts)`
  pattern stays. The roadmap entry is now annotated with the trade-off
  so the question doesn't get re-litigated.

### Removed

- **`build/ai-hwaccel` untracked from git.** The compiled binary had been
  accidentally committed at an early release (`c173383 cleanup for
  release`) and re-resurfaced as a modified file after every local build
  because `/build/` in `.gitignore` only ignores *untracked* files. Removed
  from the index; the directory stays gitignored.

### Carried forward

- **`case N: { ... }` switch blocks** — separately attempted (and
  reverted) in a pre-2.1.2 spike. cc5 5.10.x's `PARSE_CASE` rejects
  enum-name labels (`case FAMILY_CPU:` → `expected number, got
  identifier`); the v5.10.48 enum-const-fold landed for
  `PARSE_ARRAY` / `PARSE_GVAR_ARR` only. Roadmap entry documents the
  upstream limitation; the if-chain dispatch in `accel_name` /
  `family_name` / `format_name` / `_gguf_file_type_name` /
  `requirement_satisfied` stays until cyrius extends the fold to
  case labels.

## [2.1.1] — 2026-05-10

**Rust-port parity verification + scaffolding cleanup.** No code changes;
documents the audit and removes stale defensive lines.

### Verified

- **Parity audit against the 1.x Rust crate** — every public API in
  `docs/migration.md`'s Rust↔Cyrius mapping table (21 entries) has a
  matching `fn` in `src/`. Spot-checked: `registry_detect`,
  `registry_detect_threaded`, `builder_all`/`builder_none`/`builder_with`,
  `reg_best_available` / `reg_total_memory` / `reg_has_accelerator`,
  `reg_plan_sharding` / `reg_suggest_quant`, `cached_registry_new` /
  `cached_get`, `lazy_new` / `lazy_by_family`, `profile_cuda` /
  `profile_memory_bytes` / `profile_accel_type`, `accel_is_gpu`,
  `model_can_run`, `detect_model_format` / `detect_format_from_bytes`.
  No drift since the 2.0.0 mapping was authored.
- **JSON `schema_version` still `2`** — wire-compatible with v1.x
  Rust output.
- **Cyrius-only additions confirmed live**: `requirement.cyr` (scheduling
  integration), `async_detect.cyr` (threaded backends), `cache.cyr` (TTL
  + disk variant), `lazy.cyr` (per-family deferred detection),
  `model_format.cyr` (SafeTensors/GGUF/ONNX/PyTorch).
- **Documented intentional gaps stay accurate**: Windows detection is
  the only post-Rust feature gap; reachable now that cc5 5.10.x ships
  a Win64 PE backend, tracked in the roadmap's 2.2.0 slot.

### Removed

- **`.gitignore`: `rust-old/` line** — `rust-old/` was deleted from the
  tree at the 2.0.0 release (see [2.0.0] § Removed). The defensive
  ignore line has been a no-op for the project's entire Cyrius
  lifetime; removed now that the parity audit confirms nothing depends
  on the historical directory.
- **`.gitignore`: `tarpaulin-report.html`, `criterion/`** — Rust-era
  coverage / benchmark artifacts. No Rust toolchain runs against this
  tree anymore.

## [2.1.0] — 2026-05-10

**Slot 1 of the cc5 adoption arc — test reorg + CI tighten.** Pure
structural work; no source-code feature changes, no API surface changes
for consumers. Sets up cleaner ground for the `#derive(accessors)` /
multi-return / switch-block refactors planned later in the arc.

### Changed

- **Tests renamed to descriptive names + relocated under `tests/tcyr/`.**
  Phase numbers (`test_phase1.cyr` … `test_phase11.cyr`) mapped to no
  concept readable from the source; replaced with mission-named units:
  - `foundation_test.tcyr`     (errors, accel types, units, quantization)
  - `profile_test.tcyr`        (profile construction + setters)
  - `registry_test.tcyr`       (registry assembly)
  - `detect_gaudi_test.tcyr`   (Gaudi detection)
  - `detect_neuron_test.tcyr`  (Neuron detection)
  - `sharding_test.tcyr`       (plan / training)
  - `system_io_test.tcyr`      (sysfs / proc reading)
  - `cost_model_test.tcyr`     (cost / recommend)
  - `json_output_test.tcyr`    (JSON serialization)
  - `model_format_test.tcyr`   (SafeTensors / GGUF / ONNX / PyTorch)
  - `requirement_test.tcyr`    (requirement matching)

  Extension change `.cyr` → `.tcyr` matches the agnosys / libro /
  cyrius-internal convention for standalone test units. Content is
  byte-identical to the 2.0.1 phase files — git tracks them as pure
  renames. All 518 assertions still pass.

### CI

- **`cyrius vet` step added** — walks every `include` from `src/main.cyr`
  and reports `<N> deps, <untrusted>, <missing>`. Today: `36 deps, 0
  untrusted, 0 missing`. Catches drift between the in-tree include
  graph and the manifest snapshot the moment a new module lands without
  being wired into main.cyr (and vice versa).
- **Test loop iterates `tests/tcyr/*.tcyr`** instead of the
  `test_phase*.cyr` glob.
- **fmt drift sweep widened** to cover `tests/tcyr/*.tcyr`,
  `fuzz/*.fcyr`, `benches/*.bcyr` alongside `src/`.
- **`cyrius.lock` step reworded** — was misleadingly called "soft-skip".
  cyrius only writes a lockfile for `[deps.<git>]` entries, and
  ai-hwaccel is stdlib-only, so there is nothing to verify until a git
  dep gets added. The step stays in place so verification engages
  automatically the day that happens.

### Not yet adopted (carried in roadmap)

- `cyrius capacity --check` — stalled on toolchain. cc5 5.10.x's
  capacity doesn't honour the manifest `[deps].stdlib` auto-prepend,
  so it warns on every stdlib symbol against `src/main.cyr`. Revisit
  when the warning floor is 0.
- `lib/test.cyr` adoption — current local `assert` helpers work; the
  `"N passed, M failed"` summary line is what CI greps for. Pure
  cleanup, not unblocking anything.

## [2.0.1] — 2026-05-10

**Toolchain bump — pure mechanical slot.** No source-code feature changes,
no API surface changes for consumers. Compiles clean against the cc5
generation; sets up the scaffolding 2.1.0's adoption arc needs.

### Changed

- **Cyrius pin: 3.10.0 → 5.10.34.** Two majors (3 → 4 → 5) and ten minors
  absorbed in one bump. Highlights of what landed upstream during that
  window (none adopted yet — see 2.1.0 roadmap):
  - **4.0.0** (2026-04-13): `#derive(accessors)`, native multi-return
    (`return (a, b)` / `var x, y = fn()`), `case N: { ... }` switch
    blocks with scoped vars, defer-on-all-paths.
  - **5.0.0** (2026-04-15): cc5 generation — basic-block IR + CFG +
    LASE between parse and emit (transparent today, optimization-ready).
    `cyrius.cyml` becomes the first-class manifest.
  - **5.1 → 5.10.34**: `${file:VERSION}` interpolation, `cyrius.lock`
    hash verification, `cyrius vet` / `cyrius capacity` / `cyrius audit`,
    stricter `cyrius fmt`, 20+ new stdlib modules
    (regex, toml, sandhi, net, audit_walk, test, …).
- **Manifest: `cyrius.toml` → `cyrius.cyml`** with `version = "${file:VERSION}"`
  interpolation. `VERSION` is now the single source of truth — no more
  multi-file sed in `scripts/version-bump.sh`. Adds `repository` field.
- **`lib/` vendored stdlib removed from tree** and `/lib/` gitignored.
  `cyrius deps` repopulates it from the version-pinned `[deps].stdlib`
  snapshot in `cyrius.cyml`. Matches agnosys / libro / patra / yukti
  convention. Run `cyrius deps` after a fresh clone or cyrius upgrade.
- **`.cyrius-toolchain` retired.** CI now reads the pin directly from
  `cyrius.cyml`'s `cyrius = "X.Y.Z"` line — one source of truth, no
  drift between manifest and toolchain file.
- **`scripts/version-bump.sh` simplified to a one-liner.** Writes only
  `VERSION`; the manifest auto-tracks via `${file:VERSION}`. Reminder
  to add the CHANGELOG section before tagging.
- **`CLAUDE.md`** — compiler reference updated to cc5 5.10.34. New
  principles documented: vendored stdlib in `lib/` is gitignored;
  bump only `VERSION`.

### CI / Release workflows

- **Canonical installer** — `.github/workflows/{ci,release}.yml` now
  install Cyrius via `https://raw.githubusercontent.com/MacCracken/cyrius/main/scripts/install.sh`
  rather than ad-hoc tarball fetch. Lays out `~/.cyrius/{bin,lib,versions}`
  with the symlinks `cyrius deps` expects (without those, stdlib
  resolution fails with `cannot read ./lib/<name>.cyr`).
- **`cyrius deps` step** added before any compile, with `cyrius deps --verify`
  gating against `cyrius.lock` (soft-skips on first push that introduces
  a new dep before the lockfile lands).
- **fmt drift gate** — `cyrius fmt <file>` emits to stdout and CI diffs
  against the committed file. Catches over-indented blocks the older
  fmt tolerated.
- **Version consistency gate** — asserts `cyrius.cyml` carries the
  `${file:VERSION}` literal and the `VERSION` value appears in
  `CHANGELOG.md`. Release workflow additionally enforces tag ↔ VERSION
  match (accepts both `vX.Y.Z` and `X.Y.Z` tag styles).
- **Multi-arch release** — best-effort aarch64 cross-build when
  `cc5_aarch64` is in the toolchain bundle. Releases ship src tarball
  + per-arch binaries + SHA256SUMS, with the `## [x.y.z]` CHANGELOG
  section auto-extracted as the release body.

### Fixed

- **`cyrius fmt` drift in 4 files** — `src/json_out.cyr`,
  `src/model_format.cyr`, `tests/test_phase10.cyr`,
  `tests/test_phase11.cyr` had inner blocks over-indented by 4 spaces
  past their opening brace (a pattern the older 3.x fmt tolerated;
  cc5 `cyrius fmt` enforces strict 4-space-per-scope). Test files also
  had assertion-message continuations indented under the call's arg
  list instead of aligned to the argument column. All four re-formatted
  in place; entire `src/` / `tests/` / `fuzz/` / `benches/` swept clean.
- **`str_contains` API migration** — cc5's `lib/str.cyr` changed the
  second argument from cstr to `Str` (v5.10.25 generalize pass). The
  cstr-needle behavior moved to `str_contains_cstr`. Migrated 14 files
  (11 detection backends + 3 test phases) — every site passing a cstr
  literal as the needle. `src/model.cyr:172` already passed a `Str` and
  was left unchanged. The old call sites compiled silently against
  3.10.0 because the type system didn't catch the i64 vs Str mismatch;
  the new stdlib still doesn't reject it but treats the cstr pointer
  as a `Str` struct address, reading garbage as length — every detect
  branch that gated on `str_contains(...)` was effectively a no-op
  until this fix.
- **`str_from(str_data(json))` round-trip removed in test_phase9** — cc5
  registers `str_from_int` as `str_from`'s `_int` overload, so
  `str_from(str_data(...))` (where `str_data` returns `i64`) routes to
  `str_from_int` and produces a `Str` containing the decimal string of
  the pointer value (e.g. `"727597728"`) instead of wrapping the data.
  The round-trip was always semantically pointless — `str_builder_build`
  already returns a `Str`. Six call sites in test_phase9 collapsed to
  use the builder result directly; restored 17 previously-passing
  assertions across `test_profile_json` / `test_registry_json` /
  `test_summary_json` / `test_json_with_warnings`.

## [2.0.0] - 2026-04-13

### Breaking

Complete rewrite from Rust to [Cyrius](https://github.com/MacCracken/cyrius).
The Rust crate (`ai-hwaccel` on crates.io) is superseded by a native Cyrius
binary with zero external dependencies. API surface is equivalent but calling
conventions have changed from Rust method syntax to Cyrius function calls.

**Migration**: `AcceleratorRegistry::detect()` → `registry_detect()`.
See `docs/benchmarks-rust-v-cyrius.md` for full API mapping.

### Added

- **Cyrius port** — entire codebase rewritten in Cyrius (v3.10.0). 37 source
  modules (18 core + 19 detect), 5,602 LOC. Zero external dependencies.
  Binary: 217 KB (was 708 KB Rust release). Compile time: 215 ms (was ~1.8s).
- **`model_format.cyr`** — model file format detection from headers. Parses
  SafeTensors (JSON header → param count, dtype, tensor count), GGUF (magic +
  version + tensor count + file_type metadata), ONNX (protobuf ir_version
  validation), and PyTorch (ZIP magic). Reads only first 16 KB. Both
  file-path and byte-slice APIs.
- **`requirement.cyr`** — accelerator requirement matching for scheduling
  integration. 7 requirement types: None, GPU, TPU (with min_chips), Gaudi,
  AwsNeuron, GpuOrTpu, AnyAccelerator. `requirement_satisfied()`,
  `find_satisfying_profile()`, `count_satisfying()`.
- **`async_detect.cyr`** — threaded concurrent hardware detection. CLI-based
  backends (nvidia-smi, hl-smi, vulkaninfo, neuron-ls, xpu-smi,
  system_profiler) run in parallel threads via `thread.cyr`. Sysfs-only
  backends run on the main thread. Results merged after join.
  `registry_detect_threaded()` API.
- **`cache.cyr`** — detection result caching with configurable TTL.
  `cached_registry_new(ttl_secs)`, `cached_get()`, `cached_invalidate()`.
  Mutex-protected, thread-safe. `DiskCachedRegistry` variant writes JSON to
  `~/.cache/ai-hwaccel/registry.json` with atomic write (temp + rename).
- **`lazy.cyr`** — per-family lazy detection. Defers backend probing until a
  specific accelerator family is queried. `lazy_new()`, `lazy_by_family()`,
  `lazy_into_registry()`. Avoids spawning nvidia-smi when only TPU info needed.
- **`model.cyr` extensions** — `models_by_family()` for family-based filtering,
  `model_headroom_x100()` for spare memory percentage,
  `compatible_with_registry()` for registry-aware compatibility.
- **`cost.cyr` + `model.cyr` re-included** — previously excluded due to Cyrius
  compiler fixup table overflow (4096 limit). Compiler v3.7.0 expanded limit
  to 16384, unblocking inclusion. `--cost` CLI flag now uses real
  `recommend_instances()` instead of a stub.
- **`json_out.cyr`** — JSON serialization via `str_builder` (replaces serde).
  `registry_to_json()`, `registry_to_summary_json()`, `profile_to_json()`,
  `model_meta_to_json()`.
- **11 test phases** — 518 assertions covering all modules. Phase 10: model
  format detection (46 assertions). Phase 11: requirement matching (27
  assertions).
- **6 fuzz harnesses** — `cuda_parser.fcyr`, `model_format.fcyr`,
  `vulkan_parser.fcyr`, `neuron_parser.fcyr`, `apple_parser.fcyr`,
  `gaudi_parser.fcyr`. Edge cases: empty input, garbage data, truncated
  headers, adversarial bytes.
- **3 benchmark suites** — core (8 benchmarks: memory estimation, quantization,
  training), parsing (5: CUDA CSV, Vulkan, Neuron JSON, SafeTensors, GGUF),
  registry (7: queries, sharding, JSON serialization). 20 benchmarks total.
- **`bench-history.sh`** — rewritten for Cyrius. Appends results to CSV with
  timestamp, commit, branch.
- **Str auto-coercion** — 61 `str_from("literal")` wrappers removed across
  src/ and tests/ using Cyrius v3.6.0 `: Str` parameter annotations.
- **Vendored stdlib synced** — all 26 vendored modules identical to upstream
  Cyrius. Added: `thread.cyr`, `async.cyr`, `fnptr.cyr` (fncall3/4),
  `freelist.cyr`. Synced: `str.cyr`, `syscalls.cyr`, `hashmap.cyr`, and 9
  others.

### Changed

- **Language**: Rust → Cyrius 3.10.0. No LLVM, no cargo, no crates.io.
- **Binary size**: 708 KB → 217 KB (**-69%**).
- **Compile time**: ~1.8s → 215 ms (**-88%**).
- **Source LOC**: 11,278 → 5,602 (**-50%**).
- **Dependencies**: 131 crates → 0 (**-100%**).
- **Detection modules consolidated**: cerebras + graphcore + groq → `cloud_asic.cyr`,
  qualcomm + samsung + mediatek → `edge.cyr`, intel_npu + intel_oneapi → `intel.cyr`.
- **Hardware types unified**: `hardware/mod.rs` + `hardware/*.rs` → `types.cyr`
  (enums, classification, throughput multipliers, HBM lookups, rank — all in one file).
- **Sharding merged**: `sharding.rs` + `plan.rs` → `plan.cyr`.
- **Fixed-point arithmetic** throughout — `x1000` multipliers replace all
  floating-point operations. No floats in the entire codebase.
- **Lint limit**: 100 → 120 characters (Cyrius 3.8.0). `#skip-lint` for
  unavoidable long strings.
- **`--cost` mode**: now shows real cloud instance recommendations with pricing
  (was stub directing to JSON file).

### Removed

- **Rust source** — `rust-old/` directory removed. Final Rust benchmarks
  preserved in `docs/benchmarks-rust-v-cyrius.md`.
- **FFI bindings** (`ffi.rs`) — not applicable, Cyrius is native code.
- **Windows detection** (`detect/windows.rs`) — Cyrius doesn't target Windows
  yet (v4.0.0 roadmap).
- **Cargo/crates.io** — no `Cargo.toml`, `Cargo.lock`, or Rust toolchain files.
- **serde/tokio/tracing** — replaced by manual JSON, thread.cyr, and direct
  stderr output respectively.

### Fixed

- **`requirement.cyr` undefined function** — `profile_tpu_chip_count()` called
  but function is `profile_tpu_chips()`. Would have crashed at runtime. Caught
  by Cyrius 3.10.0 undefined function diagnostic.
- **`async_detect.cyr` undefined functions** — `enrich_disk()` (should be
  `detect_storage()`), `detect_interconnect()` (should be
  `detect_interconnects()`). Same diagnostic.
- **`lazy.cyr` undefined function** — `builder_enable()` (should be
  `builder_with()`). Same diagnostic.

### Performance

All benchmarks on the same machine. Cyrius compiles to direct x86_64 without
LLVM optimization (no jump tables, no register allocation, no LTO).

| Benchmark | Rust (LLVM) | Cyrius | Ratio |
|-----------|-------------|--------|-------|
| estimate_memory 70B BF16 | 256 ps | 8 ns | 31x |
| bits_per_param (5 levels) | 295 ps | 3 ns | 10x |
| train_7B_full_gpu | 3.29 ns | 30 ns | 9x |
| parse_cuda_output 8gpu | 5.80 µs | 18 µs | 3x |
| parse_vulkan_2gpu | 1.85 µs | 3 µs | 1.6x |
| best_available (13 dev) | 39.56 ns | 722 ns | 18x |
| total_memory (13 dev) | 7.88 ns | 122 ns | 15x |
| json_serialize (13 dev) | 4.89 µs | 27 µs | 6x |
| detect_safetensors | — | 931 ns | new |
| detect_gguf | — | 498 ns | new |

All times sub-microsecond to microsecond. Detection dominated by 100ms+ CLI
tool execution — per-call overhead is irrelevant to end-to-end latency.

## [1.2.0] - 2026-04-05

### Added

- **NVSwitch auto-detection** — probes sysfs
  `/sys/devices/virtual/nvidia-nvswitch/` and parses `nvidia-smi topo -m`
  output. High NVLink counts (NV8+) with multiple GPUs indicate NVSwitch
  fabric. Both sync and async detection paths supported.
- **AMD XGMI / Infinity Fabric detection** — reads XGMI hive IDs from
  `/sys/class/drm/card*/device/xgmi_hive_info` and parses
  `rocm-smi --showtopo` link-type matrix as fallback.
- **Google TPU ICI detection** — multi-chip TPU configurations now report
  `InterconnectKind::Ici` with version-specific bandwidth (v4: 192 GB/s/chip,
  v5e: 204.8, v5p: 409.6).
- **RoCE v2 detection** — new `InterconnectKind::RoCEv2` variant. Distinguished
  from RoCE v1 by reading sysfs `gid_attrs/types` for "RoCE v2" entries.
- **Fuzz targets** for NVSwitch topology and XGMI topology parsers.
- **Model compatibility database** (`model_compat` module) — embedded catalogue
  of 26 popular models (Llama, Mistral, Gemma, Phi, Qwen, DeepSeek, Falcon,
  etc.) with `can_run()`, `compatible_models()`, `find_model()`,
  `compatible_with_registry()` APIs. Answers "can I run Llama 70B on 2x RTX
  4090?" without manual memory math.
- **Model format detection** (`model_format` module) — parses `.safetensors`,
  `.gguf`, `.onnx`, and `.pt` file headers to extract format, parameter count,
  data type, and tensor count. Both file-path and byte-slice APIs (WASM-safe).
- **WASM target support** — `wasm32-unknown-unknown` builds cleanly with all
  features. `from_profiles()`, `from_json()`, planning, sharding, cost,
  training, model compat, and model format detection all work in WASM.
- **Kubernetes GPU detection** — detects GPU devices allocated via Kubernetes
  device plugins (`NVIDIA_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`,
  `GPU_DEVICE_ORDINAL`). New `KubernetesGpuInfo` struct in
  `RuntimeEnvironment`.
- **What-if analysis** — `what_if_add()`, `what_if_remove()`,
  `what_if_replace()` methods on `AcceleratorRegistry` for simulating hardware
  changes and re-planning sharding strategies.
- **Fuzz target** for model format byte-level parser.
- **130+ new tests** — interconnect parsers (NVSwitch, XGMI, ICI, RoCE v2),
  model compatibility database, model format detection (SafeTensors, GGUF,
  ONNX, PyTorch), what-if analysis, edge cases. Total: 538 tests.

### Changed

- **`DetectBuilder` uses `u32` bitmask** — replaces `Vec<bool>` with zero-alloc
  bitmask. `DetectBuilder` is now `Copy`. `enabled_count()` uses
  `count_ones()` instead of iterator filter.
- **Sharding planner now accounts for all interconnect types** —
  `InterconnectInfo::scan()` incorporates InfiniBand, RoCE, RoCEv2, and ICI
  bandwidth into `high_bw` for better sharding decisions.
- **Sharding throughput uses `reduce()` instead of `fold(INFINITY)`** —
  prevents NaN/Inf when device lists are empty. Throughput estimates now
  return `None` instead of `Some(NaN)` when non-finite.
- **Single-pass Vulkan/dedicated GPU check** — combined 3 separate iterator
  passes into one `fold` in `detect_with_builder` and timed variant.
- **Simplified `read_dir` patterns** — replaced 12 `read_dir().into_iter()
  .flatten().flatten()` double-flatten patterns across 5 files with idiomatic
  `let Ok(entries) = read_dir() else { return }`.
- **Async interconnect detection collects warnings** — CLI tool errors
  (non-tool-not-found) are now propagated as warnings in the async path.

### Fixed

- **`parse_nvswitch_topo` header row inflation** — the GPU column header
  (`GPU0 GPU1 ...`) was incorrectly counted as a data row, inflating the
  reported GPU count by 1.
- **`detect_xgmi_sysfs` false positive** — returned `true` (suppressing CLI
  fallback) even when all XGMI hives had only 1 GPU.
- **`detect_tpu_ici` non-TPU device counting** — non-TPU accelerator devices
  under `/dev/accel` are now skipped by requiring `tpu_version` sysfs file.

### Security

- **`read_sysfs_string` hard cap** — added 1 MiB absolute maximum to prevent
  DoS from callers passing huge `max_bytes` values.
- **NVLink link-count cap** — `parse_nvlink_output` caps at 256 links per GPU
  to bound bandwidth accumulation from malformed output.

### Dependencies

- Certified `semver 1.0.28` and `fastrand 2.4.0` via `cargo vet`.

## [1.1.1] - 2026-04-03

### Fixed

- **Division-by-zero in cost headroom** — `recommend_instance` no longer
  produces NaN when an instance has `total_gpu_memory_gb == 0`.
- **Invalid layer range in pipeline sharding** — tiny models (< 250M params)
  with multiple pipeline stages could produce shards where `start > end`;
  layer ranges are now clamped to valid bounds.
- **`DetectionError::ToolFailed` Display allocation** — removed intermediate
  `String` allocation when formatting exit codes.

### Changed

- **`QuantizationLevel::bits_per_param()` and `memory_reduction_factor()` are
  now `const fn`** — enables use in const contexts.
- **Added `#[must_use]` attributes** to `plan_sharding()`, `total_memory()`,
  `has_accelerator()`, `available()`, `by_family()`, `satisfying()`, and
  `ShardingPlan::shards()`.
- **Replaced magic numbers** — `Display` impls in `ShardingPlan` and
  `AcceleratorProfile` now use `units::BYTES_PER_GIB` instead of
  `1024.0 * 1024.0 * 1024.0`.
- **`cost::all_instances()`** now logs `tracing::warn` on malformed embedded
  pricing JSON instead of silently returning empty.
- Updated doc examples to reference `version = "1.1"` (was `"0.19"`).

### Dependencies

- criterion 0.5.1 → 0.8.2 (dev-dependency, major version bump)
- criterion-plot 0.5.0 → 0.8.2
- itertools 0.10.5 → 0.13.0 (transitive)
- Removed transitive deps: is-terminal, hermit-abi

---

## [1.1.0] - 2026-04-03

### Changed

- **License: AGPL-3.0-only → GPL-3.0-only** — updated Cargo.toml, deny.toml,
  CLAUDE.md, and LICENSE file. Removes the network-use copyleft clause.
- **`available()`, `by_family()`, `satisfying()` return `impl Iterator`** —
  zero-alloc queries for callers using `.count()`, `.any()`, `.next()`.
  Callers needing a `Vec` use `.collect()` explicitly. Benchmarked: 3–35x
  faster for non-materializing callers; `.collect()` path unchanged.
- **Detection macro consolidation** — 6 local macros (`run_backend!`,
  `spawn_backend!`, `run_backend_timed!`, `spawn_backend_timed!`,
  `spawn_async_backend!`, `run_sysfs!`) replaced by 3 registration table
  macros (`backend_table!`, `async_cli_backends!`, `sysfs_backends!`) with
  local callback dispatch. Adding a new backend is now a 1-line table entry
  instead of editing 6 locations.
- **Watch mode allocations reduced** — delta tracking uses index+Display
  keys instead of Debug format, avoiding per-tick `format!("{:?}")` allocs.
- **`tracing-subscriber` slimmed** — dropped `env-filter` (regex engine,
  ~347 KB .text) and `json` (tracing-serde) features. `EnvFilter` replaced
  with simple `LevelFilter` match on `RUST_LOG`. `--json-log` flag removed.
- **Release profile optimized** — `lto = true`, `codegen-units = 1`,
  `strip = true`, `panic = "abort"`, `opt-level = "z"`. Binary size:
  2.6 MB → 838 KB (68% reduction).

### Added

- **Dual iterator/collect benchmarks** — `registry_queries` group now
  benchmarks both `_count` (lazy) and `_collect` (materialized) variants
  for `available()`, `by_family()`, and `satisfying()` to transparently
  show the allocation cost difference.

### Fixed

- Cleaned up unused license allowances in `deny.toml` (BSD-2-Clause,
  BSD-3-Clause, ISC, Unicode-DFS-2016 were not in the dependency tree).
- Pruned unnecessary `cargo-vet` exemptions.
- Certified 14 updated dependency versions for `cargo-vet`.

### Dependencies

- indexmap 2.13.0 → 2.13.1
- js-sys 0.3.91 → 0.3.94
- libc 0.2.183 → 0.2.184
- mio 1.1.1 → 1.2.0
- proptest 1.10.0 → 1.11.0
- tokio 1.50.0 → 1.51.0
- tokio-macros 2.6.1 → 2.7.0
- wasm-bindgen 0.2.114 → 0.2.117
- web-sys 0.3.91 → 0.3.94
- zerocopy 0.8.47 → 0.8.48

---

## [1.0.0] - 2026-03-27

### Breaking Changes

- **`AcceleratorType` is now `Copy`** — `device_name: String` moved from
  `VulkanGpu` variant into `AcceleratorProfile::device_name: Option<String>`.
  `VulkanGpu` is now `VulkanGpu { device_id: u32 }`. All `.clone()` calls on
  `AcceleratorType` are eliminated. Callers that matched on
  `VulkanGpu { device_name, .. }` should read `profile.device_name` instead.
- **`ShardingPlan::shards` is now `pub(crate)`** — use `plan.shards()` accessor
  instead of direct field access.
- **`cost::CloudInstance` renamed to `cost::CloudGpuInstance`** — the actual
  type is now `CloudGpuInstance`, removing the `as` alias in re-exports.
- **`system_io::CloudInstance` renamed to `system_io::CloudInstanceMeta`** —
  disambiguates from the cost pricing type.

### Added

#### API

- `TryFrom<u32>` for `QuantizationLevel` — map `32 → None`, `16 → Float16`,
  `8 → Int8`, `4 → Int4`. Returns `Err(bits)` for unsupported values.
- `AcceleratorProfile::device_name: Option<String>` — human-readable device
  name (e.g. "RTX 4090"), populated by CUDA and Vulkan detectors.
- `#[non_exhaustive]` on `ShardingStrategy`, `TrainingMethod`, `TrainingTarget`,
  `InterconnectKind`, `StorageKind`, `CloudProvider`.
- `#[must_use]` on 18 pure public methods across registry, profile,
  quantization, requirement, training, cost, sharding, and plan modules.
- `#[inline]` on 9 additional hot-path getters.
- `Backend::WindowsWmi` variant with `with_windows_wmi()` /
  `without_windows_wmi()` builder methods.

#### Platform & Validation

- **Cloud hardware validation fixtures** — realistic parser tests for
  A100 80GB 8-GPU, H100 80GB SXM, Grace Hopper GH200 (unified memory),
  Gaudi 3 8-device, Neuron trn1.32xlarge/inf2.48xlarge, MI300X 192GB.
  Planning pipeline tests for 8x A100 sharding, TPU v5p 256-chip pod,
  TPU v5e 4-chip, 8x Gaudi3, 8x MI300X.
- **macOS `system_profiler -json` GPU detection** — `parse_displays_json()`
  parses `SPDisplaysDataType -json` for GPU name, vendor, Metal family,
  core count, discrete VRAM. `parse_sysctl_output()` for CPU topology
  (memory, core count, perf/efficiency cores).
- **Windows WMI GPU detection** — new `detect/windows.rs` module behind
  `windows-wmi` feature flag. `parse_wmic_output()` for
  `Win32_VideoController` CSV, `parse_powershell_csv()` for
  `Get-CimInstance` fallback. `find_nvidia_smi_windows()` for path resolution.
- **Platform abstraction trait** — `PlatformProbe` trait in
  `detect/platform.rs` abstracting filesystem reads, command execution,
  device enumeration, and system memory. `LivePlatform` + `MockPlatform`.
- **Feature profiles**: `minimal` (CPU-only) and `common`
  (cuda+rocm+apple+vulkan+intel-npu) feature sets.

#### Testing

- **471 tests** (up from 358): cloud hardware fixtures, ASIC quantization
  coverage, macOS/Windows parser tests, platform trait tests, planning
  pipeline tests, `TryFrom<u32>` tests.

### Changed

- `AcceleratorType` derives `Copy` — zero-cost pass-by-value for all 19
  hardware variants.
- `AcceleratorProfile::Display` includes device name when present.
- CLI decomposed: `print_table()` split into `filter_profiles()`,
  `sort_profiles()`, `render_header()`, `render_row()`, `render_footer()`.
  `handle_cost_mode()` and `handle_profile_mode()` extracted.
- `Column` type gains `header()`, `width()`, `is_left_aligned()` methods.
- `parse_csv_line()` shared CSV helper for cuda/gaudi/intel_oneapi parsers.

### Fixed

- Scaffold hardening audit: all public enums now `#[non_exhaustive]`, all pure
  functions `#[must_use]`, all hot-path getters `#[inline]`.

---

## [0.23.3] - 2026-03-23

### Added

#### Benchmark infrastructure

- **Benchmark history tracking**: `scripts/bench-history.sh` captures criterion
  results to `bench-history.csv` with 7-column format (timestamp, commit,
  branch, benchmark, low_ns, estimate_ns, high_ns). Auto-generates
  `benchmarks.md` with 3-point trend tables (baseline → previous → current).
- **95 benchmarks across 16 groups**: detection, parsing, planning, training,
  cost, quantization, registry queries, caching, lazy detection, large-registry
  sharding, and JSON serialization.
- **New bench files**: `benches/training.rs`, `benches/cost.rs`,
  `benches/quantization.rs`, `benches/parsing.rs`, `benches/registry.rs`.
- `make bench` target for running the full benchmark suite.

#### Testing

- **358 tests** (up from ~280): added FFI module tests (11), async detection
  tests (5), parser fixture tests for all backends (Vulkan summary, Apple
  system_profiler, Gaudi multi-device, CUDA edge cases, Neuron JSON, Intel
  oneAPI CSV, Cerebras memory, Graphcore memory), and named-constant
  verification tests.
- **23 test modules** covering all public API surface.

#### API

- `DetectBuilder::with(Backend)` / `without(Backend)` — generic methods for
  enabling/disabling backends. Existing `with_cuda()` etc. are now inline
  wrappers.
- `ShardingPlan::shards()` accessor method.
- `Default` derive for `ShardingStrategy`, `TrainingMethod`, `TrainingTarget`.
- `Display` impl for `MemoryEstimate`.
- `Default` impl for `AcceleratorProfile` — simplifies construction with
  `..Default::default()`.

### Changed

- **`plan_sharding()` decomposed** into `InterconnectInfo::scan()`,
  `build_tpu_tensor_plan()`, `build_gpu_tensor_plan()`, `build_pipeline_plan()`
  helper functions. Main method is now a dispatcher (~40 lines vs 225).
- **`suggest_quantization()` precomputes estimates**: 4 calls instead of up to 9
  redundant `estimate_memory()` invocations.
- **`/dev` device iteration helpers**: `iter_dev_devices()` and
  `has_dev_device()` replace ~70 lines of duplicated `/dev` scanning across 8
  backends (neuron, tpu, groq, cerebras, graphcore, qualcomm, samsung, mediatek).
- **`..Default::default()` in all detector profiles**: 22 profile constructions
  across 15 files simplified, eliminating ~100 lines of explicit `None` fields.
- **Detection modules made public**: `detect::bandwidth`, `detect::interconnect`,
  `detect::pcie`, `detect::cuda`, `detect::gaudi`, `detect::vulkan` — enables
  external benchmarking and testing of parsing functions.

### Performance

- `#[inline]` on 12 hot-path methods: `bits_per_param`, `memory_reduction_factor`,
  `is_gpu/npu/tpu/ai_asic`, `family`, `throughput_multiplier`,
  `training_multiplier`, `supports_training`, `has_interconnect`.
- **Single-pass interconnect scan** in `plan_sharding()` — combined 3 iterator
  passes into 1 `for` loop with `match`.
- **Direct JSON deserialization** in `cost.rs` — eliminated intermediate
  `serde_json::Value` clone.
- **Deferred string allocation** in CUDA parser — `&str` until non-empty check.
- **Filter-before-clone** in environment detection — AWS instance fields.
- **ROCm sysfs filter-before-alloc** — trim-then-check avoids empty String alloc.
- **Disk detection deferred `to_string()`** — skip checks use `&str` reference.

### Fixed

- **Integer overflow in Graphcore parser** (`parse_memory_from_gcinfo`): fuzz
  input with huge MB/GB values caused `u64` multiply overflow. Now uses
  `saturating_mul`. Also fixed in Cerebras and Apple memory parsers.
- **Fuzz CI timeout**: reduced per-target fuzz time from 30s to 15s (11 targets),
  added `timeout-minutes: 15` job limit.
- **Clippy `len_zero`**: `registry.all_profiles().len() >= 1` replaced with
  `!is_empty()` in async detection tests.
- Removed dead `use std::path::Path` import in TPU detector.
- Removed 3 unnecessary `return;` statements in Samsung/MediaTek/Qualcomm
  detectors.

### Exports

- `units` module (named constants for hardware math).

---

## [0.21.3] - 2026-03-23

### Added

#### Detection performance

- **Lazy detection**: `LazyRegistry::new()` defers backend probing until a
  specific accelerator family is queried. Avoids spawning `nvidia-smi` when
  the caller only needs TPU info.
- **vulkaninfo timeout + caching**: 3s subprocess timeout (down from 5s).
  Results cached to `$XDG_CACHE_HOME/ai-hwaccel/vulkan.json` with 60s TTL.
  Falls back to sysfs-only detection on timeout.
- **Sysfs-only Vulkan fallback**: Detects GPUs via
  `/sys/class/drm/card*/device/{vendor,device}` with PCI ID lookup table.
  Covers NVIDIA, AMD, and Intel GPUs without spawning `vulkaninfo`.
- **Detection result disk caching**: `DiskCachedRegistry::new(ttl)` persists
  full registry to `$XDG_CACHE_HOME/ai-hwaccel/registry.json` with atomic
  writes (temp+rename) to prevent multi-process corruption.
- **Per-backend timing**: `AcceleratorRegistry::detect_with_timing()` returns
  `TimedDetection` with per-backend `Duration` map. CLI: `--profile` flag.

#### Planning

- **Topology-aware sharding**: `plan_sharding()` now prefers tensor parallel
  for NVSwitch-connected groups or high-bandwidth NVLink (>100 GB/s). Pipeline
  parallel orders stages by NUMA locality. Throughput estimates account for
  interconnect overhead.
- **Cost-aware planning**: Static pricing table in `data/cloud_pricing.json`
  (18 instances across AWS/GCP/Azure). `cost::recommend_instance()` returns
  cheapest viable cloud instance. CLI: `--cost 70B --quant bf16`.

#### Platform

- **Container/VM detection**: Detects Docker, Kubernetes, and cloud provider
  (AWS/GCE/Azure) via DMI sysfs. Exposed as `SystemIo::environment`.
  No HTTP metadata calls — purely filesystem-based.

#### Python bindings (groundwork)

- **PyO3 module scaffold**: `py/` directory with maturin build wrapping
  `detect()`, `suggest_quantization()`, `plan_sharding()`, `system_io()`,
  `estimate_training_memory()`.
- **Type stubs**: `ai_hwaccel.pyi` for IDE support.
- **Examples**: `basic_detect.py`, `sharding_plan.py`, `training_memory.py`.

### Changed

- **Schema version**: v2 → v3 (new `environment` field in `SystemIo`).
  Old v1/v2 JSON deserializes cleanly with `environment: None`.
- **Pipeline parallel throughput**: Now scales by `num_stages` with
  interconnect overhead factor (15% NVLink, 35% PCIe-only).

### Performance

- **cost.rs OnceLock**: Pricing JSON parsed once per process (was re-parsing
  on every `recommend_instance()` call).
- **CachedRegistry lock scope**: Mutex released before running `detect()` —
  concurrent readers no longer blocked during detection.
- **DMI caching**: Cloud detection reads DMI files once, shares across
  AWS/GCE/Azure detectors (was 6-7 redundant sysfs reads).
- **read_sysfs_string**: Heap path avoids `.to_vec()` double-allocation.
- **list_driver_pci_addrs**: Uses `Path::join()` and byte-level validation.
- **Atomic cache writes**: Disk cache uses temp+rename to prevent corruption.

### Exports

- `LazyRegistry`, `DiskCachedRegistry`, `TimedDetection`
- `CloudGpuInstance`, `CloudProvider`, `InstanceRecommendation` (cost module)
- `RuntimeEnvironment`, `CloudInstance` (system_io)

---

## [0.20.3] - 2026-03-19

### Added

#### System I/O and monitoring

- **VRAM bandwidth probing**: `AcceleratorProfile::memory_bandwidth_gbps`
  calculates theoretical memory throughput from clock speed and bus width.
  NVIDIA via `nvidia-smi --query-gpu=clocks.max.memory` + compute capability
  lookup; AMD via sysfs `pp_dpm_mclk` + PCI device ID lookup. Includes
  fallback tables for known GPU specs.
- **Runtime VRAM usage**: `memory_used_bytes` and `memory_free_bytes` for
  CUDA (via `nvidia-smi`) and ROCm (via sysfs).
- **PCIe link detection**: `pcie_bandwidth_gbps` reads sysfs
  `current_link_width`/`current_link_speed` for CUDA and ROCm GPUs.
- **NUMA topology**: `numa_node` maps GPUs to their NUMA node via sysfs PCI
  device info.
- **Power and thermal monitoring**: `temperature_c`, `power_watts`,
  `gpu_utilization_percent` on `AcceleratorProfile`. CUDA via `nvidia-smi`
  (`temperature.gpu`, `power.draw`, `utilization.gpu`). ROCm via sysfs hwmon
  (`temp1_input`, `power1_average`, `gpu_busy_percent`).
- **Network interconnect detection**: `SystemIo::interconnects` detects
  InfiniBand and RoCE via `/sys/class/infiniband/`, NVLink via `nvidia-smi
  nvlink -s`. Exposes bandwidth and link state.
- **Disk I/O detection**: `SystemIo::storage` probes `/sys/block/*/queue/`
  to classify NVMe, SATA SSD, and HDD with estimated bandwidth.
- **Ingestion estimation**: `SystemIo::estimate_ingestion_secs()` estimates
  data loading time given dataset size and detected storage throughput.
- **New types**: `SystemIo`, `Interconnect`, `InterconnectKind`,
  `StorageDevice`, `StorageKind` — all serializable.

#### Detection improvements

- **AMD ROCm enrichment**: clock speeds (`pp_dpm_sclk`/`pp_dpm_mclk`),
  VBIOS version, GPU temperature, power draw, and utilization from sysfs.
  CXL-attached memory detection for MI300X/MI350.
- **Vulkan full parsing**: compute queue families, queue counts, and subgroup
  sizes from full `vulkaninfo` output (not just `--summary`).
- **NVIDIA Grace Hopper**: detects GH200/GH100 from GPU name, adds 480 GB
  unified LPDDR5X to reported HBM for capacity planning.

#### New backends (untested — written from documentation)

- **Cerebras WSE**: `cerebras_cli system-info` + `/dev/cerebras*` fallback.
- **Graphcore IPU**: `gc-info` JSON parsing + `/dev/ipu*` fallback.
- **Groq LPU**: `/dev/groq*` placeholder (driver not yet public).
- **Samsung NPU**: `/sys/class/misc/samsung_npu` + `/dev/samsung_npu*`.
- **MediaTek APU**: `/sys/class/misc/mtk_apu` + `/dev/mtk_mdla*`.

#### API and CLI

- **Schema v2**: `SCHEMA_VERSION` bumped to 2, formalizing all system I/O
  fields, power/thermal fields, and the `Timeout` error variant.
- **`DetectionError::Timeout`**: new error variant for timed-out tools,
  separate from `ToolFailed`. Enables programmatic retry logic.
- **True async detection**: `detect_async()` now uses
  `tokio::process::Command` for non-blocking subprocess I/O. CLI backends
  run as concurrent tokio tasks, sysfs-only backends in a single
  `spawn_blocking`.
- **`--columns`**: select specific table columns (`--columns name,mem,bw`).
- **`--tsv`**: tab-separated output for machine-readable table data.
- **`--watch` deltas**: memory usage changes shown between refreshes.
- **`--alert`**: threshold alerts during watch mode (`--alert mem>90`).
- **CLI table**: now shows Free VRAM, BW, PCIe, NUMA, plus Interconnects
  and Storage sections.

#### Testing

- **Hardware integration tests**: `tests/hardware_integration.rs` with 17
  tests covering CPU, ROCm, Vulkan, PCIe, bandwidth, storage, interconnects,
  JSON roundtrip, and concurrent detection. Auto-skips when hardware absent.
- **Fuzz testing**: 9 `cargo-fuzz` targets covering all CLI output parsers.
  Found and fixed integer overflow in CUDA memory parser.
- **Load testing**: concurrent 4-thread detection test + benchmark.
- **System I/O benchmarks**: per-backend, serialization, deserialization,
  and query benchmarks in `benches/detect.rs`.

#### Documentation

- **Troubleshooting guide**: `docs/troubleshooting.md`.
- **Performance tuning guide**: `docs/performance.md`.
- **Migration guide**: `docs/migration.md` (v0.19.3 → v0.20.3).
- **Crate-level docs**: expanded with error handling, custom backends, serde
  integration, and system I/O examples.

### Security

- **Subprocess environment sanitization**: `run_tool()` strips `LD_PRELOAD`,
  `LD_LIBRARY_PATH`, `DYLD_INSERT_LIBRARIES`, `DYLD_LIBRARY_PATH` from child
  processes to prevent library injection.
- **Windows `which()` improvements**: tries `.exe`, `.cmd`, `.bat` extensions
  when the tool name has no extension.
- **TOCTOU documentation**: the inherent time-of-check-time-of-use gap
  between path resolution and execution is documented as an accepted risk.

### Fixed

- **Integer overflow in CUDA parser**: `memory.used`/`memory.free` values
  exceeding u64 range on multiply now use `saturating_mul` with range filter.
  Found via fuzz testing.
- **Unbounded CSV field parsing**: CUDA parser now caps CSV splits to 20
  fields to prevent memory exhaustion from malicious `nvidia-smi` output.
- **Path traversal in PCI address handling**: PCI addresses in `pcie.rs`
  and `numa.rs` are now validated (hex+colon+dot only) and paths are
  canonicalized to prevent symlink-based information disclosure.
- **Grace Hopper memory validation**: unified memory is only added when
  reported HBM is in the realistic 80–100 GB range (prevents miscalculation
  from malformed nvidia-smi output).
- **Silent device ID fallback**: TPU and Neuron `/dev` parsers now skip
  malformed device names instead of silently mapping them to device 0.
- **Unbounded Vulkan device name**: `vulkaninfo` device names capped at
  256 characters to prevent memory exhaustion.
- **Defensive CSV bounds**: CUDA parser uses `.get()` for all field access
  instead of direct indexing.
- **Neuron JSON defaults removed**: malformed `neuron-ls` JSON devices are
  now skipped instead of using fabricated defaults (2 cores, 8192 MB).
- **Sysfs read size cap**: all sysfs reads across the codebase now use
  `read_sysfs_string()` with byte limits (64 B for values, 256 B for
  strings, 4 KiB for multi-line files, 64 KiB for /proc/meminfo).
  Handles sysfs pseudo-files correctly (they report 4096 as size regardless
  of content). Applied to: ROCm, TPU, PCIe, NUMA, interconnect, disk,
  bandwidth, neuron, and apple detectors.
- **Subprocess zombie prevention**: `child.kill()` in timeout handler now
  polls `try_wait()` for up to 100ms instead of blocking `wait()` to avoid
  hanging on zombie processes.
- **Cache lock poisoning**: `CachedRegistry` now invalidates cached state
  when the mutex is poisoned instead of continuing with potentially corrupt
  data.
- **Shard memory truncation**: `plan.rs` pipeline sharding now uses
  `div_ceil` instead of truncating division, preventing unallocated bytes.
- **Gaudi/oneAPI CSV caps**: both parsers now use `.take(20)` field limit
  matching CUDA, preventing DoS from malicious CLI output.
- **Intel oneAPI device ID validation**: uses `validate_device_id()` instead
  of `unwrap_or(0)`.
- **Neuron JSON array bounded**: capped at 256 devices to prevent DoS from
  crafted `neuron-ls` output. Device index truncation eliminated.
- **Schema version validation**: new `AcceleratorRegistry::from_json()` warns
  when deserializing registries with newer schema versions.

### Performance

- **Batched nvidia-smi**: CUDA detection and bandwidth probing merged into
  a single subprocess call, eliminating one nvidia-smi invocation per
  detection cycle (~5-10ms saved on NVIDIA systems).
- **Shared PCI address lists**: PCIe and NUMA enrichment now share a single
  `list_driver_pci_addrs()` computation instead of scanning sysfs twice.
- **Single-pass plan_sharding**: TPU and GPU device collection fused into
  one iteration over profiles instead of two separate filter passes.
- **Cached sort keys**: `--table` sort uses `sort_by_cached_key` for O(n)
  string allocations instead of O(n log n).
- **Stack buffer for sysfs reads**: `read_sysfs_string()` uses a 512-byte
  stack buffer for common small reads, avoiding heap allocation.
- **Pre-allocated collections**: profile collection uses `with_capacity(8)`,
  plan_sharding device vectors use `with_capacity(8/16)`.
- **`tracing-subscriber` made optional**: moved behind `cli` feature flag.
  Library users no longer pull 23 transitive crates.
- **`#[inline]` on hot-path queries**: `available()`, `total_memory()`,
  `has_accelerator()`, `by_family()`.
- **Vulkan output cap**: full `vulkaninfo` output parsing capped at 256 KiB.
- **Apple field cap**: `system_profiler` field values capped at 256 chars.

### CI

- **Cross-platform test matrix**: Ubuntu, macOS, and Windows runners for
  unit, integration, and doc tests.
- **Benchmark regression tracking**: `github-action-benchmark` on main with
  120% alert threshold.
- **Fuzz CI**: all 9 fuzz targets run for 30s each on every push/PR.
- **Minimal feature testing**: `--no-default-features` and single-backend
  builds verified in CI.
- **Cross-platform release builds**: Linux AMD64/ARM64, Windows AMD64, macOS
  ARM64 binaries built and published on tag.

## [0.19.3] - 2026-03-19

### Performance

- **Detection 3.5x faster**: eliminated per-subprocess reader threads in the
  command runner. Pipes are now read after the child exits (no deadlock risk
  since output is capped at 1 MiB). Poll interval reduced from 50ms to 10ms.
- **Single-pass `suggest_quantization`**: replaced 5 separate profile scans
  (`best_memory_for` per family) with one loop collecting all family maxima.
  Reduces O(5n) → O(n) on the profile list.
- **Sequential path for ≤1 backend**: `DetectBuilder::none().with_cuda()`
  skips `std::thread::scope` entirely, avoiding thread spawn/join overhead
  for selective single-backend detection.
- **CachedRegistry zero-copy**: `get()` now returns `Arc<AcceleratorRegistry>`
  instead of cloning the entire profile list on every call.
- **Reduced allocations**: `/proc/meminfo` parsing uses `nth()` iterator
  instead of collecting into a `Vec`. `read_limited` pre-allocates with
  `Vec::with_capacity`. `String::from_utf8_lossy().into_owned()` avoids
  double allocation. `#[inline]` on hot accessors (`all_profiles`,
  `warnings`, `estimate_memory`).

### Added

- **Async detection**: `AcceleratorRegistry::detect_async()` and
  `DetectBuilder::detect_async()` behind the `async-detect` cargo feature.
  Uses `tokio::task::spawn_blocking` to avoid blocking the async runtime.
- **CLI `--watch <secs>` mode**: re-detects on interval with screen clear and
  device-count change notifications.
- **CLI `--sort` flag**: sort `--table` output by `mem`, `name`, or `family`.
- **CLI `--family` flag**: filter `--table` output to a specific family
  (`gpu`, `tpu`, `npu`, `asic`, `cpu`).
- **C FFI** (`src/ffi.rs` + `include/ai_hwaccel.h`): `extern "C"` API with
  `ai_hwaccel_detect()`, `ai_hwaccel_device_count()`,
  `ai_hwaccel_has_accelerator()`, `ai_hwaccel_accelerator_memory()`,
  `ai_hwaccel_json()` and corresponding free functions.
- **Framework integration guide**: `docs/guides/framework-integration.md` with
  code examples for `candle`, `burn`, `tch-rs`, `ort`, multi-device sharding,
  and training memory budgeting.
- `tokio` optional dependency (behind `async-detect` feature).
- **Feature flags**: each of the 11 hardware backends is gated behind a cargo
  feature (`cuda`, `rocm`, `apple`, `vulkan`, `intel-npu`, `amd-xdna`, `tpu`,
  `gaudi`, `aws-neuron`, `intel-oneapi`, `qualcomm`). All enabled by default
  via `all-backends`. Disabled backends are not compiled.
- **CLI `--table` / `-t` flag**: human-readable tabular device listing with
  device ID, name, memory, family, and status columns.
- **CLI `--debug` / `-d` flag**: sets `RUST_LOG=debug` for verbose detection
  diagnostics without manually setting the environment variable.
- **Serde schema version**: `AcceleratorRegistry` now serializes a
  `schema_version` field (currently `1`) for forward-compatibility. Accessible
  via `registry.schema_version()` and `SCHEMA_VERSION` constant.
- **Property-based tests**: `proptest` fuzzing for `estimate_memory`,
  `plan_sharding`, `suggest_quantization`, and `estimate_training_memory`
  across random parameter counts and device configurations.
- **Architecture decision records**: `docs/decisions/` with 4 ADRs:
  sysfs-over-vendor-SDKs, calendar versioning, parallel detection, and
  feature flags per backend.
- **Crate-level guide**: expanded `lib.rs` documentation with a 4-step
  walkthrough (detect → query → plan → train) and cargo feature reference
  table.
- **JSON schema**: `docs/schema.json` documenting the serialized registry
  format (JSON Schema draft 2020-12).
- **`CachedRegistry`**: thread-safe detection cache with configurable TTL.
  Avoids redundant CLI tool invocations on repeated `detect()` calls.
- **Mock detection tests**: `tests/mock_detection.rs` with 11 tests using
  `tempfile` to build fake sysfs trees for hardware-independent backend
  testing, plus serde `deny_unknown_fields` rejection tests and schema
  version validation.
- **Windows CI**: added `x86_64-pc-windows-msvc` to the CI test matrix.
- `proptest` and `tempfile` dev-dependencies.
- Test suite expanded to 173 tests (140 unit + 9 integration + 11 mock +
  13 doc-tests).
- **Modular architecture**: refactored 3 monolithic source files into 23
  focused modules with single responsibilities.
  - `types.rs` (714 lines) split into `hardware/` (with `tpu.rs`, `gaudi.rs`,
    `neuron.rs`), `profile.rs`, `quantization.rs`, `requirement.rs`,
    `sharding.rs`, and `training.rs`.
  - `detect.rs` (693 lines) split into `registry.rs` (struct + query methods)
    and `detect/` module with one file per hardware backend.
  - `plan.rs` split into `plan.rs` (sharding logic) and `training.rs`
    (training types and memory estimation).
  - `tests.rs` (849 lines) split into `tests/` module with 10 files by concern.
- **`DetectionError` type** (`src/error.rs`): non-fatal detection errors
  captured as structured warnings (`ToolNotFound`, `ToolFailed`, `ParseError`,
  `SysfsReadError`) and accessible via `AcceleratorRegistry::warnings()`.
- **`DetectBuilder`**: selective backend detection via builder pattern —
  `AcceleratorRegistry::builder().with_cuda().without_vulkan().detect()`.
  Includes `Backend` enum with `ALL` constant.
- **`#[non_exhaustive]`** on `AcceleratorType`, `AcceleratorFamily`,
  `QuantizationLevel`, `AcceleratorRequirement`, and `DetectionError` for
  semver-safe enum extension.
- **Convenience constructors** on `AcceleratorProfile`: `cuda()`, `rocm()`,
  `tpu()`, `gaudi()`, `cpu()` for test and manual-config ergonomics.
- **`Display` for `ShardingPlan`**: human-readable multi-line plan summary
  showing strategy, memory, throughput, and per-shard device assignments.
- **CLI `--pretty` / `-p` flag**: pretty-printed JSON output.
- **CLI warnings**: detection warnings appear in `--summary` JSON output and
  are logged at `warn` level.
- **Structured logging**: CLI binary uses `tracing-subscriber` with `RUST_LOG`
  environment variable support and `--json-log` flag for structured JSON
  output to stderr.
- **Parallel detection**: all backends run concurrently via
  `std::thread::scope`, reducing wall-clock latency on multi-tool systems.
  Vulkan deduplication moved to a post-pass.
- **Safe command runner** (`detect/command.rs`): all CLI-based detectors use
  `run_tool()` which enforces:
  - Absolute path resolution via `which()` to prevent `$PATH` hijacking.
  - 5-second timeout with `child.kill()` on expiry.
  - Output size limits: stdout capped at 1 MiB, stderr at 4 KiB.
- **Input validation**: `validate_device_id()` (0--1024) and
  `validate_memory_mb()` (0--16 TiB) reject out-of-range parsed values from
  CLI tool output.
- **`#[serde(deny_unknown_fields)]`** on `AcceleratorRegistry`,
  `AcceleratorProfile`, `ModelShard`, `ShardingPlan` to reject unexpected
  JSON fields during deserialization.
- **`deny.toml`**: `cargo-deny` configuration for license allowlist, advisory
  checks, and crate source restrictions. New `make deny` target.
- **Threat model**: `docs/development/threat-model.md` documenting attack
  surface, trust assumptions, and mitigations.
- **Integration tests**: `tests/integration.rs` with 9 end-to-end tests
  covering the detect-query-plan pipeline, builder, JSON roundtrip, manual
  registry, training estimation, Display impls, and warnings.
- **Benchmark suite**: `criterion` benchmarks in `benches/` for `detect()`,
  `plan_sharding()`, `suggest_quantization()`, `estimate_memory()`, and
  `estimate_training_memory()`.
- **`examples/` directory**: four runnable examples — `detect.rs`, `plan.rs`,
  `training.rs`, `json_output.rs`.
- **Rustdoc examples**: `# Examples` sections on `AcceleratorRegistry`,
  `AcceleratorProfile`, `QuantizationLevel`, `DetectionError`, and
  `estimate_training_memory()`. All compile as doc-tests.
- **CI improvements**: cross-platform matrix (Linux + macOS), MSRV job
  (Rust 1.89), coverage via `cargo-llvm-cov` + Codecov, `cargo-deny`
  supply-chain checks.
- `tracing-subscriber` dependency (with `env-filter` and `json` features).
- `criterion` dev-dependency for benchmarks.
- `docs/development/roadmap.md` documenting the path to v1.0.
- Test suite expanded from 46 to 149 tests (133 unit + 9 integration +
  7 doc-tests).

### Changed

- **Switched from CalVer to SemVer**: version is now `0.19.3` (pre-1.0). The
  `0.x` series may contain breaking changes between minor versions.
- **NVIDIA detection**: now parses `driver_version` from `nvidia-smi` and
  reports structured `DetectionError` on tool failure or parse errors.
- **Vulkan detection**: parses `vulkaninfo --summary` for real device names,
  memory heap sizes, API version, and driver version instead of registering a
  generic placeholder device.
- **Apple detection**: macOS support via `system_profiler SPHardwareDataType`
  for chip name and unified memory size. ANE memory estimate varies by chip
  generation (M1: 4 GB, M2: 6 GB, M3/M4: 8 GB). Linux Asahi detection
  preserved as fallback.
- **CPU memory detection**: macOS fallback via `sysctl hw.memsize` when
  `/proc/meminfo` is unavailable.
- All detection backends now report structured warnings and use the safe
  command runner for CLI tool invocations.

### Fixed

- **`suggest_quantization` semantic bugs**: no longer returns BF16 for
  Qualcomm/Neuron AI ASICs (only Gaudi). Falls back through FP16→INT8→INT4
  on CPU instead of unconditionally returning FP16 for models that don't fit.
- **CPU memory detection**: macOS `sysctl` fallback now uses the safe command
  runner (`run_tool`) with absolute path resolution and timeout, matching all
  other CLI tool invocations.
- **Integer overflow safety**: all memory multiplications (TPU HBM × chip
  count, Neuron cores × memory, KB→bytes) use `saturating_mul` to prevent
  panics on extreme values.
- **Pipeline parallel layer assignment**: last shard now captures all remaining
  layers instead of potentially leaving a gap when layer count doesn't divide
  evenly across devices.
- **Per-chip memory precision**: TPU tensor-parallel uses ceiling division
  (`div_ceil`) so no bytes are lost to rounding.
- **Cache mutex poisoning**: `CachedRegistry` recovers from poisoned locks
  instead of panicking.
- **UTF-8 safe truncation**: CLI table column truncation uses `chars().take()`
  instead of byte slicing.
- **Windows compatibility**: mock detection tests gate Unix symlinks behind
  `#[cfg(unix)]` so the test file compiles on Windows.
- **Gaudi detection**: malformed CSV lines now produce `ParseError` warnings
  instead of being silently skipped. Device IDs and memory values are
  validated with `validate_device_id` / `validate_memory_mb`.
- `Cargo.toml` license field corrected from `AGPL-3.0` to the SPDX-correct
  `AGPL-3.0-only`.
- Added missing `homepage` and `readme` fields to `Cargo.toml` for crates.io
  compliance.

## [2026.3.19] - 2026-03-19 (CalVer, pre-SemVer switch)

### Added

- Initial public release.
- Hardware detection for 13 accelerator families: NVIDIA CUDA, AMD ROCm,
  Apple Metal, Apple ANE, Intel NPU, AMD XDNA, Google TPU (v4/v5e/v5p),
  Intel Gaudi (2/3), AWS Inferentia, AWS Trainium, Qualcomm Cloud AI 100,
  Vulkan Compute, and CPU fallback.
- `AcceleratorRegistry` with `detect()`, querying, and planning APIs.
- Quantization-aware memory estimation (`FP32`, `FP16`, `BF16`, `INT8`, `INT4`).
- Model sharding planner with tensor-parallel, pipeline-parallel, and
  data-parallel strategies.
- Training memory estimator for full fine-tune, LoRA, QLoRA, DPO, RLHF, and
  distillation methods.
- Serde support for all public types.
- CLI binary with `--summary` and `--version` flags.
- CI pipeline (format, clippy, tests, cargo-audit).
- Release automation with version consistency checks.
- Project documentation: `README.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `CHANGELOG.md`.
- `LICENSE` (AGPL-3.0-only).
- `Makefile` for local development (`check`, `fmt`, `clippy`, `test`, `build`,
  `doc`, `clean`).
- `scripts/version-bump.sh` for calendar versioning.
