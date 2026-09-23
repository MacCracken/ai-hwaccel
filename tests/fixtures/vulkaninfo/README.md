# vulkaninfo captures

Real `vulkaninfo` output, used by `tests/tcyr/gpu_parser_test.tcyr` to test
`src/detect/vulkan.cyr` against what the tool prints, not a hand-written
approximation. Tests read these files relative to the repository root, which
is where CI runs them.

| file | host | GPU | driver | vulkaninfo |
|---|---|---|---|---|
| `renoir-linux-summary.txt`, `renoir-linux-full.txt` | Linux dev host (Arch, AMD Ryzen 7 5800H) | AMD Radeon Graphics, an integrated APU GPU (vendorID 0x1002, deviceID 0x1638) | Mesa 26.2.3 RADV | 1.4.357 |
| `uhd600-windows-summary.txt`, `uhd600-windows-full.txt` | `cass` (Windows 11, 8 GB) | Intel UHD Graphics 600, an integrated GPU (vendorID 0x8086, deviceID 0x3185) | Intel's Windows driver | Vulkan Runtime's `vulkaninfo.exe` |

Captured on 2026-09-23 with `vulkaninfo --summary` and plain `vulkaninfo`,
stdout only. The Windows files are the tool's own bytes, CRLF line endings
included (captured through `cmd.exe` redirection, not PowerShell, which
re-encodes to UTF-16); `.gitattributes` keeps them that way.

What they pin down:

- Renoir: RADV exposes an APU's memory as a 2:1 split of its BIOS carve-out
  (8 GiB, sysfs `mem_info_vram_total`) plus GTT. Its device-local heap
  (23.61 GiB) is therefore not the carve-out.
- UHD 600: one device-local heap of 4 202 799 104 bytes, half of the host's
  8 405 598 208 bytes of RAM. All of it is system memory.
