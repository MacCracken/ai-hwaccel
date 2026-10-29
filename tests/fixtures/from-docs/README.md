# Vendor-tool output written from documentation

Hand-written output in the shape each vendor documents, used by
`tests/tcyr/backend_test.tcyr`, `tests/tcyr/gpu_parser_test.tcyr` and
`tests/tcyr/accel_test.tcyr`. None of it is a capture from hardware: no host
the project tests on has these devices (the roadmap lists the real captures
still wanted). Tests read these files relative to the repository root, which
is where CI runs them.

| file | tool | shape taken from |
|---|---|---|
| `neuron-ls-trn1.json` | `neuron-ls --json-output` on a trn1 (2 of its 16 devices) | AWS Neuron's `neuron-ls` docs: one object per device with `neuron_device`, `bdf`, `connected_to`, `nc_count`, `memory_size` (bytes) and `neuron_processes`; no key names the chip |
| `neuron-ls-trn2.json` | the same on a trn2 (2 devices), on one line | the same docs' trn2 example: 8 NeuronCores and 96 GiB a device |
| `hl-smi-query-aip.csv` | `hl-smi --query-aip=index,name,memory.total,memory.free --format=csv,noheader,nounits` | Intel Gaudi docs; board names HL-225 (Gaudi2), HL-325L and HL-338 (Gaudi3 OAM and PCIe), memory in MiB |
| `nvidia-smi-nvlink-s-h100.txt` | `nvidia-smi nvlink -s`, two H100 SXM with 18 links each | NVIDIA's `nvidia-smi` docs: a `GPU N: <name> (UUID: …)` line, then `Link N: <rate> GB/s` per link |
| `nvidia-smi-nvlink-s-3090-unbridged.txt` | the same, two RTX 3090 without a bridge | links that are not up print `<inactive>` |

The trn1 file's process `command` holds escaped quotes and a `}` on purpose:
it is free text, and a key inside it must not be read as the device's.
