---
name: Cyrius blockers for ai-hwaccel live detection
description: ai-hwaccel port complete but live detection blocked by Cyrius bugs #9 (getenv) and #10 (exec_capture) — tracked on Cyrius roadmap
type: project
---

ai-hwaccel Cyrius port is code-complete (4,382 lines, 445 tests passing) but live hardware detection is blocked by two Cyrius P2 bugs:

- **Bug #9**: `getenv()` returns wrong values (workaround: `cmd_getenv` in detect/command.cyr)
- **Bug #10**: `exec_capture()` hangs in compiled binaries (blocks all subprocess-based detectors: nvidia-smi, hl-smi, neuron-ls, vulkaninfo, etc.)

**Why:** These bugs are in the Cyrius compiler/stdlib, not ai-hwaccel. Fix lands in Cyrius, then ai-hwaccel CLI works without code changes.

**How to apply:** Once Cyrius fixes these bugs, re-test ai-hwaccel with `cat src/main.cyr | cc2 > ai-hwaccel && ./ai-hwaccel --summary`. Remove `cmd_getenv` workaround if Bug #9 is fixed. The cost module (cost.cyr + model.cyr) also needs fixup table expansion (v1.11.0 #4) to be included in the main binary.
