---
name: Fixup table overflow — resolved
description: cost.cyr and model.cyr re-included in main.cyr after Cyrius v3.7.0 expanded fixup table to 16384
type: project
---

RESOLVED (2026-04-12): `cost.cyr` and `model.cyr` are now included in `main.cyr`. The Cyrius compiler fixup table was expanded from 4096 to 16384 entries in v3.7.0, removing the blocker.

The `--cost` CLI flag now uses the real `recommend_instances()` function from `cost.cyr` instead of a stub.

**Why:** Previously blocked by compiler limitation. Compiler upgrade resolved it.

**How to apply:** No action needed — all modules are included. Binary is 164KB with full feature set.
