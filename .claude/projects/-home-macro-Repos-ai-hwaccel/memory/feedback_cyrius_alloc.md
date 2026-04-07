---
name: Cyrius bump allocator
description: alloc.cyr is bump-only — call alloc_init() first, no individual free(), use alloc() not malloc()
type: feedback
---

The Cyrius stdlib allocator (lib/alloc.cyr) is a bump allocator.

**Why:** Simple, fast, no fragmentation. Individual `free()` does not exist. Use `alloc_reset()` for batch deallocation or arenas for scoped allocation.

**How to apply:** Always call `alloc_init()` before any `alloc()` calls. Never call `free()` — it's undefined and causes warnings. Use `alloc()` not `malloc()`. For long-running programs that need memory reclamation, use `arena_new()`/`arena_alloc()`/`arena_reset()`.
