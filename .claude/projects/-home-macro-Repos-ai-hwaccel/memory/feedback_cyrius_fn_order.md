---
name: Cyrius function ordering
description: Functions must be defined before any global statements — alloc_init() etc. must come after all fn defs
type: feedback
---

In Cyrius, all `fn` definitions must appear before any global-scope statements (function calls, variable assignments).

**Why:** cc2 uses a two-pass parser. Once it sees a non-fn statement, it switches to code emission mode and can't handle further fn definitions. Produces "unexpected fn" error.

**How to apply:** In every .cyr file, structure as: includes → enums → global vars → fn definitions → executable statements. In test files, put `alloc_init()` and test calls at the very bottom after all fn defs.
