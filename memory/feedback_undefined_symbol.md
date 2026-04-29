---
name: Undefined symbol crash — fixed in cc3 3.10.0
description: Calling non-existent functions compiled silently and crashed at runtime; fixed by undefined function diagnostic
type: feedback
---

Calling a non-existent function (typo or wrong name) compiled without error and crashed at runtime (SIGILL/SIGSEGV). Cost hours of debugging during the port — `assert_report()` vs `assert_summary()` was a 3-character typo.

**Why:** Forward-reference resolution never validated that the target function exists. The compiler emitted a jump to address 0.

**How to apply:** Always build with cc3 3.10.0+ which emits `error: undefined function 'name'` warnings. This diagnostic caught 3 additional bugs (enrich_disk, detect_interconnect, builder_enable) on first build after upgrading. Pin `cyrius = "3.10.0"` in cyrius.toml.
