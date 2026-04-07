---
name: Cyrius enum syntax
description: Cyrius enums use bare variant names, not dot-prefixed — ERR_NONE not DetectionError.ERR_NONE
type: feedback
---

Cyrius enums use bare variant names at the call site, not `EnumName.VARIANT` dot syntax.

**Why:** The cc2 compiler (v1.9.4) parses dot access as struct field access, not enum namespacing. Using `Foo.BAR` in expressions causes "unexpected ','" or "unexpected ')'" errors.

**How to apply:** When writing Cyrius code, always reference enum variants by their bare name (e.g., `ERR_NONE`, `ACCEL_CUDA`, `QUANT_FP16`). Prefix variant names to avoid collisions across enums (e.g., `ERR_` for DetectionError, `ACCEL_` for AcceleratorType).
