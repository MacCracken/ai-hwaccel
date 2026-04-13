---
name: Rust to Cyrius port status
description: ai-hwaccel has been fully ported from Rust to Cyrius; original Rust code preserved in rust-old/
type: project
---

ai-hwaccel was ported from Rust to Cyrius. The original Rust implementation is preserved in `rust-old/`.

The Cyrius port includes 14 source modules, 19 detection modules, and 9 test phases. Two modules — `cost.cyr` and `model.cyr` — are deferred and not included in `main.cyr` due to a Cyrius compiler fixup table overflow (4096 entry limit).

**Why:** Cyrius is the AGNOS ecosystem language. All first-party projects are moving to Cyrius for sovereignty — no crates.io, no external governance.

**How to apply:** When working on ai-hwaccel, all new code is Cyrius. Reference `rust-old/` for the original logic when porting or debugging. The fixup table overflow is a compiler-side blocker tracked in the Cyrius repo.
