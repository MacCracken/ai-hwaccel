---
name: cc5_win 5.11.5 exit-code propagation broken
description: PE binaries built with cc5_win 5.11.5 (shipped in cyrius 5.11.6+) crash on cass with exit 0x40001000 instead of propagating ExitProcess(N); blocks cross-host smoke validation
type: feedback
---

cc5_win 5.11.5 — the cross-compiler that cyrius 5.11.6 added to
`~/.cyrius/bin/cc5_win` — produces PE binaries that crash on Win11
cass (build 26200) with exit code **0x40001000** (1073745920)
instead of propagating the source's `ExitProcess(N)` call.

**Why:** unclear. Probably IAT / entry-point / CRT-init bug in the
minimal PE that cc5_win emits. Both `cc5_win` (from the install
tree) and `cc5_win_cross` (from `/home/macro/Repos/cyrius/build/`,
v5.10.37) produce **byte-identical** PE output for the same source,
and both crash the same way. So the regression is in the PE emit
path, not the cross-compiler binary itself.

**How to apply:** when validating Windows-side work on `cass`,
treat exit-code propagation as **not currently trustworthy**. The
PE *runs* on Windows — Windows accepts and loads it, the process
starts — but the exit code seen by `cmd /v /c "exe & echo
exit=!errorlevel!"` is the crash status, not what the source
wrote. Two implications:

1. **Cross-host smoke** in CI / dev should validate by **stdout
   content match** (`echo "hello"`) not exit-code match. Stdout
   propagation works (verified earlier in the cyrius 5.10.x cycle).
2. **Exit-code-based gates** (the cyrius `_pe_exit_gate` in
   `programs/check.cyr`) will report broken until cc5_win is
   patched. Cyrius's own regression test for this lives at
   `tests/regression-pe-exit.sh` — when that goes green again,
   re-verify here.

**Wrapper rules unchanged from the 5.10.x note**: `%ERRORLEVEL%` in
`cmd /c` expands at parse time; use `cmd /v /c "exe & echo
exit=!errorlevel!"` or `.bat` indirection to see the real exit
code. 5.11.6's "fix" of the *wrapper* gotcha is real; the *PE
emit* bug surfaces underneath it.

**Status:** Filed against cyrius as
`/home/macro/Repos/cyrius/docs/development/issues/2026-05-11-ai-hwaccel-cc5-win-pe-exit-propagation.md`
(cyrius's in-tree issue tracking convention; pending the user's
review + commit on the cyrius side). ai-hwaccel pin stays at 5.11.8
so `cc5_win` is available for cross-build, but the Windows DXGI
work tests via Linux-hosted **fixture parsing** only until exit-code
propagation comes back. Real cass validation gates on a cc5_win
patch.
