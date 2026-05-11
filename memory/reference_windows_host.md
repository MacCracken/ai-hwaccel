---
name: Windows test host
description: cass is the project's Windows machine for Win64 PE binary validation, reachable via `ssh cass`
type: reference
---

`ssh cass` connects to the project's Windows host. Use it to verify
cc5_win-produced PE binaries (Win64 target) actually run end-to-end, and
to capture real DXGI / WMI / `dxdiag` output for `tests/fixtures/windows/`
when the Windows DXGI backend slot lands.

This is the same `cass` referenced throughout the cyrius CHANGELOG's
v5.10.x cross-host smoke testing (e.g., entry 5.10.49 debunked a PE
exit-code propagation false-negative using `cmd /v /c "prog.exe & echo
exit=!errorlevel!"` on cass).

### Errorlevel wrapper gotcha (current 5.10.x)

Until cyrius 5.11.6 lands (see below), exit-code tests on cass must
use one of these shapes:

- ✗ `cmd /c "prog.exe & echo exit=%errorlevel%"` — expands at parse
  time, falsely reports `exit=0` regardless of what prog.exe returned.
- ✓ `cmd /v /c "prog.exe & echo exit=!errorlevel!"` — delayed
  expansion, reads errorlevel at exec time.
- ✓ `.bat` indirection — write two lines to a `.bat`; newlines split
  parse passes so the second line sees the updated errorlevel.

### Upstream fix: cyrius 5.11.6 (next in line)

Per Robert (2026-05-11): **the wrapper gotcha is addressed in cyrius
5.11.6, the next release after 5.10.x.** When ai-hwaccel bumps its
cyrius pin to 5.11.6+, the cross-host smoke can collapse to a plain
`ssh cass 'prog.exe; echo exit=%errorlevel%'` shape. Re-evaluate this
memory file at that pin bump.

For ai-hwaccel:

- **Cross-host smoke** — after `cc5_win_cross src/main.cyr build/ai-hwaccel.exe`,
  `scp build/ai-hwaccel.exe cass:` then
  `ssh cass 'cmd /v /c "ai-hwaccel.exe --version & echo exit=!errorlevel!"'`.
- **DXGI fixture capture** — once `src/detect/windows.cyr` lands,
  `ssh cass 'dxdiag /t %TEMP%\dxdiag.txt && type %TEMP%\dxdiag.txt'`
  captures the human-readable adapter list; the COM/DXGI binding's
  output should be cross-checked against this.

No further auth needed — the `cass` host alias is configured in
`~/.ssh/config` on the dev machine.
