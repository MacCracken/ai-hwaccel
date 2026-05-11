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

### Exit-code propagation

ai-hwaccel pinned cyrius **5.11.8** in 2.2.1, which includes the 5.11.6
wrapper fix. Exit-code tests on cass can use plain shell shape:

```sh
ssh cass 'ai-hwaccel.exe --version; echo exit=%errorlevel%'
```

Historical note — under the 5.10.x pin (pre-2.2.1), the wrapper had
to be one of:

- ✓ `cmd /v /c "prog.exe & echo exit=!errorlevel!"` — delayed
  expansion, reads errorlevel at exec time.
- ✓ `.bat` indirection — write two lines to a `.bat`; newlines split
  parse passes so the second line sees the updated errorlevel.
- ✗ `cmd /c "prog.exe & echo exit=%errorlevel%"` — expanded at parse
  time, false-reported `exit=0`.

Cyrius 5.11.6 fixed the underlying PE exit-code propagation so the
plain shape works again. Keep this section as a reference for any
future 5.10.x-pinned consumer that hits the same gotcha.

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
