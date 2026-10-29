# cyrius 6.7.18: on Windows, `getenv` answers 0 for a value of 4096 characters or more

**Filed:** 2026-10-29 (found in the ai-hwaccel 2.5.5 audit)
**Severity:** Medium — a Windows `PATH` of 4096+ characters (the CUDA
toolkit, Visual Studio, conda and oneAPI add entries) made ai-hwaccel's
`which()` find no tool at all: every vendor tool read as "not found".
**Owner: CYRIUS (upstream).** `lib/io.cyr` is vendored stdlib; ai-hwaccel
never edits `lib/`.
**Repos:** cyrius `6.7.18` · ai-hwaccel `2.5.5`
**Status:** OPEN upstream. ai-hwaccel 2.5.5 works around it in
`cmd_getenv` (src/detect/command.cyr): on PE it calls the 0xF015
`GetEnvironmentVariableA` reroute itself, with a buffer of the size the call
asks for.

## Summary

The PE branch of `getenv` (lib/io.cyr:1802-1811) reads into a fixed buffer:

```
var wbuf = alloc(4096);
var wn = syscall(0xF015, name, wbuf, 4096);
if (wn <= 0) { return 0; }
if (wn >= 4096) { return 0; }        # truncated: refuse rather than hand back a cut value
```

When the buffer is short, `GetEnvironmentVariableA` returns the size it
needs, NUL included, so the refusal is right but the answer is wrong: the
variable exists and `getenv` says it does not. Windows allows values up to
32767 characters. Each call also keeps 4 KiB of bump heap whatever the
value's length.

## Reproduction

Cross-built with `CYRIUS_DCE=1 cyrius build --win` and run under Wine with
`WINEPATH` padding `PATH` (the same result on a 4096-, 4201-, 5999- and
6112-character `PATH`): `getenv("PATH")` returns 0 and a `PATH` walk finds
nothing; at 4095 characters both work. With the 2.5.5 `cmd_getenv`, a
6098-character `PATH` comes back whole and `which("cmd")` finds
`C:\windows\system32\cmd.exe`. Not yet run on `cass`.

## Proposed fix

Grow and retry: when the first call returns `n >= size`, allocate `n` bytes
and call again (a second time if the variable grew in between). A small
first buffer on the stack, copied into an exact-size allocation, also drops
the 4 KiB kept per call.
