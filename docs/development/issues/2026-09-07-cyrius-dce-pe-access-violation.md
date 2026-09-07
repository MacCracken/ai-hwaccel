# cyrius 6.6.0: `CYRIUS_DCE=1` produces a PE that dies at startup with `0xC0000005`

**Filed:** 2026-09-07
**Severity:** HIGH — it broke the `windows-smoke` CI gate and would have shipped a
Windows wheel whose bundled EXE crashes on launch.
**Filed upstream 2026-09-07** as `cyrius/docs/development/issues/ai-hwaccel-dce-pe-access-violation-2026-09-07.md`,
with a minimal repro at `cyrius/docs/development/issues/repros/2026-09-07-dce-pe-access-violation.cyr`.
Bisected there: **first bad cycc is 6.5.72**, last known good **6.5.71** — 6.5.72 being
the release that turned DCE from NOP-padding into real elimination.
**Owner: CYRIUS (upstream).** Not an ai-hwaccel defect. The toolchain is not this
repo's to patch; ai-hwaccel carries a one-line workaround (below) until a cyrius
release fixes it.
**Repos:** cyrius `6.6.0` · ai-hwaccel `2.3.21`

## Summary

`CYRIUS_DCE=1 cyrius build --win` emits a PE32+ that access-violates before
producing any output. The same source, same toolchain, **without** the flag runs
correctly.

## Reproduction

Measured on `cass` (Windows 11, 10.0.26200), binaries cross-built on Linux from
the published `cyrius-6.6.0-x86_64-linux` tarball:

```
cyrius build --win src/main.cyr win_nodce.exe                -> OK (497,152 bytes)
CYRIUS_DCE=1 cyrius build --win src/main.cyr win_dce.exe     -> OK (497,152 bytes)
                                     note: 244,277 bytes of dead code eliminated

# on Windows:
.\win_nodce.exe   -> exit 0,             499 bytes of registry JSON
.\win_dce.exe     -> exit -1073741819    (0xC0000005 STATUS_ACCESS_VIOLATION), no output
```

Under Git Bash — which is what `defaults.run.shell: bash` gives the
`windows-latest` runner — that surfaces as **exit code 139** (128 + SIGSEGV), and
the failing step is titled by its first line, `set -euo pipefail`.

**Note the smell:** DCE reports eliminating 244,277 bytes and the output file is
**byte-for-byte the same size** with and without the flag. Something is being
removed from the image that the loader or the entry path still needs, without the
file shrinking.

## Scope — PE only

| target | `CYRIUS_DCE=1` | result |
|---|---|---|
| x86_64 ELF (Linux)   | yes | exit 0 — and genuinely reclaims (419 360 → 214 504 B) |
| ELF-aarch64 (qemu)   | yes | exit 0, output identical to no-DCE, size unchanged |
| Mach-O arm64 (ecb)   | yes | exit 0, output identical to no-DCE, size unchanged |
| **x86_64 PE (cass)** | yes | **0xC0000005 at startup** |
| x86_64 PE (cass)     | no  | exit 0 |

Only the x86_64 ELF backend actually reclaims. aarch64 and Mach-O report
elimination and do not shrink either — they simply do not crash.

## ai-hwaccel's workaround

`bindings/python/scripts/stage_win_cross.sh` no longer passes `CYRIUS_DCE=1` on
the `--win` build. It cost nothing: the PE was 497,152 bytes either way.

`CYRIUS_DCE=1` is retained everywhere else (`ci.yml`, `release.yml`,
`stage_binary.sh`), where it is correct and, on x86_64 ELF, worth −48.8%.

## Bisect result (done, 2026-09-07)

Same source, same host, same manifest, four toolchains:

| cycc | reports "eliminated" | `--win` + DCE | `--win` no DCE |
|---|---|---|---|
| 6.5.36 | no  | exit 0 | exit 0 |
| 6.5.71 | no  | exit 0 | exit 0 |
| **6.5.72** | **yes** | **0xC0000005** | exit 0 |
| 6.6.0  | yes | **0xC0000005** | exit 0 |

First bad is **6.5.72** — the release that made DCE actually eliminate. The
minimal repro is a four-line `main` built inside a project whose manifest
declares `[deps] stdlib`; without a manifest nothing links and it does not
reproduce.
