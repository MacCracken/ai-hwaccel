# `profile_from_json_str` calls `json_v_parse_str`, which bayan 1.3.0 removed — RESOLVED

**Status:** ✅ **RESOLVED in 2.3.16** (2026-07-30) — renamed to `json_v_parse_buf`, and the pin
moved to cyrius 6.5.2 so ai-hwaccel's own build exercises the new name. See CHANGELOG [2.3.16].

**Correction to the severity assessment below:** this was filed as "silent in a consumer that never
calls the function", which is true of hoosh but understated ai-hwaccel's own case. Building
`tests/tcyr/json_roundtrip_test.tcyr` on 6.5.2 fails outright — *refusing to emit binary with 1
reachable undefined function(s)* — because that suite calls `profile_from_json_str` three times. In
this repo it was a build break waiting for a toolchain bump, which is the good failure mode.

**Original report:** filed 2026-07-30 against ai-hwaccel 2.3.15. Verified by building hoosh
against cyrius 6.5.2 with the 2.3.15 distlib vendored: the link emits
`warning: undefined function 'json_v_parse_str'`, and the only reference is
`src/json_out.cyr:180` (`dist/ai-hwaccel.cyr:6149` in the bundle).
**Placement:** unpinned — next patch. One-line rename plus a toolchain-pin bump.
**Discovered:** 2026-07-30 while updating hoosh to cyrius 6.5.2 and ai-hwaccel 2.3.15.
**Severity:** Medium — silent in a consumer that never calls the function (cycc lowers an undefined
call to `ud2`, so it is a SIGILL rather than a link failure), and a hard crash in one that does.
**Affects:** ai-hwaccel 2.3.15 and earlier, when consumed by a project on cyrius ≥ 6.5.0.
ai-hwaccel's own pin is `6.4.69`, where the symbol still exists, so its own build is green.

## Summary

bayan 1.3.0 (shipped in cyrius 6.5.0) **renamed** its cstr+len JSON parse entry points:

```
bayan_json_v_parse_str  →  bayan_json_v_parse_buf
json_v_parse_str        →  json_v_parse_buf
```

The rename is deliberate and not reversible: `X_str` is a **reserved overload slot** in Cyrius —
a call `X(a, …)` routes to `X_str` whenever `a` is Str-typed at the call site — so a `(ptr, len)`
form may never occupy that name. While it did, every bare `bayan_json_v_parse(someStr)` in the
ecosystem was silently rewritten into a 1-arg call to a 2-arg function and returned 0 for valid
JSON. See bayan's CHANGELOG [1.3.0].

`src/json_out.cyr:180` still calls the old name:

```cyrius
# buf+len parse entry (json_v_parse_str) directly on the cstr.
fn profile_from_json_str(js_cstr) {
    var v = json_v_parse_str(js_cstr, strlen(js_cstr));
```

The bodies are byte-identical, so this is a pure rename with no semantic change.

## Reproduction

Any project on cyrius ≥ 6.5.0 that vendors `dist/ai-hwaccel.cyr`:

```sh
cd <consumer>            # e.g. hoosh, with cyrius = "6.5.2" and ai-hwaccel 2.3.15
cyrius deps
cyrius build src/main.cyr build/out
# warning: undefined function 'json_v_parse_str'
```

The build **succeeds** — it is a warning, not an error, because nothing reachable calls
`profile_from_json_str`. That is exactly what makes it worth filing: the failure mode is a runtime
SIGILL in the first consumer that does call it, not a build break at the point of the mistake.

## Root cause

`src/json_out.cyr:180`. No other reference — `grep -rn 'json_v_parse_str' src/` returns the call and
its own doc comment.

## Proposed fix

Rename the call to `json_v_parse_buf`, update the comment on line 178, and bump
`cyrius.cyml`'s pin past 6.5.0 so ai-hwaccel's own build exercises the new name. Worth a quick
`grep -rnE '_parse_str|_str\(' src/` for any other reserved-slot collisions while in there.

If ai-hwaccel needs to keep building on pre-6.5.0 toolchains, the rename is still the right move —
the old name simply does not exist on new ones, and there is no version of the code that works on
both without an `#ifdef`.

## Consumer-side workaround

None applied in hoosh, and none needed there: `profile_from_json_str` has no callers in that binary,
so the symbol is dead and the warning is cosmetic. hoosh 2.5.12 ships with it. A consumer that does
call the function has no workaround short of vendoring a patched bundle.
