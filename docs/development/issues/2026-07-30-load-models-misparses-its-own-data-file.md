# `load_models` returns 1 model instead of 26 — it mis-parses ai-hwaccel's own `data/models.json`

**Status:** 🟡 **OPEN** — filed 2026-07-30 against ai-hwaccel 2.3.15/2.3.16. Verified by reading
`src/model.cyr:37-71` against `data/models.json` as shipped. Not fixed in 2.3.16, because the fix
has two incompatible shapes and the choice is a compatibility decision (see *Proposed fix*).
**Placement:** unpinned — next patch, once the shape question is settled.
**Discovered:** 2026-07-30 while updating hoosh to ai-hwaccel 2.3.15. hoosh already carries a
workaround and a guard test for it.
**Severity:** Medium — silent wrong answer, not a crash. A consumer gets one model where it expects
the full catalog, with no error.
**Affects:** every version whose `data/models.json` uses the `{"models":[…]}` wrapper.

## Summary

`load_models` (`src/model.cyr:37`) does not parse JSON. It scans the raw bytes for a `{`, brace-matches
to find the object's end, extracts fields from that slice, and resumes from the end:

```cyrius
while (pos < n) {
    if (load8(buf + pos) != 123) { pos = pos + 1; continue; }   # find '{'
    ...
    var obj_end = pos + 1;
    var depth = 1;
    while (obj_end < n && depth > 0) {                          # brace-match
        var c = load8(buf + obj_end);
        if (c == 123) { depth = depth + 1; }
        if (c == 125) { depth = depth - 1; }
        obj_end = obj_end + 1;
    }
    var obj = str_new(buf + pos, obj_end - pos);
    _parse_model_field_str(obj, "name", m, 0);
    ...
    pos = obj_end;
}
```

That works on a **top-level array** of model objects. `data/models.json` is not one — it ships as:

```json
{
  "models": [
    { "name": "Llama 3.1 8B", … },
    …25 more…
  ]
}
```

So the first `{` found is the **wrapper**, brace-matching runs to the last byte of the document, and
`obj` becomes the entire file. `_parse_model_field_str(obj, "name", …)` then finds the first `"name":`
in that blob — model one's — and one profile is pushed. `pos = obj_end` is now the end of the buffer,
so the loop exits. **26 models in the file, 1 in the returned vec, no error.**

## Reproduction

```sh
python3 -c "import json;d=json.load(open('data/models.json'));print(type(d).__name__, len(d['models']))"
# dict 26
```

Then call `load_models()` and count the result: 1.

There is no test — `grep -rn 'load_models' tests/ src/ benches/` returns only the definition. The
function has **zero callers inside ai-hwaccel**, which is why its own suite is green: it is shipped
API exercised only downstream.

## Downstream evidence

hoosh consumes this and hit it. Its `CLAUDE.md` carries the workaround as a standing instruction:

> `models.json` MUST be a **top-level JSON array** — ai-hwaccel's `load_models` only parses the first
> object from a `{"models":[…]}` wrapper (guarded by the `hardware_data_files` test).

hoosh vendors an **unwrapped** copy and has a test that fails the build if anyone re-syncs the
wrapped form verbatim. That test fired on 2026-07-30 during the 2.3.15 re-sync, which is how this
was found.

## Proposed fix

Two options, and the choice is a compatibility decision rather than a technical one:

**(a) Teach the loader the wrapper.** Skip a leading `{"models"` and start scanning at the `[`. Keeps
the shipped data shape, so nothing that reads `data/models.json` as a document breaks. Roughly six
lines. Consumers already unwrapping (hoosh) keep working, since a top-level array still parses.

**(b) Ship `data/models.json` as a top-level array.** Matches what the loader already expects and what
hoosh already vendors. But it is a **breaking change** for anything reading the file as a JSON object
— including anything keying on `"models"` — and the file is a published artifact.

**Recommend (a)**, because it fixes the defect without changing a shipped artifact, and because a
loader that only accepts one of two reasonable shapes is the more fragile half of the pair.

Whichever is chosen, **add a test**: load the real `data/models.json` and assert the count matches the
file's array length. The absence of that test is what let the two drift.

### Three adjacent weaknesses in the same function, worth fixing alongside

1. **`alloc(32768)` then `store8(buf + n, 0)`** — at exactly 32768 bytes read, this writes one byte
   past the allocation. Allocate `32769`, or read at most `32767`.
2. **Silent truncation.** `file_read_all` stops at `maxlen` and reports the count with no error, so a
   `models.json` over 32 KB quietly loses models. The file is 5.4 KB today, so this is latent, not
   live — but it fails the same way the wrapper bug does: fewer models, no signal.
3. **ai-hwaccel already depends on a real JSON parser.** `src/json_out.cyr:180` calls
   `json_v_parse_buf`. Using it here would delete the hand-rolled brace matcher, the byte cap and the
   off-by-one together, and handle both document shapes for free. That is a larger change than (a)
   and probably wants its own release, but it is the version worth converging on.

## Consumer-side workaround

hoosh unwraps `data/models.json` to a top-level array when re-syncing, and guards it with the
`hardware_data_files` test. Any other consumer calling `load_models` against the shipped file today is
silently getting one model.
