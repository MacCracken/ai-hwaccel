# Contributing to ai-hwaccel

Thank you for your interest in contributing! This document explains how to get
started, what we expect from contributions, and how the review process works.

## Getting started

1. **Fork and clone** the repository.
2. Install the Cyrius toolchain at the version `cyrius.cyml` pins
   (`cyrius = "6.6.6"`).
3. Vendor the standard library and dependencies into `lib/` (gitignored). Run
   both commands, in this order; `lib sync` alone leaves `lib/` incomplete:

   ```sh
   cyrius lib sync    # the stdlib files cyrius.cyml lists, from the pinned snapshot
   cyrius deps        # bayan's JSON sublib and the remaining stdlib files
   ```

4. Run the checks below. They should all pass on a fresh clone.

## Development workflow

Run everything from the repository root (some tests read `tests/fixtures/`):

```sh
CYRIUS_DCE=1 cyrius build src/main.cyr build/ai-hwaccel   # build the CLI
cyrius tests tests/tcyr                                   # 15 test units
cyrius fuzz                                               # 6 fuzz harnesses
cyrius vet src/main.cyr                                   # include-graph audit
for f in src/*.cyr src/detect/*.cyr; do cyrius lint --strict "$f"; done   # fails on a warning
for f in src/*.cyr src/detect/*.cyr tests/tcyr/*.tcyr fuzz/*.fcyr benches/*.bcyr; do
    cyrius fmt "$f" --check                               # plain `cyrius fmt <file>` rewrites the file
done
cyrius distlib                                            # after changing any module under src/
./scripts/bench-history.sh                                # benchmarks
```

`dist/ai-hwaccel.cyr` is the library bundle consumers pull; it is generated
from the `[lib]` modules in `cyrius.cyml`. Commit the regenerated file with
your change.

CI (`.github/workflows/ci.yml`) runs these checks plus the raw-offset guard,
the `dist/` drift check and the `cyrius.lock` hash check. CI runs lint without
`--strict`, so a lint warning does not fail it yet: check lint yourself.

## What to contribute

Contributions are welcome in several areas:

- **Captures from real hardware** -- the output of a vendor tool (`nvidia-smi`,
  `neuron-ls --json`, `hl-smi`, ...) or a sysfs tree from a machine we have no
  access to, added under `tests/fixtures/`. The roadmap's 2.5.x section lists
  the ones we need most.
- **New accelerator backends** -- see *Adding a new accelerator* below.
- **Improved detection accuracy** -- better sysfs paths, parser robustness,
  version identification.
- **Training/inference planning** -- more accurate memory models, new sharding
  strategies in `src/plan.cyr` and `src/training.cyr`.
- **Documentation** -- doc improvements, examples, guides.
- **Bug fixes** -- always welcome.

## Code style

- `cyrius fmt --check` and `cyrius lint --strict` clean.
- Comments use `#`. Put a short comment above each function saying what it
  returns and anything surprising about it.
- Integers only: fixed-point values scaled by 1000 (`_x1000`), no floats.
- New heap structs use `#derive(accessors)`: the constructor stays
  hand-written and calls the generated setters. CI rejects raw
  `load64(p + N)` / `store64(p + N, ...)` on a derived struct outside the file
  that defines it.
- Prefer enum constants to global `var`s (the compiler limits the number of
  globals), and `str_builder` to repeated concatenation.
- Keep parsers pure (text in, profiles out) and separate from file and process
  I/O, so tests and fuzz harnesses can feed them captured output.
- No third-party dependencies and no vendor SDKs. Never edit the vendored
  stdlib in `lib/`: it comes from the pinned toolchain.

## Project layout

[docs/architecture/overview.md](docs/architecture/overview.md) has the full
module map. In brief:

```
src/main.cyr            CLI: arguments and output modes
src/types.cyr           accelerator types and families, backends, ranks, throughput
src/profile.cyr         AcceleratorProfile
src/registry.cyr        registry queries, detection entry points, post-passes, duplicate pass
src/detect/             one detector per backend (or group), plus the post-pass enrichers
src/plan.cyr, training.cyr, cost.cyr, model.cyr, model_format.cyr, requirement.cyr
                        sharding, training memory, cloud cost, model catalog and headers
src/async_detect.cyr, cache.cyr, lazy.cyr
                        threaded, cached and lazy detection
src/json_out.cyr        JSON output, and profile_from_json
tests/tcyr/             test units;  tests/fixtures/  captured tool output
fuzz/, benches/         fuzz harnesses and benchmark suites
data/                   models.json, cloud_pricing.json
dist/                   the generated library bundle
bindings/python/        the Python package
```

## Adding a new accelerator

1. **Types** (`src/types.cyr`):
   - Add `ACCEL_<NAME>` to `AcceleratorType` (the next free value) and its
     display name to `accel_name`.
   - Classify it in its family predicate (`accel_is_gpu`, `accel_is_npu`,
     `accel_is_ai_asic`) and give it an `accel_throughput_x1000`,
     `accel_training_x1000` (0 means inference only) and `accel_rank`.
   - If it has no memory of its own, add it to `accel_uses_system_memory` and
     `ACCEL_SYSMEM_MASK` (`foundation_test` checks that the two agree).
   - Add `BACKEND_<NAME>` to `AiHwBackend`, raise `AIHW_BACKEND_COUNT`, name it
     in `backend_name`, and list it in `backend_uses_exec` if its detector runs
     a tool.
2. **Detector**: write `detect_<name>(profiles, warnings)`, in a new
   `src/detect/<name>.cyr` or in an existing file of its kind (`edge.cyr`,
   `cloud_asic.cyr`). Build profiles with `profile_new(ACCEL_<NAME>,
   device_id, memory_bytes)`, run tools only through `run_tool*` (it resolves
   the tool on `$PATH`, runs it without a shell and reports a missing tool as
   a warning), and push a warning for output it cannot parse.
3. **Build**: include the new file in `src/main.cyr`, and list it at the same
   position in `cyrius.cyml`'s `[lib] modules`. The compiler is single-pass, so
   it must come after the modules it uses. Then run `cyrius distlib`.
4. **Wire it into every entry point**: `registry_detect_with_opts`
   (`src/registry.cyr`), `registry_detect_threaded_with`
   (`src/async_detect.cyr`: its thread list if the detector runs a tool,
   `_detect_sysfs_backends` if not), and `_family_builder_mask`
   (`src/lazy.cyr`). A backend missing from one of them is silently skipped by
   that entry point.
5. **Duplicates**: if Vulkan can see the same device, set the profile's
   `pci_id` (`pci_id_make(vendor, device)`) and teach `_dedup_twin`
   (`src/registry.cyr`) about the new type, so the device is listed once.
6. **Tests**: parser tests against captured output (in `tests/fixtures/`, or
   inline) in the matching unit (`gpu_parser_test`, `backend_test`),
   `foundation_test` for the new names and backend count, and a fuzz harness
   in `fuzz/` for a new text parser.
7. **Docs**: add the name to `docs/schema.json` (the `accelerator` enum, and
   `accel_type_id`'s maximum) and a row to the README's *Supported Hardware*
   table.

## Commit messages

- Use imperative mood: "add TPU v6 detection", not "added" or "adds".
- Keep the first line under 72 characters.
- Reference issues with `#123` where applicable.

## Pull requests

- One logical change per PR.
- Include tests for new functionality.
- Update documentation if public API or output changes.
- For a change to a hot path (parsers, registry queries, JSON output), include
  before/after numbers from the benchmark suites.
- PRs must pass CI.
- Maintainers may request changes before merging.

## Versioning

This project uses semantic versioning. Maintainers bump versions with
`scripts/version-bump.sh`, and every release carries a before/after benchmark
review in its CHANGELOG section. Contributors do **not** need to bump the
version in their PRs.

## License

By contributing you agree that your contributions will be licensed under the
[GNU General Public License v3.0](LICENSE).
