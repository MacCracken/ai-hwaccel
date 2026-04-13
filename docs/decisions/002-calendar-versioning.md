# ADR-002: CalVer to SemVer transition

## Status

Superseded — switched to SemVer at v0.19.3.

## Context

The crate initially used calendar versioning (`YYYY.M.D`) during rapid early
development. As the API stabilised, the lack of breaking-change signalling
became a liability for downstream consumers.

## Decision

Switch to semantic versioning starting at `0.19.3`. The `0.x` series indicates
pre-1.0 instability — breaking changes may occur between minor versions. Once
the API is frozen, we will release `1.0.0`.

## Consequences

- Downstream users can rely on SemVer guarantees once we reach `1.0.0`.
- The `VERSION` file + `scripts/version-bump.sh` continue to work unchanged.
- The release workflow (`release.yml`) validates VERSION file/tag
  consistency regardless of versioning scheme.
