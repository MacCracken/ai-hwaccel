# ADR-002: Calendar versioning (CalVer)

## Status

Accepted

## Context

The crate needs a versioning scheme. Options considered:

1. **Semantic versioning** (SemVer) — `MAJOR.MINOR.PATCH`.
2. **Calendar versioning** (CalVer) — `YYYY.M.D`.

## Decision

We use CalVer (`YYYY.M.D`) for pre-1.0 releases. After the API stabilises at
v1.0, we will evaluate whether to switch to SemVer.

## Consequences

**Benefits:**

- Version numbers immediately communicate when a release was cut.
- No debates about "is this a minor or patch bump" during rapid early
  development.
- Encourages frequent, small releases.
- The `VERSION` file + `scripts/version-bump.sh` keeps everything in sync.

**Trade-offs:**

- CalVer does not encode breaking-change information. Downstream users cannot
  tell from the version number alone whether an upgrade is safe.
- crates.io and cargo treat versions as SemVer, so `2026.3.19` is technically
  a valid SemVer version (major 2026, minor 3, patch 19) which implies extreme
  stability — misleading for a pre-1.0 crate.

**Mitigations:**

- `CHANGELOG.md` documents every breaking change explicitly.
- `#[non_exhaustive]` on key enums reduces the surface for downstream breakage.
- We plan to adopt SemVer (starting from `1.0.0`) once the API is stable.
