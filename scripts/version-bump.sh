#!/usr/bin/env bash
# Version bump — VERSION is the single source of truth. cyrius.cyml
# carries `version = "${file:VERSION}"` so the manifest tracks
# automatically.
#
# IMPORTANT: dist/ai-hwaccel.cyr embeds the project version in its
# header (`# Version: X`), so it MUST be regenerated on every bump or
# the CI "distlib drift" gate fails — even when no .cyr source changed.
# This script regenerates it for you, using the pinned toolchain
# (~/.cyrius/versions/<pin>/bin/cyrius) so the output matches CI, which
# installs the pinned version rather than whatever the local wrapper has
# drifted to. distlib output is version-independent apart from that
# header line, but matching CI exactly is the safe default. It runs with
# --no-default-features, as CI does, so the bundle's `# Requires` block is
# verified without the optional bayan dep (see ci.yml's distlib step).
#
# Every step that can fail runs before VERSION is written, or puts VERSION
# back when it fails (2.5.5: a failed distlib used to be reported as a
# successful bump, and the GNU-only sed below failed on macOS after VERSION
# was already written).
#
# Tag and push after bumping.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>  (current: $(cat "$REPO_ROOT/VERSION"))" >&2
    exit 1
fi
NEW_VERSION="$1"
# release.yml accepts a `v` on the tag, but VERSION itself is bare x.y.z.
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "error: version must be x.y.z (digits only, no leading v): '$NEW_VERSION'" >&2
    exit 1
fi

PIN="$(grep -E '^cyrius[[:space:]]*=' "$REPO_ROOT/cyrius.cyml" | sed -E 's/.*"([^"]+)".*/\1/')"
PINNED_BIN="$HOME/.cyrius/versions/${PIN}/bin/cyrius"
if [ -x "$PINNED_BIN" ]; then
    CYBIN="$PINNED_BIN"
elif command -v cyrius >/dev/null 2>&1; then
    CYBIN="cyrius"
    echo "warning: pinned cyrius ${PIN} not found at ${PINNED_BIN}; using PATH cyrius (verify CI parity)."
else
    echo "error: no cyrius toolchain found; install cyrius ${PIN}, which dist/ai-hwaccel.cyr is regenerated with. Nothing was changed." >&2
    exit 1
fi

OLD_VERSION="$(tr -d '[:space:]' < "$REPO_ROOT/VERSION")"
echo "$NEW_VERSION" > "$REPO_ROOT/VERSION"

# Regenerate the dist bundle so its embedded version matches.
if ! (cd "$REPO_ROOT" && "$CYBIN" distlib --no-default-features >/dev/null); then
    echo "$OLD_VERSION" > "$REPO_ROOT/VERSION"
    echo "error: 'cyrius distlib --no-default-features' failed; VERSION is back to ${OLD_VERSION}." >&2
    echo "       Check dist/ai-hwaccel.cyr (git diff) before retrying." >&2
    exit 1
fi
echo "Regenerated dist/ai-hwaccel.cyr (embeds ${NEW_VERSION})."

# Propagate into the Python bindings' pyproject.toml. setuptools cannot
# read VERSION from outside the package root, so the wheel metadata
# version is a literal there; keep it in lockstep with VERSION here (CI's
# "python bindings version" gate enforces the match). Only the first
# `version = "..."` line, which is [project]'s. awk to a temp file rather
# than `sed -i`, whose GNU and BSD forms differ. The package's runtime
# __version__ derives from this via installed metadata.
PYPROJECT="$REPO_ROOT/bindings/python/pyproject.toml"
if [ -f "$PYPROJECT" ]; then
    awk -v v="$NEW_VERSION" '!done && /^version = "/ { print "version = \"" v "\""; done = 1; next } { print }' \
        "$PYPROJECT" > "$PYPROJECT.tmp"
    mv "$PYPROJECT.tmp" "$PYPROJECT"
    echo "Synced bindings/python/pyproject.toml (version = ${NEW_VERSION})."
fi

echo "Bumped to ${NEW_VERSION}. Add the section to CHANGELOG.md, then tag and push."
