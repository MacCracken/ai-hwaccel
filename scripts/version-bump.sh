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
# header line, but matching CI exactly is the safe default.
#
# Tag and push after bumping.
set -euo pipefail
[ $# -ne 1 ] && echo "Usage: $0 <version>  (current: $(cat VERSION))" && exit 1
NEW_VERSION="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "$NEW_VERSION" > "$REPO_ROOT/VERSION"

# Regenerate the dist bundle so its embedded version matches.
PIN="$(grep -E '^cyrius[[:space:]]*=' "$REPO_ROOT/cyrius.cyml" | sed -E 's/.*"([^"]+)".*/\1/')"
PINNED_BIN="$HOME/.cyrius/versions/${PIN}/bin/cyrius"
if [ -x "$PINNED_BIN" ]; then
    CYBIN="$PINNED_BIN"
elif command -v cyrius >/dev/null 2>&1; then
    CYBIN="cyrius"
    echo "warning: pinned cyrius ${PIN} not found at ${PINNED_BIN}; using PATH cyrius (verify CI parity)."
else
    CYBIN=""
    echo "warning: no cyrius toolchain found — regenerate dist/ai-hwaccel.cyr manually with 'cyrius distlib'."
fi
if [ -n "$CYBIN" ]; then
    (cd "$REPO_ROOT" && "$CYBIN" distlib >/dev/null) \
        && echo "Regenerated dist/ai-hwaccel.cyr (embeds ${NEW_VERSION})."
fi

# Propagate into the Python bindings' pyproject.toml. setuptools cannot
# read VERSION from outside the package root, so the wheel metadata
# version is a literal there; keep it in lockstep with VERSION here (CI's
# "python bindings version" gate enforces the match). Only the first
# `version = "..."` under [project] — a plain sed of the first match. The
# package's runtime __version__ derives from this via installed metadata.
PYPROJECT="$REPO_ROOT/bindings/python/pyproject.toml"
if [ -f "$PYPROJECT" ]; then
    sed -i.bak -E "0,/^version = \"[^\"]*\"/s//version = \"${NEW_VERSION}\"/" "$PYPROJECT"
    rm -f "$PYPROJECT.bak"
    echo "Synced bindings/python/pyproject.toml (version = ${NEW_VERSION})."
fi

echo "Bumped to ${NEW_VERSION}. Add the section to CHANGELOG.md, then tag and push."
