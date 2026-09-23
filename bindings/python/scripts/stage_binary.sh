#!/usr/bin/env bash
# Stage the per-platform binary + its data files into the Python package
# so a wheel (2.3.4) bundles a self-contained ai-hwaccel.
#
# Layout produced (consumed by ai_hwaccel._runner, which sets
# AI_HWACCEL_DATA_DIR to this dir for the bundled binary):
#
#   src/ai_hwaccel/_bin/
#     ai-hwaccel                   (the compiled binary)
#     VERSION                      (so --version works off-cwd)
#     data/cloud_pricing.json      (so --cost works off-cwd)
#
# Builds with the PINNED cyrius (matching CI), falling back to PATH.
# Usage: bindings/python/scripts/stage_binary.sh [--aarch64]
set -euo pipefail

EXTRA_ARGS=()
[ "${1:-}" = "--aarch64" ] && EXTRA_ARGS+=("--aarch64")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # bindings/python
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"
BIN_DIR="$PKG_DIR/src/ai_hwaccel/_bin"

PIN="$(grep -E '^cyrius[[:space:]]*=' "$REPO_ROOT/cyrius.cyml" | sed -E 's/.*"([^"]+)".*/\1/')"
PINNED_BIN="$HOME/.cyrius/versions/${PIN}/bin/cyrius"
if [ -x "$PINNED_BIN" ]; then CYBIN="$PINNED_BIN"; else CYBIN="cyrius"; fi

EXE_NAME="ai-hwaccel"
case "${EXTRA_ARGS[*]:-}" in *--aarch64*) TAG="aarch64";; *) TAG="native";; esac

echo "Staging ai-hwaccel ($TAG) with $CYBIN into $BIN_DIR"
mkdir -p "$BIN_DIR/data"
# Drop any stray Windows EXE so a native wheel bundles only the ELF/Mach-O.
rm -f "$BIN_DIR/ai-hwaccel.exe"

if [ "$(uname -s)" = "Darwin" ] && [ "$TAG" = "native" ]; then
    # macOS: `cyrius build` checks and rewrites cyrius.lock with sha256sum, which it
    # cannot reach on the macOS runner, so the build runs as its two halves instead:
    #   1. `cyrius deps --no-lock` vendors ./lib, with cyrius.lock set aside so it is
    #      neither checked nor rewritten here (CI's Linux job verifies it); the lock
    #      is put back when this script exits.
    #   2. The pinned cycc compiles the same input `cyrius build` composes (#@incdir,
    #      #@pkgver, one include per [deps].stdlib leaf in manifest order, bayan's
    #      result leaf and module, #@srcline, then src/main.cyr), in the empty
    #      environment `cyrius build` gives it on macOS. Checked byte-identical to
    #      `cyrius build src/main.cyr` output.
    CYCC="$(dirname "$CYBIN")/cycc"
    [ -x "$CYCC" ] || CYCC="$(command -v cycc)"
    LOCK_BAK=""
    TU=""
    cleanup() {
        if [ -n "$TU" ]; then rm -f "$TU"; fi
        rm -f "$BIN_DIR/$EXE_NAME.tmp"
        if [ -n "$LOCK_BAK" ] && [ -f "$LOCK_BAK" ]; then mv -f "$LOCK_BAK" "$REPO_ROOT/cyrius.lock"; fi
        return 0
    }
    trap cleanup EXIT
    if [ -f "$REPO_ROOT/cyrius.lock" ]; then
        LOCK_BAK="$(mktemp "$REPO_ROOT/.cyrius.lock.stage.XXXXXX")"
        mv -f "$REPO_ROOT/cyrius.lock" "$LOCK_BAK"
    fi
    ( cd "$REPO_ROOT" && "$CYBIN" deps --no-lock )

    TU="$(mktemp "${TMPDIR:-/tmp}/aih-stage-tu.XXXXXX")"
    VER="$(tr -d ' \t\r\n' < "$REPO_ROOT/VERSION")"
    LEAVES="$(grep -E '^stdlib[[:space:]]*=' "$REPO_ROOT/cyrius.cyml" | sed -E 's/^[^[]*\[//; s/].*//' | tr ',' '\n' | tr -d ' "')"
    {
        printf '#@incdir src\n#@pkgver %s\n' "$VER"
        for leaf in $LEAVES; do printf 'include "lib/%s.cyr"\n' "$leaf"; done
        printf 'include "lib/result.cyr"\ninclude "lib/bayan-json.cyr"\n#@srcline\n'
        cat "$REPO_ROOT/src/main.cyr"
    } > "$TU"
    ( cd "$REPO_ROOT" && env -i "$CYCC" < "$TU" > "$BIN_DIR/$EXE_NAME.tmp" )
    [ -s "$BIN_DIR/$EXE_NAME.tmp" ] || { echo "ERROR: cycc produced no binary" >&2; exit 1; }
    mv -f "$BIN_DIR/$EXE_NAME.tmp" "$BIN_DIR/$EXE_NAME"
else
    # `${arr[@]+"${arr[@]}"}` expands to nothing on an EMPTY array without
    # tripping `set -u` — the bare `"${arr[@]}"` is an "unbound variable"
    # error on the macOS runner's bash 3.2 (native staging passes no flag, so
    # EXTRA_ARGS is empty). Do NOT use `"${arr[@]:-}"` here: that injects an
    # empty-string argv into `cyrius build`.
    ( cd "$REPO_ROOT" && CYRIUS_DCE=1 "$CYBIN" build ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} src/main.cyr "$BIN_DIR/$EXE_NAME" )
fi
chmod +x "$BIN_DIR/$EXE_NAME"
cp "$REPO_ROOT/VERSION" "$BIN_DIR/VERSION"
cp "$REPO_ROOT/data/cloud_pricing.json" "$BIN_DIR/data/cloud_pricing.json"

echo "Staged:"
ls -1 "$BIN_DIR" "$BIN_DIR/data"
