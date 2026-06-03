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
# `${arr[@]+"${arr[@]}"}` expands to nothing on an EMPTY array without
# tripping `set -u` — the bare `"${arr[@]}"` is an "unbound variable"
# error on the macOS runner's bash 3.2 (native staging passes no flag, so
# EXTRA_ARGS is empty). Do NOT use `"${arr[@]:-}"` here: that injects an
# empty-string argv into `cyrius build`.
( cd "$REPO_ROOT" && CYRIUS_DCE=1 "$CYBIN" build ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} src/main.cyr "$BIN_DIR/$EXE_NAME" )
chmod +x "$BIN_DIR/$EXE_NAME"
cp "$REPO_ROOT/VERSION" "$BIN_DIR/VERSION"
cp "$REPO_ROOT/data/cloud_pricing.json" "$BIN_DIR/data/cloud_pricing.json"

echo "Staged:"
ls -1 "$BIN_DIR" "$BIN_DIR/data"
