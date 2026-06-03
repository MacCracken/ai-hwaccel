#!/usr/bin/env bash
# Build the ai-hwaccel binary on a remote SSH host and stage it locally.
#
# For platforms with no local cross-compiler (macOS — cyrius emits Mach-O
# only on a Mac). Ships the source tree to the host, runs `cyrius lib sync`
# + `cyrius build` there, and copies the resulting binary back into the
# Python package's _bin/ (alongside VERSION + data/). Then package it:
#
#   scripts/build_remote.sh ecb macosx_11_0_arm64
#   scripts/build_wheel.sh  macosx_11_0_arm64
#
# Assumes the remote has cyrius on PATH or at ~/.cyrius/bin. The remote
# needs no repo checkout — this script provides the sources.
#
# Usage: build_remote.sh <ssh-host> <plat-tag> [remote-exe-name]
set -euo pipefail
HOST="${1:?usage: build_remote.sh <ssh-host> <plat-tag> [exe-name]}"
PLAT="${2:?platform tag, e.g. macosx_11_0_arm64}"
EXE="${3:-ai-hwaccel}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BIN_DIR="$SCRIPT_DIR/../src/ai_hwaccel/_bin"
REMOTE_DIR="aih-build-$PLAT"

echo ">> packing sources"
TARBALL="$(mktemp -t aih-src-XXXXXX).tar.gz"
tar -C "$REPO_ROOT" -czf "$TARBALL" src cyrius.cyml VERSION data

echo ">> shipping to $HOST:$REMOTE_DIR"
ssh "$HOST" "rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR"
scp -q "$TARBALL" "$HOST:$REMOTE_DIR/src.tar.gz"
rm -f "$TARBALL"

echo ">> building on $HOST"
# Non-login ssh may not have cyrius on PATH; add the canonical location.
#
# `cyrius lib sync` is best-effort: on macOS (Darwin) it false-negatives
# the snapshot dir (the directory-listing / getdents64 surface is still
# unported there — tracked in cyrius issue
# 2026-06-02-macos-arm64-deps-stdlib-pin-check.md, "separate, still-open"
# note). It is not required: `cyrius build` resolves `[deps] stdlib` into
# ./lib by name (the path fixed in cyrius v6.0.40–.43), so the build
# populates its own lib. Keep the sync where it works, tolerate it where
# it doesn't.
ssh "$HOST" "cd $REMOTE_DIR && tar xzf src.tar.gz && \
    export PATH=\"\$HOME/.cyrius/bin:\$PATH\" && \
    { cyrius lib sync || echo '>> lib sync unsupported here; [deps] resolution will populate ./lib'; } && \
    CYRIUS_DCE=1 cyrius build src/main.cyr $EXE && \
    file $EXE"

echo ">> retrieving binary"
mkdir -p "$BIN_DIR/data"
scp -q "$HOST:$REMOTE_DIR/$EXE" "$BIN_DIR/ai-hwaccel"
chmod +x "$BIN_DIR/ai-hwaccel"
cp "$REPO_ROOT/VERSION" "$BIN_DIR/VERSION"
cp "$REPO_ROOT/data/cloud_pricing.json" "$BIN_DIR/data/cloud_pricing.json"

echo ">> cleaning remote"
ssh "$HOST" "rm -rf $REMOTE_DIR"

echo "Staged $PLAT binary into $BIN_DIR:"
file "$BIN_DIR/ai-hwaccel"
echo "Next: scripts/build_wheel.sh $PLAT"
