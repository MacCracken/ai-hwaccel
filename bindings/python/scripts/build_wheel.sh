#!/usr/bin/env bash
# Build a platform-tagged wheel from the already-staged _bin/ contents.
#
# Stage first (native or a copied cross-built binary):
#   scripts/stage_binary.sh            # host binary into _bin/
# then:
#   scripts/build_wheel.sh manylinux2014_x86_64
#
# Platform tags per target:
#   linux x86_64   -> manylinux2014_x86_64   (binary is static; portable)
#   linux aarch64  -> manylinux2014_aarch64
#   macOS arm64    -> macosx_11_0_arm64
#   windows x86_64 -> win_amd64
#
# Usage: build_wheel.sh <platform-tag> [python]
set -euo pipefail
PLAT="${1:?usage: build_wheel.sh <platform-tag> [python]}"
PY="${2:-python3}"

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$PKG_DIR/src/ai_hwaccel/_bin"
if [ ! -f "$BIN_DIR/ai-hwaccel" ] && [ ! -f "$BIN_DIR/ai-hwaccel.exe" ]; then
    echo "no staged binary in $BIN_DIR — run scripts/stage_binary.sh first" >&2
    exit 1
fi

export AIH_WHEEL_PLAT="$PLAT"
# Clear setuptools' build/ cache: package_data (the _bin/ binary) is
# copied into build/lib.../ and stale per-platform binaries from a prior
# run would otherwise be bundled (e.g. a Linux ELF leaking into a
# win_amd64 wheel). CI checkouts are clean; this matters for local
# back-to-back cross-platform builds.
rm -rf "$PKG_DIR/build"
( cd "$PKG_DIR" && "$PY" -m build --wheel --outdir dist )
echo "Built:"
ls -1 "$PKG_DIR/dist/"
