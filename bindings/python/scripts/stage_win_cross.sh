#!/usr/bin/env bash
# Cross-build the Windows (PE) ai-hwaccel.exe on Linux and stage it into
# the Python package, so a win_amd64 wheel bundles a self-contained EXE.
#
# Unlike macOS (which needs a remote Mac), Windows is a *Linux* cross
# build: cyrius's `cycc_win` is a Linux-hosted ELF compiler that emits
# PE32+ (cyrius 6.0.50+ unfroze it; 6.0.51 routed Win32 process creation
# so detection's subprocess spawns work — see CHANGELOG 2.3.7). No
# Windows runner needed for the build; `cass` is used only for runtime
# smoke.
#
# `cycc_win` is the raw compiler (stdin -> PE) and does NOT do [deps]
# stdlib resolution the way `cyrius build` does, so we synthesize the
# same translation unit the wrapper would: every `[deps] stdlib` module
# (in manifest order) followed by src/main.cyr, piped to cycc_win.
# CYRIUS_TARGET_WIN is auto-defined by cycc_win, so the #ifdef Windows
# paths (alloc_windows, process_win, detect_windows) compile in.
#
# Layout produced (consumed by ai_hwaccel._runner via AI_HWACCEL_DATA_DIR):
#   src/ai_hwaccel/_bin/{ai-hwaccel.exe, VERSION, data/cloud_pricing.json}
#
# Usage: bindings/python/scripts/stage_win_cross.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # bindings/python
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"
BIN_DIR="$PKG_DIR/src/ai_hwaccel/_bin"

PIN="$(grep -E '^cyrius[[:space:]]*=' "$REPO_ROOT/cyrius.cyml" | sed -E 's/.*"([^"]+)".*/\1/')"
CYCC_WIN="$HOME/.cyrius/versions/${PIN}/bin/cycc_win"
[ -x "$CYCC_WIN" ] || CYCC_WIN="cycc_win"   # fall back to PATH

cd "$REPO_ROOT"

# Populate ./lib so the `include "lib/<mod>.cyr"` lines resolve. lib sync
# works on Linux (the Darwin getdents64 gap doesn't apply to the build host).
cyrius lib sync >/dev/null 2>&1 || echo ">> lib sync skipped; using existing ./lib"

# Synthesize the translation unit: [deps] stdlib (manifest order) + main.
ENTRY="$(mktemp -t aih-win-XXXXXX.cyr)"
trap 'rm -f "$ENTRY"' EXIT
grep -E '^stdlib[[:space:]]*=' cyrius.cyml \
    | grep -oE '"[^"]+"' | tr -d '"' \
    | while IFS= read -r mod; do echo "include \"lib/${mod}.cyr\""; done > "$ENTRY"
echo 'include "src/main.cyr"' >> "$ENTRY"

echo ">> cross-building Windows PE with $CYCC_WIN"
mkdir -p "$BIN_DIR/data"
# Drop any stray native ELF so the win_amd64 wheel bundles only the EXE
# (CI checkouts start clean; this matters for local back-to-back builds).
rm -f "$BIN_DIR/ai-hwaccel"
CYRIUS_DCE=1 "$CYCC_WIN" < "$ENTRY" > "$BIN_DIR/ai-hwaccel.exe"

# A PE32+ starts with "MZ"; sanity-check we didn't capture an error.
if [ "$(head -c2 "$BIN_DIR/ai-hwaccel.exe")" != "MZ" ]; then
    echo "ERROR: output is not a PE binary (cross-build failed)" >&2
    exit 1
fi

cp "$REPO_ROOT/VERSION" "$BIN_DIR/VERSION"
cp "$REPO_ROOT/data/cloud_pricing.json" "$BIN_DIR/data/cloud_pricing.json"

echo "Staged win_amd64 EXE into $BIN_DIR:"
ls -1 "$BIN_DIR"
file "$BIN_DIR/ai-hwaccel.exe" 2>/dev/null || true
echo "Next: scripts/build_wheel.sh win_amd64"
