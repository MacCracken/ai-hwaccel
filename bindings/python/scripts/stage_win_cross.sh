#!/usr/bin/env bash
# Cross-build the Windows (PE) ai-hwaccel.exe on Linux and stage it into
# the Python package, so a win_amd64 wheel bundles a self-contained EXE.
#
# Unlike macOS (which needs a remote Mac), Windows is a *Linux* cross
# build: the Linux-hosted `cyrius`/`cycc` toolchain emits PE32+ natively
# via `cyrius build --win` (cyrius 6.0.50+ unfroze the PE surface; 6.0.51
# routed Win32 process creation so detection's subprocess spawns work —
# see CHANGELOG 2.3.7). No Windows runner needed for the build; `cass` is
# used only for runtime smoke.
#
# NOTE (2.3.9): the standalone `cycc_win` binary shipped in the toolchain
# is itself a *Windows PE* (Windows-hosted compiler), not a Linux ELF —
# running it on Linux only works under Wine (the `DOSWin` binfmt handler),
# which a bare CI runner (ubuntu-latest, no Wine) does not have, so the
# old `cycc_win < entry > out.exe` pipe failed in CI with `Exec format
# error` (exit 126). `cyrius build --win` is the correct Linux-native
# cross path: the ELF `cycc` produces the PE directly (no Wine), and the
# wrapper resolves `[deps] stdlib` + includes, so we no longer synthesize
# the translation unit by hand. CYRIUS_TARGET_WIN is auto-defined by the
# --win target, so the #ifdef Windows paths (alloc_windows, process_win,
# detect_windows) compile in.
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

cd "$REPO_ROOT"

# Populate ./lib so includes resolve. `cyrius build` resolves [deps]
# stdlib on its own, but syncing first keeps the snapshot pinned and
# matches the other stage_* scripts. lib sync works on Linux (the Darwin
# getdents64 gap doesn't apply to this build host).
cyrius lib sync >/dev/null 2>&1 || echo ">> lib sync skipped; using existing ./lib"

echo ">> cross-building Windows PE with 'cyrius build --win' (Linux-native, no Wine)"
mkdir -p "$BIN_DIR/data"
# Drop any stray native ELF so the win_amd64 wheel bundles only the EXE
# (CI checkouts start clean; this matters for local back-to-back builds).
rm -f "$BIN_DIR/ai-hwaccel"
CYRIUS_DCE=1 cyrius build --win src/main.cyr "$BIN_DIR/ai-hwaccel.exe"

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
