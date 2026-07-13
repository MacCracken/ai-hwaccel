#!/usr/bin/env bash
set -euo pipefail

# Run Cyrius benchmarks and append results to CSV history.
#
# Usage:
#   ./scripts/bench-history.sh
#   ./scripts/bench-history.sh results.csv

HISTORY_FILE="${1:-bench-history.csv}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

if [ ! -f "$HISTORY_FILE" ]; then
    echo "timestamp,commit,branch,suite,benchmark,avg_ns,min_ns,max_ns,iters" > "$HISTORY_FILE"
fi

echo "╔══════════════════════════════════════════╗"
echo "║      ai-hwaccel benchmark suite          ║"
echo "╠══════════════════════════════════════════╣"
echo "║  commit: $COMMIT"
echo "║  branch: $BRANCH"
echo "║  cyrius: $(cyrius --version 2>/dev/null || echo unknown)"
echo "╚══════════════════════════════════════════╝"
echo ""

for f in benches/*.bcyr; do
    name=$(basename "$f" .bcyr)
    echo "--- Building $name ---"
    cyrius build "$f" "/tmp/bench_${name}" 2>&1
    echo "--- Running $name ---"
    result=$("/tmp/bench_${name}" 2>&1)
    echo "$result"

    # Parse output and append to CSV. bench_report (cyrius 6.4.x) prints
    # decimal units — "<int>ns" or "<major>.<3-digit-frac><us|ms|s>"
    # (e.g. "19.460us", "1.712ms"). The frac is thousandths of the unit,
    # so it converts to an exact integer-ns value. Pre-6.4 emitted a
    # truncated integer "<int>us" with no frac; the same converter
    # handles it (frac empty -> 0). This replaced a grep that captured
    # the fractional digits as the whole number (2.3.13 toolchain bump).
    echo "$result" | grep -E '^\s+\S+:' | while IFS= read -r line; do
        bname=$(echo "$line" | sed 's/^[[:space:]]*//' | cut -d: -f1)
        csv=$(echo "$line" | awk '
            function tons(tok,   n,u,dot,ip,fr,fv) {
                if (!match(tok, /[0-9.]+/)) return 0
                n = substr(tok, RSTART, RLENGTH); u = substr(tok, RSTART + RLENGTH)
                dot = index(n, ".")
                if (dot) { ip = substr(n, 1, dot - 1) + 0; fr = substr(n, dot + 1) }
                else     { ip = n + 0; fr = "" }
                fv = (fr == "") ? 0 : substr(fr "000", 1, 3) + 0
                if (u ~ /^ns/) return ip
                if (u ~ /^us/) return ip * 1000 + fv
                if (u ~ /^ms/) return ip * 1000000 + fv * 1000
                if (u ~ /^s/)  return ip * 1000000000 + fv * 1000000
                return ip
            }
            {
                avg = "0"; for (i = 1; i <= NF; i++) if ($i == "avg") avg = $(i - 1)
                mn = "0"; if (match($0, /min=[0-9.]+[a-z]+/)) mn = substr($0, RSTART + 4, RLENGTH - 4)
                mx = "0"; if (match($0, /max=[0-9.]+[a-z]+/)) mx = substr($0, RSTART + 4, RLENGTH - 4)
                it = "0"; if (match($0, /[0-9]+ iters/))      it = substr($0, RSTART, RLENGTH - 6) + 0
                printf "%d,%d,%d,%d", tons(avg), tons(mn), tons(mx), it
            }')
        echo "$TIMESTAMP,$COMMIT,$BRANCH,$name,$bname,$csv" >> "$HISTORY_FILE"
    done
    echo ""
done

echo "Results appended to $HISTORY_FILE"
