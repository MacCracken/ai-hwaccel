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

    # Parse output and append to CSV
    echo "$result" | grep -E '^\s+\S+:' | while IFS= read -r line; do
        bname=$(echo "$line" | sed 's/^\s*//' | cut -d: -f1)
        avg=$(echo "$line" | grep -oP '\d+(?=ns avg)' || echo "0")
        if [ -z "$avg" ] || [ "$avg" = "0" ]; then
            avg=$(echo "$line" | grep -oP '\d+(?=us avg)' || echo "0")
            [ -n "$avg" ] && [ "$avg" != "0" ] && avg=$((avg * 1000))
        fi
        min=$(echo "$line" | grep -oP 'min=\K\d+(?=ns)' || echo "0")
        if [ -z "$min" ] || [ "$min" = "0" ]; then
            min=$(echo "$line" | grep -oP 'min=\K\d+(?=us)' || echo "0")
            [ -n "$min" ] && [ "$min" != "0" ] && min=$((min * 1000))
        fi
        max=$(echo "$line" | grep -oP 'max=\K\d+(?=ns)' || echo "0")
        if [ -z "$max" ] || [ "$max" = "0" ]; then
            max=$(echo "$line" | grep -oP 'max=\K\d+(?=us)' || echo "0")
            [ -n "$max" ] && [ "$max" != "0" ] && max=$((max * 1000))
        fi
        iters=$(echo "$line" | grep -oP '\d+(?= iters)' || echo "0")
        echo "$TIMESTAMP,$COMMIT,$BRANCH,$name,$bname,$avg,$min,$max,$iters" >> "$HISTORY_FILE"
    done
    echo ""
done

echo "Results appended to $HISTORY_FILE"
