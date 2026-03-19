#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <new-version>"
    echo "Example: $0 2026.3.20"
    exit 1
fi

NEW_VERSION="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Bumping ai-hwaccel to version ${NEW_VERSION}..."

# Update VERSION file
echo "$NEW_VERSION" > "$REPO_ROOT/VERSION"
echo "  Updated VERSION"

# Update Cargo.toml
sed -i "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" "$REPO_ROOT/Cargo.toml"
echo "  Updated Cargo.toml"

# Regenerate Cargo.lock
cd "$REPO_ROOT"
cargo generate-lockfile 2>/dev/null
echo "  Regenerated Cargo.lock"

# Validate
FILE_VERSION=$(cat "$REPO_ROOT/VERSION" | tr -d '[:space:]')
CARGO_VERSION=$(grep '^version = ' "$REPO_ROOT/Cargo.toml" | head -1 | sed 's/version = "\(.*\)"/\1/')

if [ "$FILE_VERSION" != "$NEW_VERSION" ]; then
    echo "ERROR: VERSION file mismatch: expected $NEW_VERSION, got $FILE_VERSION"
    exit 1
fi

if [ "$CARGO_VERSION" != "$NEW_VERSION" ]; then
    echo "ERROR: Cargo.toml mismatch: expected $NEW_VERSION, got $CARGO_VERSION"
    exit 1
fi

echo ""
echo "Version bumped to ${NEW_VERSION}"
echo ""
echo "Next steps:"
echo "  git add VERSION Cargo.toml Cargo.lock"
echo "  git commit -m \"bump to ${NEW_VERSION}\""
echo "  git tag ${NEW_VERSION}"
echo "  git push && git push --tags"
