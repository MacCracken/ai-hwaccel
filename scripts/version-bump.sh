#!/bin/sh
# Version bump script — single source of truth for all version references
# Usage: ./scripts/version-bump.sh 2.1.0

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Current: $(cat VERSION)"
    exit 1
fi

NEW="$1"
OLD=$(cat VERSION | tr -d '[:space:]')

if [ "$NEW" = "$OLD" ]; then
    echo "Already at $OLD"
    exit 0
fi

# 1. VERSION file (source of truth)
echo "$NEW" > VERSION

# 2. cyrius.toml
sed -i "s/version = \"$OLD\"/version = \"$NEW\"/" cyrius.toml 2>/dev/null || true

# 3. CLAUDE.md
sed -i "s/- \*\*Version\*\*: SemVer (post-1.0)/- **Version**: SemVer ($NEW)/" CLAUDE.md 2>/dev/null || true

# 4. CHANGELOG.md — add unreleased section if not present
if ! grep -q "## \[$NEW\]" CHANGELOG.md 2>/dev/null; then
    sed -i "/## \[Unreleased\]/a\\
\\
## [$NEW] — $(date +%Y-%m-%d)" CHANGELOG.md 2>/dev/null || true
fi

# 5. README.md — update key numbers if present
# (version shown via --version output, not hardcoded)

# 6. Validate
FILE_VERSION=$(cat VERSION | tr -d '[:space:]')
TOML_VERSION=$(grep '^version = ' cyrius.toml | head -1 | sed 's/version = "\(.*\)"/\1/')

if [ "$FILE_VERSION" != "$NEW" ]; then
    echo "ERROR: VERSION file mismatch: expected $NEW, got $FILE_VERSION"
    exit 1
fi

if [ "$TOML_VERSION" != "$NEW" ]; then
    echo "ERROR: cyrius.toml mismatch: expected $NEW, got $TOML_VERSION"
    exit 1
fi

echo "$OLD -> $NEW"
echo ""
echo "Updated:"
echo "  VERSION"
echo "  cyrius.toml"
echo "  CHANGELOG.md"
echo ""
echo "Still manual:"
echo "  - CHANGELOG.md entries (add sections under new version)"
echo "  - README.md key numbers if binary size changed"
echo "  - docs/benchmarks-rust-v-cyrius.md if benchmarks changed"
