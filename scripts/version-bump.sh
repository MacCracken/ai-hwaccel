#!/usr/bin/env bash
# Version bump — VERSION is the single source of truth. cyrius.cyml
# carries `version = "${file:VERSION}"` so the manifest tracks
# automatically. Tag and push after bumping.
set -euo pipefail
[ $# -ne 1 ] && echo "Usage: $0 <version>  (current: $(cat VERSION))" && exit 1
NEW_VERSION="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "$NEW_VERSION" > "$REPO_ROOT/VERSION"
echo "Bumped to ${NEW_VERSION}. Add the section to CHANGELOG.md, then tag and push."
