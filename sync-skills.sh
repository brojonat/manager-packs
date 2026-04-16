#!/usr/bin/env bash
set -euo pipefail

# sync-skills.sh — copy skills from manager-pack/bundles/ to llmsrules/skills/
#
# manager-pack is the source of truth for skill content. This script copies
# files INTO llmsrules, skipping manager-pack-specific files (manifest.json,
# build artifacts, caches). Does NOT delete files in llmsrules — if a file
# is removed locally, remove it from llmsrules manually. This protects
# against accidentally wiping the release repo when assets/*.html files
# (which are gitignored here) haven't been regenerated yet.
#
# Usage:
#   ./sync-skills.sh                     # sync ALL skills
#   ./sync-skills.sh bayesian-ab-testing # sync one skill
#   ./sync-skills.sh --dry-run           # show what would be copied
#   ./sync-skills.sh --dry-run go-service
#
# Tip: use `make sync` to chain `make export-html` + this script in one shot.

BUNDLES_DIR="$(cd "$(dirname "$0")" && pwd)/bundles"
TARGET_DIR="${LLMSRULES_SKILLS:-/home/brojonat/projects/llmsrules/skills}"

# Files/dirs to skip when syncing (manager-pack-specific or build artifacts)
EXCLUDE=(
    "manifest.json"
    "__marimo__"
    "__pycache__"
    "mlflow.db"
    "unsloth_compiled_cache"
    "_unsloth_sentencepiece_temp"
    "nhtsa_cache/FLAT_CMPL.txt"
)

DRY_RUN=false
SKILLS=()

# Parse args
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ] || [ "$arg" = "-n" ]; then
        DRY_RUN=true
    else
        SKILLS+=("$arg")
    fi
done

# If no skills specified, sync all
if [ ${#SKILLS[@]} -eq 0 ]; then
    for d in "$BUNDLES_DIR"/*/; do
        SKILLS+=("$(basename "$d")")
    done
fi

# Build rsync exclude flags
RSYNC_EXCLUDES=()
for pattern in "${EXCLUDE[@]}"; do
    RSYNC_EXCLUDES+=(--exclude "$pattern")
done

synced=0
for skill in "${SKILLS[@]}"; do
    src="$BUNDLES_DIR/$skill/"
    dst="$TARGET_DIR/$skill/"

    if [ ! -d "$src" ]; then
        echo "SKIP $skill (not found in bundles/)"
        continue
    fi

    if $DRY_RUN; then
        echo "=== $skill (dry run) ==="
        rsync -avn "${RSYNC_EXCLUDES[@]}" "$src" "$dst" 2>&1 | grep -v '^\.$' | sed 's/^/  /'
    else
        mkdir -p "$dst"
        rsync -a "${RSYNC_EXCLUDES[@]}" "$src" "$dst"
        echo "  synced: $skill"
    fi
    synced=$((synced + 1))
done

echo "--- $synced skill(s) processed ---"
