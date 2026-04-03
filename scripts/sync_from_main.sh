#!/usr/bin/env bash
set -euo pipefail

BRANCH=$(git branch --show-current)

if [[ "$BRANCH" == "main" ]]; then
  echo "Error: You're on main branch"
  exit 1
fi

echo "=== Syncing from origin/main → $BRANCH (excluding outputs/) ==="

echo "Fetching origin/main..."
git fetch origin main

# Get changed files vs origin/main (exclude outputs/)
mapfile -t FILES < <(git diff --name-only origin/main -- . ':(exclude)outputs/**')

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No files to sync."
  exit 0
fi

echo "Files to sync:"
printf '  %s\n' "${FILES[@]}"
echo ""

echo "Restoring files from origin/main..."
git restore --source=origin/main -- "${FILES[@]}"

# Only commit if something actually changed
if ! git diff --quiet; then
  git add "${FILES[@]}"
  git commit -m "synched"
  git push
  echo "Committed and pushed."
else
  echo "No changes after restore."
fi

echo "=== Done ==="
