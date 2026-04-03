#!/bin/bash
set -e

BRANCH=$(git branch --show-current)

if [ "$BRANCH" = "main" ]; then
    echo "Error: Don't run this on main branch"
    exit 1
fi

MSG="${1:-Results from $BRANCH}"

echo "=== Pushing results from $BRANCH ==="

git add .
git commit -m "$MSG"
git push origin "$BRANCH"

echo ""
echo "Push complete."
echo ""

read -p "Delete local outputs to free quota? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Applying skip-worktree and deleting..."

    # 1. Identify tracked files in outputs/ and mark them to be ignored locally
    git ls-files outputs/ | xargs git update-index --skip-worktree
    
    # 2. Safely remove the files from disk
    rm -rf outputs/
    
    # 3. Recreate empty directory for future runs
    mkdir -p outputs
    
    echo "Done. Local outputs deleted and hidden from Git status."
else
    echo "Skipped. Outputs kept locally."
fi
