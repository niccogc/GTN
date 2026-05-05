#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_TARGET="test_outputs"
TARGET="$SCRIPT_DIR/$REL_TARGET"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

sync_host () {
  local HOST="$1"
  local TMPDIR="$TMP/$HOST"

  mkdir -p "$TMPDIR"

  echo "Fetching from $HOST..."
    ssh "$HOST" "tar -I zstd -cf - -C ~/GTN \"$REL_TARGET\"" \
    | tar -I zstd -xf - -C "$TMPDIR"

  echo "Merging from $HOST..."

  rsync -a "$TMPDIR/$REL_TARGET/" "$TARGET/"
}

# sync_host titans &
sync_host hpc &

wait
echo "Done."
