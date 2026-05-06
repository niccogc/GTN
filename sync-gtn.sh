#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REL_TARGET="outputs"
OTHER_FOLDER="/work3/aveno/repos/GTN"

TARGET="$SCRIPT_DIR/$REL_TARGET"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

sync_host () {
  local HOST="$1"

  local TMPDIR="$TMP/$HOST"
  mkdir -p "$TMPDIR"

  echo "Fetching ~/GTN/$REL_TARGET from $HOST..."
  ssh "$HOST" "
    tar -I zstd -cf - -C ~/GTN \"$REL_TARGET\"
  " | tar -I zstd -xf - -C "$TMPDIR"

  if [[ "$HOST" != "titans" ]]; then
    echo "Fetching $OTHER_FOLDER/$REL_TARGET from $HOST..."
    ssh "$HOST" "
      tar -I zstd -cf - -C \"$OTHER_FOLDER\" \"$REL_TARGET\"
    " | tar -I zstd -xf - -C "$TMPDIR"
  fi
  
  mkdir -p "$TARGET"

  echo "Merging outputs from $HOST..."
  rsync -a "$TMPDIR/$REL_TARGET/" "$TARGET/"
}

sync_host titans &
sync_host hpc &

wait
echo "Done."
