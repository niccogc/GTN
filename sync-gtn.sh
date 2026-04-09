#!/usr/bin/env bash

set -euo pipefail

TARGET="/home/nicci/Desktop/remote/GTN/outputs"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

sync_host () {
  local HOST="$1"
  local TMPDIR="$TMP/$HOST"

  mkdir -p "$TMPDIR"

  echo "Fetching from $HOST..."

  ssh "$HOST" 'tar -I zstd -cf - -C ~/GTN outputs' \
    | tar -I zstd -xf - -C "$TMPDIR"

  echo "Merging from $HOST..."

  rsync -a --update "$TMPDIR/outputs/" "$TARGET/"
}

sync_host titans &
sync_host transfer.gbar.dtu.dk &

wait
echo "Done."
