#!/usr/bin/env bash
cd "$(dirname "$0")/../results" || exit 1

echo "=== GTN Datasets ==="
for dir in gtn_*; do
  [[ "$dir" == *_lmpo2 ]] && continue
  dataset="${dir#gtn_}"
  std_complete=$([[ -f "$dir/.complete" ]] && echo "✓" || echo "✗")
  lmpo2_complete=$([[ -f "gtn_${dataset}_lmpo2/.complete" ]] && echo "✓" || echo "✗")
  both=$([[ "$std_complete" == "✓" && "$lmpo2_complete" == "✓" ]] && echo "DONE" || echo "")
  printf "%-20s std:%s  lmpo2:%s  %s\n" "$dataset" "$std_complete" "$lmpo2_complete" "$both"
done

echo ""
echo "=== NTN Datasets ==="
for dir in ntn_*; do
  [[ "$dir" == *_lmpo2 ]] && continue
  dataset="${dir#ntn_}"
  std_complete=$([[ -f "$dir/.complete" ]] && echo "✓" || echo "✗")
  lmpo2_complete=$([[ -f "ntn_${dataset}_lmpo2/.complete" ]] && echo "✓" || echo "✗")
  both=$([[ "$std_complete" == "✓" && "$lmpo2_complete" == "✓" ]] && echo "DONE" || echo "")
  printf "%-20s std:%s  lmpo2:%s  %s\n" "$dataset" "$std_complete" "$lmpo2_complete" "$both"
done 2>/dev/null || echo "No NTN datasets found"
