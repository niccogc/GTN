#!/bin/bash
# Sanity check: Run all models on small configs to verify everything works
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=============================================="
echo "  GTN/NTN Sanity Check - All Models"
echo "=============================================="

# Clean previous results
rm -rf sanity_check/results/

echo ""
echo "[1/4] NTN + Concrete (regression)"
python run_grid_search.py --config sanity_check/test_ntn_concrete.json

echo ""
echo "[2/4] NTN + Iris (classification)"
python run_grid_search.py --config sanity_check/test_ntn_iris.json

echo ""
echo "[3/4] GTN + Concrete (regression)"
python run_grid_search_gtn.py --config sanity_check/test_gtn_concrete.json

echo ""
echo "[4/4] GTN + Iris (classification)"
python run_grid_search_gtn.py --config sanity_check/test_gtn_iris.json

echo ""
echo "=============================================="
echo "  Sanity Check Complete!"
echo "=============================================="
echo "Results in: experiments/sanity_check/results/"
