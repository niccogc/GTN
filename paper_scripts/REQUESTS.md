1. Initial Script Creation
- Create paper_scripts/parse_test_results.py
- Parse test results from test_outputs/{ntn,gtn}/{MODEL}/{DATASET}/seed_*/results.json
- Find epoch with best validation quality (higher = better)
- Use test quality at that epoch
- Average over seeds
- Unify model variants: MPO2 + MPO2TypeI → MPO², pick best of the two
- Generate LaTeX tables (classification & regression, split)
- Separate tables for NTN and GTN
2. Combined Table Option
- Add --combined flag
- Single table with regression on left, classification on right
- Double vertical line || separator between sections
- Command: python paper_scripts/parse_test_results.py --combined
3. Dynamic Dataset Filtering
- Only show datasets that actually have test results
- Don't pre-populate all datasets - dynamically add based on availability
4. All Values as Percentages
- All metrics multiplied by 100 (including R² for regression)
- Both classification accuracy and regression R² shown as percentages
5. Use Validation Loss Option
- Add --use-val-loss flag
- Use val_loss (lower = better) instead of val_quality (higher = better) to select best epoch
- Uses >= for quality (last best) and <= for loss (last best)
6. Include All Results
- Remove success/singular checks in parse_results_json()
- Process all results regardless of success/error flags
7. Last Best Validation
- When validation is constant over interval, take last occurrence (use >= for quality, <= for loss)
----- TODO FROM HERE -----------
8. Add Mean Baseline
- DO NOT load all datasets for baseline, but only the one dynamically loaded from NTN or GTN trainer.
- Load baseline results from test_outputs/mean_baseline/{DATASET}/results.json
- Always show as "Mean" row in tables
- For classification: baseline already in percentage, don't multiply by 100
- For regression: multiply by 100
- Single row (no std row needed - no seeds)
9. Ring = BosonMPS
- "Ring" is BosonMPS model name in table
- Only show for GTN (doesn't exist for NTN)
- Remove BosonMPS from MODEL_ORDER, add "Ring" instead
10. Baseline Formatting
- Single row (no double row with std)
- Classification values for baselines: already in %, don't multiply by 100
- Regression values: multiply by 100
---
Final Command Options
# Separate tables (default)
python paper_scripts/parse_test_results.py
# Combined table (regression || classification)
python paper_scripts/parse_test_results.py --combined
# Use validation loss instead of quality
python paper_scripts/parse_test_results.py --use-val-loss
# GTN only
python paper_scripts/parse_test_results.py --trainer gtn
# All options
python paper_scripts/parse_test_results.py --combined --use-val-loss --trainer gtn
