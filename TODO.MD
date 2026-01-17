# TODO Items - Status Update

## Completed

### 1. GTN Configs - n_epochs fixed
- GTN configs now use 200 epochs (was 10)
- Added patience: 20 for early stopping

### 2. LMPO2 reduction_factor merged into main configs
- No longer separate config files for LMPO2 reduction factor
- `reduction_factor: [0.1, 0.3, 0.5]` is now in the main parameter grid
- Only applied to LMPO2/LMPO2TypeI models (ignored by other models)
- Reduced from 84 config files to 42 (21 NTN + 21 GTN)

### 3. AIM tracking - runs registered immediately
- Added `log_system_params=True` for better tracking
- Run hash is printed when tracker starts (not just at end)
- Runs should now appear in AIM UI immediately

### 4. "Already done" check fixed
- Bug: `result.get("success", False) is not None` returned True even for failed runs
- Fix: Changed to `result.get("success", False) is True`
- Now only successfully completed runs are skipped

### 5. Initialization
- `init_strength` set to 1.0 (normalized random initialization)
- Normalization functions in `model/initialization.py` now support:
  - Single TN or list of TNs (for TypeI models)
  - `inplace=True/False` parameter

## Notes

### Config regeneration
Run this to regenerate all configs:
```bash
cd experiments/configs
python generate_uci_configs.py
```

### Clearing results cache
To re-run failed experiments, delete the result files:
```bash
rm -rf results/<experiment_name>/*.json
```
Or just delete the specific failed run files (those with `"success": false`).
