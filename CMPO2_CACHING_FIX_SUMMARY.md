# CMPO2 Caching Fix Summary

## Date
December 30, 2025

## Problem

The `CMPO2_NTN._batch_environment()` caching implementation in `model/MPS.py` (lines 37-120) was failing because it incorrectly combined the base environment with local context, creating duplicate tensors.

### Root Cause

The code misunderstood what `env_obj()` returns:

**WRONG ASSUMPTION** (in comments, line 82):
> "The base_env excludes the whole column (Pixel + Patch + Input)."

**ACTUAL BEHAVIOR**:
`env_obj()` returns `_LEFT + _RIGHT + SAME_SITE`, where:
- `_LEFT`: Contracted environment from sites 0 to i-1
- `_RIGHT`: Contracted environment from sites i+1 to L-1
- `SAME_SITE`: **Uncontracted tensors at site i** (MPS tensors + inputs)

### The Bug

```python
# OLD CODE (lines 78-95)
base_env = env_obj()  # Has: _LEFT, _RIGHT, SAME_SITE (0_Pi, 0_Pa, INPUT_0)

# Get tensors at this site
full_tn_at_site = env_obj.tn.select(site_tags)  # {0_Pi, 0_Pa, INPUT_0}
local_context = full_tn_at_site.copy()
local_context.delete(target_tag)  # {0_Pa, INPUT_0}

# BUG: Combining creates DUPLICATES!
final_env = base_env | local_context  
# Now has: _LEFT, _RIGHT, 0_Pi, 0_Pa (from base_env), 0_Pa (DUPLICATE!), INPUT_0 (DUPLICATE!)
```

**Result**: 
- Duplicate tensors create new intermediate bond indices
- `outer_inds()` no longer includes bonds to the target
- Environment is incorrectly contracted

### Debug Output (Before Fix)

```
Final environment (base_env | local_context):
  Num tensors: 7  ← Should be 4!
  Outer indices: ('out',)  ← Lost all bonds!
  
Bonds to target 0_Pi:
  Outer inds: {'out'}
  Target inds: {'_xxx', '0_pixels'}
  Overlap (bonds): set()  ← NO BONDS!
  
✗✗✗ PROBLEM: outer_inds doesn't include any bonds to target!
```

## Solution

Simply delete the target from `base_env` directly:

```python
# NEW CODE (lines 78-86)
base_env = env_obj()  # Has: _LEFT, _RIGHT, SAME_SITE (all tensors at site)

# Create hole by deleting only the target
final_env = base_env.copy()
final_env.delete(target_tag)  # Simple and correct!
```

### Debug Output (After Fix)

```
Final environment after deleting 0_Pi:
  Num tensors: 4  ← Correct!
  Outer indices: ('out', '_xxx', '0_pixels')  ← All bonds present!
  
Bonds to target 0_Pi:
  Outer inds: {'_xxx', 'out', '0_pixels'}
  Target inds: {'_xxx', '0_pixels'}
  Overlap (bonds): {'_xxx', '0_pixels'}  ← 2 bonds found!
  
✓ Good! Found 2 bond(s) to target
```

## Changes Made

### File: `model/MPS.py`

**Lines 78-95** (old):
```python
# 3. Get Base Environment (Excludes ALL tensors at site_idx, i.e., excludes I{site_idx})
base_env = env_obj() 

# 4. Reconstruct Local Context
# The base_env excludes the whole column (Pixel + Patch + Input).
# We need to ADD back the parts of the column that are NOT the target_tag.

# Select all tensors at this site (including inputs, pixels, patches)
site_tags = env_obj.site_tag(site_idx) # e.g. "I0"
full_tn_at_site = env_obj.tn.select(site_tags)
# Identify the "other" tensors at this site (Context = Column - Target)
# We copy to avoid modifying the original TN
local_context = full_tn_at_site.copy()
local_context.delete(target_tag) # Remove ONLY the target (e.g. 0_Pi)

# 5. Form the Final Environment
# Env = Base_Env (Left/Right) + Local_Context (Other parts of column)
final_env = base_env | local_context
```

**Lines 78-86** (new):
```python
# 3. Get Base Environment
# env_obj() returns: _LEFT + _RIGHT + SAME_SITE tensors
# SAME_SITE includes all tensors at this site (MPS tensors + inputs)
base_env = env_obj() 

# 4. Create hole by deleting only the target
# We copy to avoid modifying the cached environment
final_env = base_env.copy()
final_env.delete(target_tag)
```

## Test Results

### test_cmpo2_basic.py

✅ **PASSING**

```
Starting Fit: 1 epochs (cached sweeping).
Init    | mse: 1.09277 | R2: -0.21998
...
Epoch 1 | mse: 1.11702 | R2: -0.24706

✓ Training successful!
```

All environments have correct structure:
- Batch dimension 's' preserved
- Output dimension 'out' preserved
- Correct bonds to targets (MPS bonds + physical indices)

### test_cmpo2_caching_debug.py

✅ **PASSING**

```
Bonds to target 0_Pi:
  Outer inds: {'_1b7c7fAAAAB', 'out', '0_pixels'}
  Target inds: {'_1b7c7fAAAAB', '0_pixels'}
  Overlap (bonds): {'_1b7c7fAAAAB', '0_pixels'}

✓ Good! Found 2 bond(s) to target
```

## Key Insight

**BatchMovingEnvironment** follows the standard DMRG pattern:
- `env()` returns **LEFT + RIGHT + SAME_SITE**
- **SAME_SITE** tensors are NOT pre-contracted
- To create a hole, simply **delete the target** from this environment
- No need to reconstruct local context separately

This matches the user guide at `BATCHMOVINGENVIRONMENT_USER_GUIDE.md`:

```python
# Move to target site
env.move_to(site_idx)

# Get environment (includes SAME_SITE tensors)
current_env = env()

# Create hole by deleting target
env_with_hole = current_env.copy()
env_with_hole.delete(target_tag)
```

## Related Files

- `model/MPS.py`: Fixed CMPO2_NTN._batch_environment()
- `model/batch_moving_environment.py`: BatchMovingEnvironment implementation
- `test_cmpo2_basic.py`: Basic CMPO2 test (now passing)
- `test_cmpo2_caching_debug.py`: Debug script for environment structure
- `BATCHMOVINGENVIRONMENT_USER_GUIDE.md`: Documentation on how to use BatchMovingEnvironment

## Previous Related Fixes

1. **BatchMovingEnvironment bond preservation** (`BATCHMOVINGENVIRONMENT_FIX_SUMMARY.md`):
   - Fixed `_get_inds_to_keep()` to preserve MPS bonds
   - Added `site_tag` parameter to keep bonds to current site

2. **Environment caching** (`ENVIRONMENT_CACHING_SUMMARY.md`):
   - Implemented caching for CMPO2_NTN
   - This fix completes the caching implementation

## Status

✅ **COMPLETE** - CMPO2_NTN caching now works correctly!
