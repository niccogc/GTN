# BatchMovingEnvironment Fix Summary

## ✅ **Bug Fixed: Missing MPS Bond Indices**

### Problem Identified

The `_get_inds_to_keep()` method in `BatchMovingEnvironment` was not preserving MPS bond indices when contracting LEFT and RIGHT environments.

**Root Cause:**
```python
# Old logic
active_inds = indices in tensors being contracted
passive_inds = all_inds - active_inds
bonds = active_inds ∩ passive_inds  # WRONG!
```

When contracting `(_RIGHT, site(i+bsz))`, the bond connecting `site(i)` to `site(i+1)` is:
- ✓ In `active_inds` (it's in MPS_{i+1} being contracted)
- ✗ NOT in `passive_inds` (because it's also in active!)
- ✗ Therefore NOT preserved!

### Fix Applied

Modified `_get_inds_to_keep()` to also preserve bonds to the current site:

```python
def _get_inds_to_keep(self, tn, active_tensors, site_tag=None):
    # ... existing logic ...
    
    # NEW: Keep bonds to current site
    if site_tag is not None:
        site_tensors = tn.select(site_tag)
        site_inds = set().union(*(t.inds for t in site_tensors))
        bonds_to_site = active_inds.intersection(site_inds)
        inds_to_keep.update(bonds_to_site)
    
    # ... rest of logic ...
```

Updated all call sites to pass `site_tag`:
- `init_segment()` line 83: pass `site_tag=self.site_tag(i)`
- `init_segment()` line 108: pass `site_tag=self.site_tag(i + self.bsz - 1)`
- `move_right()` line 139: pass `site_tag=self.site_tag(i)`
- `move_left()` line 165: pass `site_tag=self.site_tag(i)`

## Test Results

### ✅ All Basic Tests Pass

**verify_hole_calculation.py:**
```
SITE 0: Bond count: 2 (expected 2) ✓✓✓ CORRECT!
SITE 1: Bond count: 3 (expected 3) ✓✓✓ CORRECT!
SITE 2: Bond count: 3 (expected 3) ✓✓✓ CORRECT!
SITE 3: Bond count: 2 (expected 2) ✓✓✓ CORRECT!
```

**test_simple_mps_holes.py:**
```
All sites: ✓✓ CORRECT number of bonds
Batch dimension 's': ✓ preserved
Output dimension 'out': ✓ preserved (where present)
```

**test_env_components.py:**
```
LEFT TO RIGHT:
  Site 0: ✓ Has bond to MPS_0
  Site 1: ✓ Has bond to MPS_1 (both left and right)
  Site 2: ✓ Has bond to MPS_2 (both left and right)
  Site 3: ✓ Has bond to MPS_3

RIGHT TO LEFT:
  All sites: ✓ All bonds present
```

## What Was Fixed

1. **MPS Bond Preservation**: Bond indices connecting MPS sites are now correctly preserved during LEFT/RIGHT contractions

2. **Batch Dimension Preservation**: The batch index 's' is correctly preserved (was already working, now confirmed)

3. **Output Dimension Preservation**: Output dimensions are correctly preserved (was already working, now confirmed)

4. **Bidirectional Sweeps**: Both left-to-right and right-to-left sweeps work correctly

## Remaining Issues

The CMPO2 structure (two MPS with shared inputs) still has issues with hyper-edges, but this is a separate problem related to how the environment is used, not with the BatchMovingEnvironment itself.

## Files Modified

- `model/batch_moving_environment.py`: Added `site_tag` parameter to `_get_inds_to_keep()` and updated all call sites

## Files Created for Testing

- `verify_hole_calculation.py`: Verifies hole calculation at each site
- `test_simple_mps_holes.py`: Tests simple MPS with batch inputs and output dimension
- `test_env_components.py`: Tests LEFT, RIGHT, SAME_SITE components
- `debug_get_inds_to_keep.py`: Debug script showing the exact problem and solution

---

**Status**: ✅ **FIXED AND VERIFIED**

The BatchMovingEnvironment now correctly preserves MPS bond indices during contractions!
