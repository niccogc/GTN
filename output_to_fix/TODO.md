**READ THE ERROR FILES IN THIS FOLDER**

These errors are inacceptable, to be fixed.

ALSO need a super small grid search test file to run all models with cuda to fastly run on the cluster and see if everything works.

the only cuda problem is MMPO2 typeI with gtn. everything else works. It is probably how the ensemble handle non trainable parameters.

**IMPORTANT**

The most inacceptable thing is that the script continued and logged the results!!

If a run grid search had a failure, you CANNOT tag it with complete. Now the runs get tagged with complete

Look

```bash

[1/6] Running: MPO2-seed42
  Epoch   1 | Train Loss: 3.7537 | Train Quality: 0.3429 | Val Quality: 0.3636 | Patience: 0
  Epoch   2 | Train Loss: 2.3269 | Train Quality: 0.4286 | Val Quality: 0.5000 | Patience: 0
  ✓ SUCCESS: Test Acc=0.5217

[2/6] Running: LMPO2-seed42
  Epoch   1 | Train Loss: 0.9750 | Train Quality: 0.6762 | Val Quality: 0.9091 | Patience: 0
  Epoch   2 | Train Loss: 0.5986 | Train Quality: 0.6952 | Val Quality: 0.9091 | Patience: 0
  ✓ SUCCESS: Test Acc=0.6522

[3/6] Running: MMPO2-seed42
  Epoch   1 | Train Loss: 1.7527 | Train Quality: 0.5143 | Val Quality: 0.5455 | Patience: 0
  Epoch   2 | Train Loss: 0.8399 | Train Quality: 0.8381 | Val Quality: 0.9545 | Patience: 0
  ✓ SUCCESS: Test Acc=0.7391

[4/6] Running: MPO2TypeI_GTN-seed42
  Epoch   1 | Train Loss: 1.0677 | Train Quality: 0.8095 | Val Quality: 0.9091 | Patience: 0
  Epoch   2 | Train Loss: 0.9397 | Train Quality: 0.8190 | Val Quality: 0.9091 | Patience: 0
  ✓ SUCCESS: Test Acc=0.7391

[5/6] Running: LMPO2TypeI_GTN-seed42
  Epoch   1 | Train Loss: 1.0972 | Train Quality: 0.6952 | Val Quality: 0.8636 | Patience: 0
  Epoch   2 | Train Loss: 1.0862 | Train Quality: 0.6667 | Val Quality: 0.8636 | Patience: 0
  ✓ SUCCESS: Test Acc=0.6087

[6/6] Running: MMPO2TypeI_GTN-seed42
  ✗ FAILED: Expected all tensors to be on the same device, but got mat2 is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_mm)

======================================================================
GRID SEARCH SUMMARY
======================================================================
Total runs: 6
Completed: 5
Skipped: 0
Failed: 1
Time: 2.7s (0.5s per run)

Top 5 Configurations by Test Accuracy:

1. MMPO2-seed42
   Test Accuracy: 73.91
   Model: MMPO2, Seed: 42
   Key params: bond_dim=3, lr=0.01, wd=0

2. MPO2TypeI_GTN-seed42
   Test Accuracy: 73.91
   Model: MPO2TypeI_GTN, Seed: 42
   Key params: bond_dim=3, lr=0.01, wd=0

3. LMPO2-seed42
   Test Accuracy: 65.22
   Model: LMPO2, Seed: 42
   Key params: bond_dim=3, lr=0.01, wd=0

4. LMPO2TypeI_GTN-seed42
   Test Accuracy: 60.87
   Model: LMPO2TypeI_GTN, Seed: 42
   Key params: bond_dim=3, lr=0.01, wd=0

5. MPO2-seed42
   Test Accuracy: 52.17
   Model: MPO2, Seed: 42
   Key params: bond_dim=3, lr=0.01, wd=0

Summary saved to: cudatest/summary.json
Grid search complete. Marked as .complete
===========================================
```

**OUTPUT INDICES**

For all models the default output index label should be on the LAST L-1 block!! See if that what has been used in all grid search experiments.
