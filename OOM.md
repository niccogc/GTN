All working. Summary:
---
Done. Enhanced check_progress.py with OOM analysis:
New CLI options:
- --oom → Deep analysis: breakdowns by model, dataset, bond_dim, L, feature count, GPU memory
- --oom-detail → Same + all OOM run paths
- --oom-json → Raw JSON for scripting
Key findings from your data:
- 1447 OOM runs total
- LMPO2TypeI worst offender (670 OOMs, 46%)
- Large datasets dominate (72% of OOMs)
- bank/mushrooms/student_dropout + LMPO2TypeI = top combos (90 each)
- L=3 vs L=4 ~equal (memory scaling not main issue)
- bond_dim spread: 4=20%, 8=33%, 12=33%
- LMPO2 rf=0.9 worst (338), rf=0.3 best (225)
- V100-16GB maxes out ~14.6GB avg before OOM
