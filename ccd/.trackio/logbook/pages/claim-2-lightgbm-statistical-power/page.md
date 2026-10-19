# Claim 2: LightGBM statistical power


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ccd_v2_c2", "created_at": "2026-07-29T09:00:00+00:00", "title": "Claim 2: LightGBM statistical power"}
-->
**Anchored claim.** Table 2 reports substantially higher relative statistical
power for LightGBM L1-ERT than for the PartitionWise classifier underlying
CovGap.

**Verdict: VERIFIED BY RELEASED-OUTPUT REPLAY.**

This is a deterministic re-aggregation of the authors' official
`results.csv`, not a fresh classifier-training run. The CSV is pinned to
official commit `39a99dcad92205a15d93f2c5fec40c76540abf1c` and contains
5,600 rows: 8 datasets, 10 experiments, and 7 methods.

The audit independently implements the official notebook protocol: clip
negative ERT values to zero, normalize each metric by the best method for each
dataset/experiment, average across datasets, then average across experiments.

| Method | Relative L1 power |
| --- | ---: |
| CheapBetterLGBMClassifier | `68.929%` |
| PartitionWise | `33.671%` |
| Ratio | `2.047x` |

The values reproduce the current official notebook (`68.929005%` and
`33.670548%`) to better than `1e-5` percentage points.

```powershell
python audit_released_results.py
```
