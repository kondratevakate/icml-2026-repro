# Claim 6: Algorithm 1 cross-fitting


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ccd_v2_c6", "created_at": "2026-07-29T09:00:00+00:00", "title": "Claim 6: Algorithm 1 cross-fitting"}
-->
**Anchored claim.** Algorithm 1 uses k-fold cross-validation so that the
classifier used to estimate ERT is not evaluated on its own training labels.

**Verdict: VERIFIED.**

The mutation control uses 20 seeds, 2,000 observations, and five folds. Under
perfect conditional coverage, a label memorizer evaluated in sample creates a
large false L2-ERT signal. The same memorizer cannot recognize unique
held-out inputs and returns the target predictor. A separate two-region
classifier tests that cross-fitting still detects a genuine conditional
signal.

| Check | Result |
| --- | ---: |
| Maximum train/evaluation overlap | `0` |
| Mean in-sample null ERT | `0.08886` |
| Mean cross-fitted null ERT | `0` |
| Mean cross-fitted signal ERT | `0.01013` |

This isolates the load-bearing purpose of Algorithm 1: out-of-fold evaluation
removes an overfitting artifact without erasing a learnable held-out signal.

```powershell
python audit_algorithm1_crossfit.py --output results/algorithm1.json
```
