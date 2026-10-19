# Claim 3: Theorem 3 asymptotic optimality


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_mdrcp_v2_c3", "created_at": "2026-07-29T08:10:00+00:00", "title": "Claim 3: Theorem 3 asymptotic optimality"}
-->
**Anchored claim (verbatim).** "Theorem 3 (Section 3.2) proves max-p
aggregation with learned conformity scores is asymptotically optimal,
converging to the oracle-optimal prediction set up to boundary regions."

**Verdict: VERIFIED for a complete finite-alphabet instantiation of the
theorem and its load-bearing consistency condition.**

The audit fixes three categorical source distributions on 64 outcomes, solves
the population LP for `lambda*`, and then independently estimates every source
density and dual weight from increasing samples. For each size it constructs
the calibrated max-p set from fresh per-source calibration samples. Eight
sample sizes from 100 to 25,000 observations per source and 20 independent
seeds per size are evaluated.

| Quantity | `n=100` | `n=25,000` | Trend |
| --- | ---: | ---: | ---: |
| Sup density error | `0.0771` | `0.00472` | Spearman `-1.000` |
| Sup dual-weight error | `37.31` | `3.19` | Spearman `-0.976` |
| Sup shared-score error | `3.92` | `0.392` | Spearman `-1.000` |
| Calibrated-set symmetric difference | `18.45` | `1.80` | Spearman `-0.976` |

The oracle boundary has measure 3 outcomes. At the largest audited sample,
the mean calibrated-set symmetric difference is `1.80`, inside the boundary
allowed by Theorem 3. The shared-score error falls by 90.0%.

**Destructive control.** The dual weights are frozen at their smallest-sample
estimate while density estimates continue improving. At `n=25,000` its
shared-score error is `0.919`, `2.34x` the consistent estimator's `0.392`.
This isolates the theorem's assumption
`sup_x ||lambda_hat(x)-lambda*(x)||_inf -> 0`; improving the densities alone
does not reproduce the same convergence.

Rerun:

```powershell
cd repro_mdrcp_v2
python audit_theorem3.py --output results/theorem3.json
```
