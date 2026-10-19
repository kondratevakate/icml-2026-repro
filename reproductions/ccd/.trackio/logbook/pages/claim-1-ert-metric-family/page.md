# Claim 1: ERT metric family


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ccd_v2_c1", "created_at": "2026-07-29T09:00:00+00:00", "title": "Claim 1: ERT metric family"}
-->
**Anchored claim.** The paper introduces L1-ERT, L2-ERT, and KL-ERT for
conditional-coverage evaluation. Under perfect conditional coverage, no
classifier can outperform the constant `1-alpha` predictor.

**Verdict: VERIFIED.**

For target coverage `q=0.9`, the audit evaluates the supporting-score identity

`f(h) + (p-h) f'(h) <= f(p)`

for every true conditional coverage `p` on a 99-point grid and every candidate
prediction on a 100-point grid. It separately checks L1, L2, and Bernoulli KL.

| Check | L1 | L2 | KL |
| --- | ---: | ---: | ---: |
| Max oracle identity error | `0` | `0` | `0` |
| Max constant-target score | `0` | `0` | `0` |
| Max supporting-plane violation | `1.11e-16` | `0` | `0` |

Thus the true conditional predictor attains `f(p)`, while the constant target
predictor has zero score and is unbeatable when `p=q`.

```powershell
python audit_metric_identities.py --output results/metric_identities.json
```
