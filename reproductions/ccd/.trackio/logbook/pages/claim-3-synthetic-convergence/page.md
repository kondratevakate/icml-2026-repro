# Claim 3: Synthetic convergence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ccd_v2_c3", "created_at": "2026-07-29T09:00:00+00:00", "title": "Claim 3: Synthetic convergence"}
-->
**Anchored claim.** On the paper's synthetic experiment, group-based metrics
such as CovGap remain unreliable when their groups are not aligned with the
miscoverage, while learned ERT metrics approach their theoretical values.

**Verdict: VERIFIED.**

The independent experiment uses the paper's model:
`X ~ U([-1,1]^8)` and `Y ~ N(0, sigma(X_1))`, where
`sigma(x)=0.5+|x|+x^2`. Standard split conformal uses 3,000 calibration
samples. Gaussianity gives analytic conditional coverage, with theoretical
L1 miscoverage `0.0948`.

| Test size | L1-ERT | CovGap, oracle groups | CovGap, unaligned groups |
| ---: | ---: | ---: | ---: |
| 1,000 | `0.0210` | `0.0952` | `0.0232` |
| 3,000 | `0.0677` | `0.0930` | `0.0146` |
| 10,000 | `0.0686` | `0.0961` | `0.0076` |
| 30,000 | `0.0866` | `0.0953` | `0.0049` |

At 30,000 test points, learned L1-ERT recovers 92% of the analytic violation;
the unaligned group metric recovers 5%. The oracle-group result bounds the
claim: CovGap works when the responsible covariate is known in advance.

```powershell
cd prior_synthetic
python verify_ert.py
```
