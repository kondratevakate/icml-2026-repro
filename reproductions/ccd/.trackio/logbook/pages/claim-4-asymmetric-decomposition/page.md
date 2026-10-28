# Claim 4: Asymmetric decomposition


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ccd_v2_c4", "created_at": "2026-07-29T09:00:00+00:00", "title": "Claim 4: Asymmetric decomposition"}
-->
**Anchored claim.** The ERT metrics decompose conditional-coverage error into
separate over-coverage and under-coverage components.

**Verdict: VERIFIED.**

The same independent 30,000-point synthetic run evaluates the clipped convex
functions on either side of target coverage `0.9`.

| Component | Estimated | Analytic |
| --- | ---: | ---: |
| Over-coverage | `0.0445` | `0.0460` |
| Under-coverage | `0.0464` | `0.0488` |

Both components are nonzero and close to their analytic values. This matches
the mechanism of constant-width conformal sets on heteroscedastic data:
low-noise cases are over-covered and high-noise cases are under-covered.

```powershell
cd prior_synthetic
python verify_ert.py
```
