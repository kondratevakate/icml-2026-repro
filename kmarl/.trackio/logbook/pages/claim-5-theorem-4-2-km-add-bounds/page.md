# Claim 5: Claim A2: theorem 4.2 establishes finite-sample bias bounds for KM-ADD


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_kmadd_bounds_20260728", "created_at": "2026-07-28T00:00:00+00:00", "title": "Claim 5: Claim A2: theorem 4.2 establishes finite-sample bias bounds for KM-ADD"}
-->
**Anchored claim (verbatim).** "Theorem 4.2 establishes finite-sample bias bounds for the KM-ADD estimator under independent censoring."

**Verdict -- VERIFIED under Theorem 4.2's stated assumptions.**

**Derivation audit.** Appendix B.2 repeats the KM argument for nonnegative detection delays `Delta tau` and residual sequence lengths `C_ADD=T-nu`. The truncated-mean identity and the two Stute-inequality substitutions are algebraically identical to Theorem 4.1 after replacing `(F_ARL,G_ARL,H_ARL,a,N)` by `(F_ADD,G_ADD,H_ADD,b,N'')`. This yields exactly the lower and upper bounds in Eq. (9). I checked the substitution and signs independently rather than treating Theorem 4.2 as implied merely by the ARL experiment.

**Numerical audit.** A separate ADD context in `audit_finite_sample_bounds.py` uses event rate `0.5`, censoring rate `0.3`, truncation `b=5`, and **500,000 independent datasets for each `N''` in `{2,4,8,16,32,64}`**. All six `bias +/- 5 SE` intervals lie inside the independently integrated Eq. (9) bounds:

| N'' | lower bound | observed bias | upper bound | SE |
| ---: | ---: | ---: | ---: | ---: |
| 2 | -0.617901 | +0.435097 | +1.163602 | 0.002112 |
| 4 | -0.491311 | +0.205201 | +0.812713 | 0.001448 |
| 8 | -0.350203 | +0.083888 | +0.512294 | 0.000980 |
| 16 | -0.215015 | +0.028405 | +0.282663 | 0.000673 |
| 32 | -0.105709 | +0.006846 | +0.127416 | 0.000468 |
| 64 | -0.036045 | +0.001007 | +0.040665 | 0.000330 |

**Mechanism check.** The shared destructive control makes censoring deterministically depend on the event time. The bound then fails (`+6.832624` observed versus upper bound `+0.628612`), confirming that independent censoring is load-bearing.

**Evidence boundary.** As for Theorem 4.1, the external Stute (1994) lemma is not re-proved. The reproduction verifies the ADD-specific reduction, bound calculation, numerical behavior for one continuous exponential family, and failure when the key assumption is removed.
