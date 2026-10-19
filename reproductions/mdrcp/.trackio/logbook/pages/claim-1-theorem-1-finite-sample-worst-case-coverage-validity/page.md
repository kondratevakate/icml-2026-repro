# Claim 1: Theorem 1 finite-sample worst-case coverage validity


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bc386addc097", "created_at": "2026-07-20T08:32:00+00:00", "title": "Claim 1: Theorem 1 finite-sample worst-case coverage validity"}
-->
**Claim (Theorem 1).** MDCP's prediction sets achieve marginal coverage at least 1-alpha under *any* mixture of the K source distributions (finite-sample worst-case validity), i.e. worst-case coverage across sources >= 1-alpha = 0.90.

**Setup.** Paper's own synthetic Linear DGP (Sec. 5.1-5.3, transcribed verbatim in `spec.md`): K=3 sources, d=10, alpha=0.1, n_k=2000 per source, split train 37.5% / calib 12.5% / test 50%, tau=2.5. Classification: C=6 classes, multinomial logit DGP (Sec. 5.2). Regression: linear-Gaussian DGP with per-source SNR in [5,10] (Sec. 5.3). Conformal p-value per Eq. 7 (standard orientation, see the Eq. 7 note on the Conclusion / Claim 2 page). MDCP's shared score uses lambda_k(x) fit by maximizing the dual objective (Eq. 13) with a covariate-dependent cubic B-spline lambda, full-batch Adam (`fit_lambda_exact` / `fit_lambda_reg` in `verify_mdrcp.py`). 10 seeds. **Deviation** (forced by no scipy/sklearn in this environment): phat_k / mu_k / sigma_k are fit with correctly-specified parametric models (multinomial logistic regression; ridge mean + log-residual-squared variance) instead of the paper's gradient-boosted trees — appropriate since the DGP is Linear, and all methods (MDCP, max-p baseline, single-source baseline) share the same fits, so the coverage comparison stays apples-to-apples.

**Results (10 seeds, mean / worst-case coverage across the 3 sources):**

| method | classification avg cov | classification worst cov | regression avg cov | regression worst cov |
| --- | --- | --- | --- | --- |
| **MDCP** | 0.921 | **0.905** | 0.924 | **0.909** |
| max-p baseline (AGG) | 0.956 | 0.944 | 0.952 | 0.939 |
| single-source (SRC) | 0.908 | 0.894 | 0.564 | **0.327** |

Source: `res_clf.json` (`mdcp_worst=0.9051`, `agg_worst=0.9440`, `src_worst=0.8943`), `res_reg.json` (`mdcp_worst=0.9089`, `agg_worst=0.9385`, `src_worst=0.3266`).

**Verdict — CONFIRMED.** MDCP's worst-case coverage clears the nominal 0.90 target in both settings (0.905 / 0.909), matching the paper's own reported 90.25%. The max-p baseline over-covers as expected for a naive union aggregation (0.944 / 0.939), and single-source calibration collapses in the regression setting (worst-case coverage 0.327, vs. the paper's own claim of a collapse under source heterogeneity) — both corroborating side-observations the paper makes about the baselines. This result is robust to lambda quality: every lambda tried in this reproduction (the fitted spline lambda used here, and the oracle constant-lambda used for the Claim 2 follow-up) gave valid worst-case coverage, so Claim 1 does not depend on the harder-to-verify size-reduction magnitudes checked in Claim 2.
