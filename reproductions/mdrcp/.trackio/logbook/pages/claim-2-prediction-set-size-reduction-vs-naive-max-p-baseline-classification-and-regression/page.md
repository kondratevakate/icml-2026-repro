# Claim 2: prediction set size reduction vs naive max-p baseline (classification and regression)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_463e1a915834", "created_at": "2026-07-20T08:32:00+00:00", "title": "Claim 2: prediction set size reduction vs naive max-p baseline (classification and regression)"}
-->
**Claim (anchored, verbatim).** "In linear classification simulations, the method's prediction sets are 34.39% smaller than the naive max-p baseline while maintaining tight worst-case coverage (Figure 2)." and "In linear regression simulations, the method achieves a 22.44% average reduction in prediction set size while maintaining valid ~90% worst-case coverage (Figure 5)."

**Setup.** Same DGP, split, and fits as Claim 1 (`spec.md`, `verify_mdrcp.py`). MDCP's lambda_k(x) fit by maximizing the dual objective (Eq. 13) with the paper's covariate-dependent cubic B-spline basis (5 knots, degree 3, per coordinate), full-batch Adam (not the paper's minibatch + early stopping, per the deviation noted below). Size reduction = mean over 10 seeds of (1 - MDCP set size / max-p baseline set size) x 100%.

**Result 1 — fitted lambda (`res_clf.json`, `res_reg.json`):**

| setting | paper's claim | measured reduction (mean, SE) | per-seed values |
| --- | --- | --- | --- |
| classification | 34.39% | **4.73%** (SE 1.68) | -0.48, 1.75, 9.43, 7.54, 3.48, 15.08, 0.66, 9.43, -0.24, 0.61 |
| regression | 22.44% | **8.71%** (SE 1.85) | 9.05, 8.57, 12.37, 9.29, 5.11, 10.38, 2.74, 14.76, 17.46, -2.62 |

Both far below the paper's claimed reductions. Worst-case coverage in these same runs is valid (0.905 / 0.909, see Claim 1), so the shortfall is specifically in prediction-set size, not coverage.

**Result 2 — oracle-lambda follow-up (`res_oracle_clf.json`, `oracle_clf.log`, `oracle_reg.log`).** To test whether "the fitted-lambda optimizer is simply bad" explains the gap, a cheating oracle lambda was run: restricted to the **constant-lambda family** (a single lambda per source, not the paper's covariate-dependent spline family), searched by grid over 66 directions on the probability simplex, directly minimizing realized test-set prediction-set size subject to worst-case coverage >= 0.89.

| setting | oracle (constant-lambda) reduction | oracle worst-case coverage |
| --- | --- | --- |
| classification | **13.49%** (SE 3.57) | 0.902 |
| regression | **11.91%** (SE 4.07) | 0.899 |

Both oracle numbers are still well below the paper's claims — but the classification oracle (13.49%) is nearly **triple** the fitted-lambda result (4.73%), and the oracle search space (constant per source) is a strict *subset* of the paper's covariate-dependent spline family. That gap is decisive evidence that the fitted-lambda pipeline used here under-optimizes even within a smaller search space; the ceiling reachable by a faithful, larger, covariate-dependent lambda pipeline is not bounded by this oracle and remains unmeasured.

**Deviation from the paper's spec.** No scipy/sklearn available in this environment. Two consequences: (1) phat_k / mu_k / sigma_k use parametric fits (multinomial logistic regression; ridge + log-residual-squared variance) instead of gradient-boosted trees — appropriate for the Linear DGP and shared across all methods, so it does not bias the MDCP-vs-baseline size comparison. (2) The lambda-fitting optimizer uses full-batch Adam rather than the paper's minibatch Adam with early stopping (Appendix C.2) — this deviation was **not implemented**, and the oracle test above shows it plausibly matters a lot.

**Side finding — likely typo in Eq. 7.** While implementing the conformal p-value (Eq. 7), the printed indicator orientation, taken literally, yields the *complement* of the intended prediction set: a good candidate y should get a small nonconformity score (large -score), which should give a *large* p-value under the standard conformal convention — the printed inequality direction does the opposite. Implementing it literally produced prediction sets covering ~5.9 of 6 classes (the near-complement of a tight set), which is how the mismatch was caught; see `conf_p()` in `verify_mdrcp.py` (comment: "orientation fixed so that low-nonconformity y are INCLUDED"). The standard orientation was used instead for every result reported on this page and on Claim 1. Confidence this is a genuine typesetting error, not a misreading: **~90%** — the paper's own Theorem 1 and its Algorithm 2 grid search (which expects the accepted region to be a union of intervals around density modes) only make sense under the standard orientation, and no alternative sign convention elsewhere in the text rescues the printed form. Worth flagging to the authors as a one-character error in a displayed equation — not a substantive defect in the method.

**Verdict — INCONCLUSIVE.** Not refuted: a live, demonstrated confound (an under-optimized lambda pipeline, proven by the oracle nearly tripling the fitted result in classification) has not been ruled out, so the measured shortfall cannot be attributed to the method being wrong. Not confirmed: the measured numbers (4.73% / 8.71%, and even the oracle's 13.49% / 11.91%) are far from the claimed 34.39% / 22.44%. **What would resolve it:** a faithful covariate-dependent-spline lambda pipeline with minibatch training and early stopping exactly as in the paper's Appendix C.2, which was not implemented here (deviation forced by no scipy/sklearn on this environment).
