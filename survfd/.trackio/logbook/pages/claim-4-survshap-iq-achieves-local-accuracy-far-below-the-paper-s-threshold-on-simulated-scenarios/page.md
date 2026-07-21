# Claim 4: SurvSHAP-IQ achieves local accuracy far below the paper's threshold on simulated scenarios


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3191916f17f5", "created_at": "2026-07-20T14:27:45+00:00", "title": "Claim 4: local accuracy (Fig. 2 scenarios)"}
-->
**Setup.** The paper's Fig. 2 ten-scenario simulation design (Sec. 4.1, verbatim): n=1000, x1,x2,x3 ~ N(0,1), constant baseline hazard lambda=0.03, betas b1=0.4,b2=-0.8,b3=-0.6,b12=-0.5,b13=0.2, ten risk-score scenarios spanning additive/nonlinear main effects x time-independent/time-dependent x with/without interaction. Local accuracy (Eq. B35/B36): sigma(t) = sqrt(E[(F(t|x)-E[F]-Phi(t))^2] / E[F(t|x)^2]), averaged over t as sigma-bar, using the EXACT order-2 SurvSHAP-IQ decomposition (ground-truth F, not a fitted model). Reproducibility finding: applying the paper's Eq. (10) literally at every order (the natural reading of the paper text alone) does NOT satisfy local accuracy on the six scenarios with a real x1-x3 or x1-x2 interaction -- the correct order-1/order-2 split has to be derived from the Mobius transform in Appendix A.6 (phi_i = m(i) - m(full)/6; phi_ij = m(ij) + m(full)/2 for p=3). Code `audit_a4_localacc.py` (setup/scenario transcription) and `audit_a4_survival.py` (corrected Mobius construction, generalized here to the log-hazard scale as well as survival).

Reduced from the paper for CPU runtime: 5 timepoints on (0,70] instead of a dense grid, 97-point trapezoid integration grid for the survival-scale target, 3 seeds. Grid coarseness does not affect sigma-bar because local accuracy is an algebraic identity of the decomposition, not a numerical-integration accuracy question.

| target scale | paper threshold | measured sigma-bar (max over 10 scenarios x 3 seeds) | pass |
| --- | --- | --- | --- |
| log-hazard (ground truth) | < 1e-5 | ~1e-16 to 1e-17 | **yes, ~1e8-1e9x margin** |
| survival (ground truth) | < 1e-3 | ~1e-16 to 1e-17 | **yes, ~1e13x margin** |
| survival (fitted CoxPH/GBSA) | < 0.015 | **not attempted** — no sklearn/scikit-survival available | n/a |

**Verdict — Claim (anchored A4) confirmed at machine precision.** With the corrected order-1/order-2 construction, local accuracy holds to within floating-point error on every one of the ten scenarios and across seeds, on both the log-hazard and survival scales, many orders of magnitude inside the paper's own thresholds. This is the expected result for an exact decomposition of a known ground-truth function — local accuracy is an algebraic identity here, not an empirical claim about model fit — but it confirms the construction is implemented correctly, which the naive literal-Eq.-10 version was not. The one part of anchored A4 not covered: the paper's <0.015 threshold for **predicted** survival from a fitted CoxPH/GBSA model was not reproduced, since scikit-survival is not available in this environment; that part of the claim remains untested here.
