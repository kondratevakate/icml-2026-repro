# Claim 1: ERT metrics with modern classifiers have higher statistical power than simple classifiers behind metrics like CovGap


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1adfed6aef97", "created_at": "2026-07-19T08:18:56+00:00", "title": "Claim 1: ERT metrics with modern classifiers have higher statistical power than simple classifiers behind metrics like CovGap"}
-->
**Setup (the paper's own synthetic model).** `X ~ U([-1,1]^8)`, `Y ~ N(0, sigma(X_1))` with `sigma(x) = 0.5 + |x| + x^2`. Prediction sets: **standard split conformal** with nonconformity `S(X,Y)=|Y|` calibrated on 3,000 samples, giving a **constant-width** set `[-q, q]` (marginally valid, conditionally invalid). Because the model is Gaussian the true conditional coverage is **analytic**: `p(X) = 2*Phi(q/sigma(X_1)) - 1`, so every metric is compared against exact ground truth (theoretical L1 = **0.0948**, from 300k samples as in the paper).

**ERT (Theorem 3.1).** For convex `f` with `f(1-a)=0`, `f'(1-a)=0`, the proper loss `l(p,y) = -f(p) - (y-p) f'(p)` satisfies `l-ERT = E[f(p(X))]`. Since `l(1-a,y)=0`, the estimate is `mean[f(h(x)) + (z-h(x)) f'(h(x))]`, computed with **2-fold cross-fitting**. `f_L1(p)=|p-(1-a)|`, `f_L2(p)=(p-(1-a))^2`. Code `verify_ert.py`, alpha=0.1, 5 repetitions.

| n test | L1-ERT (learned classifier) | CovGap, **oracle** groups (bins on X1) | CovGap, **unaligned** groups (bins on X2) |
| --- | --- | --- | --- |
| 1 000 | 0.0210 | 0.0952 | 0.0232 |
| 3 000 | 0.0677 | 0.0930 | 0.0146 |
| 10 000 | 0.0686 | 0.0961 | 0.0076 |
| 30 000 | **0.0866** | 0.0953 | **0.0049** |

**Verdict — Claim 1 reproduced.** In the realistic regime, where the practitioner does not know which covariate drives miscoverage, the group metric is **blind**: CovGap on unaligned groups collapses toward 0 (0.0049 at n=30k, i.e. 5% of the true 0.0948) because it converges to the marginal coverage inside each group. The learned-classifier ERT **finds the structure without being told**, recovering 0.0866 (92% of truth). That is the statistical-power gap the paper claims.

**Precise boundary of the claim (honest).** If the groups happen to be aligned with the violation (here: binning on the very covariate X1 that drives it), CovGap is accurate (0.0953 vs 0.0948) and beats the learned ERT at small n. So the claim is about *unknown/unaligned* grouping, not a blanket superiority. Our first attempt used oracle bins and therefore did *not* reproduce the claim — the fix was to make the baseline realistic.

**Bonus: the lower-bound property is visible.** ERT is a lower bound on the true value for any classifier, and indeed it converges **from below**: 0.021 -> 0.068 -> 0.069 -> 0.087 toward 0.0948 as the classifier gets more data. Sanity check: on **oracle conditionally-valid** sets ERT returns 0.0014 (~0), correctly reporting no violation.

**Documented substitution:** the paper uses fast tabular classifiers (LightGBM-class); this environment has no sklearn/LightGBM, so the learned classifier is a small torch MLP and the 'simple classifier' is the piecewise-constant predictor underlying CovGap. The classifier family is not the paper's contribution.
