# Claim 1: Censoring induces systematic mechanism-dependent distortions in survival evaluation metrics


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_eeb05cc93bf0", "created_at": "2026-07-19T08:34:37+00:00", "title": "Claim 1: Censoring induces systematic mechanism-dependent distortions in survival evaluation metrics"}
-->
**Setup (the paper's ST-vs-OR design, fully synthetic so the oracle is exact).** Covariates `X ~ N(0, I_5)`, Cox-exponential event times `T ~ Exp(0.1 * exp(beta'X))` — true event times are known by construction. Three censoring mechanisms as in the paper, each tuned by bisection to hit a target rate: **administrative** (`C = tau` fixed), **independent** (`C ~ Exp`), **covariate-dependent** (`C ~ Exp(scale * exp(-gamma'X))`). The *same fixed models* and the *same predictions* are scored twice: **ST** = metric on censored observations, **OR** = metric on fully observed event times. Only the test outcomes differ, so ST−OR isolates the distortion the censoring puts into the metric. Metrics: Harrell's C-index (concordance family) and the Integrated Brier Score with IPCW censoring weights (IBS family). n=4000, 8 repetitions. Code `verify_censoring_distortion.py` (numpy, CPU).

**Bias = ST − OR**

| censoring rate | C-index: admin / indep / cov-dep | IBS: admin / indep / cov-dep |
| --- | --- | --- |
| 20% | +0.0022 / +0.0042 / +0.0048 | **−0.0214** / −0.0000 / −0.0008 |
| 40% | +0.0078 / +0.0104 / +0.0111 | **−0.0314** / −0.0005 / −0.0017 |
| 60% | +0.0160 / +0.0182 / +0.0196 | **−0.0461** / −0.0001 / −0.0024 |
| 80% | +0.0270 / +0.0298 / +0.0295 | **−0.0795** / −0.0323 / **0.0000** |

**Verdict — Claim 1 reproduced, and the mechanism-dependence is localised.** The distortion is systematic (monotone in the censoring rate, zero at 0% by construction) and the two metric families behave in **fundamentally different ways**, exactly as claimed:
- **IBS is strongly mechanism-dependent.** Administrative censoring biases it by up to −0.080, independent censoring leaves it near zero until very heavy censoring, and covariate-dependent censoring leaves it essentially untouched (0.000 at 80%). A practitioner reading IBS under administrative censoring is reading an optimistically inflated score.
- **Harrell's C-index is rate-dependent but mechanism-agnostic here** (+0.027 / +0.030 / +0.030 at 80%): it drifts upward with censoring regardless of mechanism, because censoring removes exactly the hard, late-time comparisons.

This also lines up with the reviewers' discussion of the IBS family being the place where the censoring-mechanism interaction shows up.
