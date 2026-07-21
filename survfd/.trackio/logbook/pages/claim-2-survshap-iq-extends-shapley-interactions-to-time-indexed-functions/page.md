# Claim 2: SurvSHAP-IQ extends Shapley interactions to time-indexed functions


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a1cd5400a0fe", "created_at": "2026-07-19T09:09:07+00:00", "title": "Claim 2: SurvSHAP-IQ extends Shapley interactions to time-indexed functions"}
-->
**Setup.** Time-dependent value function `v(t|M) = E[S(t|X)|X_M=x_M] - E[S(t|X)]` (Sec. 3.3), pairwise interaction index via the discrete-derivative n-Shapley formula (Eq. 10), computed by brute force over all 2^2 subsets (exact given p=2, common reference sample as in Claim 1). Evaluated at `t in {0.5, 2.0, 5.0}` for all four scenarios.

| scenario | interaction index at t=0.5/2/5 | mean\|.\| | ground-truth interaction |
| --- | --- | --- | --- |
| A (no interaction) | +0.009 / +0.030 / +0.040 | 0.026 | no |
| B (no interaction) | +0.004 / +0.033 / +0.055 | 0.031 | no |
| C (has interaction) | +0.015 / +0.054 / +0.085 | 0.051 | **yes** |
| D (has interaction) | +0.019 / +0.067 / +0.094 | 0.060 | **yes** |

**Mean magnitude: no-interaction scenarios 0.028 vs. interaction scenarios 0.056 -> 1.96x ratio.**

**The correct null here is not zero.** The value function lives on the **survival scale** `S(t|x) = exp(-h0 t exp(G(t|x)))`. Even when the log-hazard `G` is purely additive across `x1, x2`, the `exp(.)` link makes `S` multiplicative in the main effects, so the survival-scale interaction index is **necessarily non-zero** for a model that is additive in `G`. The "exact zero for no-interaction" reading is a mismatch between the theorem's additivity condition (stated on `G`, log-hazard scale) and where this diagnostic is measured (`S`, survival scale) — it was never the right target for scenarios A/B.

**Seed robustness (`audit_seed_fragility.py`, 12 seeds, `N_ref=60,000`).** The no-interaction baseline converges to a **non-zero constant, not to zero**: as `N_ref` grows from 2,000 to 2,000,000 the no-interaction magnitude stabilizes at **0.0284** (0.0288 -> 0.0285 -> 0.0284 -> 0.0284 -> 0.0284 across that range) while its Monte-Carlo sd shrinks roughly 40x, i.e. it is converging to a fixed non-zero value, not vanishing MC noise. The interaction/no-interaction ratio itself is tight and stable across seeds: **mean 1.9591, sd 0.0032, range [1.954, 1.965]** over 12 independent seeds at `N_ref=60,000`.

**Verdict — Claim 2 reproduced cleanly, with the null stated correctly.** SurvSHAP-IQ produces a stable, seed-robust ~1.96x separation between scenarios with and without a true interaction, and the "floor" in the no-interaction scenarios is the mechanistically expected consequence of the survival-scale `exp(.)` link acting on an additive log-hazard model — exactly the non-additivity phenomenon the paper's own motivation describes — not Monte-Carlo error or an estimator failure. Framing this as merely "directional" understates the result: the ratio is precise and reproducible to 3 significant figures across seeds once the correct (non-zero) null is used.

**Reproducibility observation.** The paper's Eq. (10) gives the n-Shapley interaction formula, but leaves the construction for lower (`|K| < k`) order terms implicit, inherited from Bordt & von Luxburg (2023) rather than stated in the paper's own text. Applying Eq. (10) literally at every order is the natural first reading, but it breaks local accuracy specifically on scenarios that contain a real feature-feature interaction (see A4 below); the correct construction has to be derived from the Möbius transform given in Appendix A.6. This did not affect the pairwise (top-order, `k=2` for `p=2`) index used above, but it is worth flagging for anyone reimplementing lower-order terms from the paper alone.
