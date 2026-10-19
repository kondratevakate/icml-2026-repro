# Claim 4: medical-and-safety-critical-gains


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e8c0319ffa14", "created_at": "2026-07-28T15:58:37+00:00", "title": "Claim 4: medical-and-safety-critical-gains"}
-->
**Anchored claim (verbatim).** "Empirical evaluations on medical diagnosis and safety-critical decision-making tasks show ROCP reduces critical mistakes compared to baselines, with the largest gains when out-of-set errors are costly."

**Verdict -- VERIFIED.**

The official cached probabilities/scores were evaluated from scratch at commit
`3ee0cf6e393d2e434368bc1fc7fe3abd03ed493f`: COVID seeds 23-42 under both
`Lambda0` and `Lambda1`, BDD split seeds 23-42, all seven alpha values, and the
full ROCP/RAC/LAS/APS/SOCOP/best-response panel.

At `alpha=0.05` under `Lambda0`, ROCP critical-mistake rates are `1.19%`,
`10.59%`, and `4.77%` for Pneumonia, COVID-19, and Lung Opacity, versus RAC
rates of `1.54%`, `12.95%`, and `5.94%`. Under the high-cost `Lambda1`, ROCP
has zero observed critical mistakes for all three labels while RAC remains at
the same nonzero rates. The gain is therefore larger for every critical label
when out-of-set errors are more costly.

On BDD, mean miscoverage at `alpha=0.05` is `4.76%`. ROCP realized loss is
`2.320 +/- 0.019` versus RAC `2.981 +/- 0.075`, and ROCP reduces collision
rates for every nonzero hazard state.

**Boundary.** The stronger Section 5.2 statement that ROCP matches or
outperforms all baselines at every alpha is not literally reproduced against
RAC: small reversals occur for realized loss at `alpha=0.005` and worst-case
risk at `alpha=0.03/0.05`. They are within about one combined standard error
and do not negate the anchored critical-mistake claim.
