# Claim 1: SurvFD separates time-dependent and time-independent higher-order effects for interpreting survival models


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bce4fea559d5", "created_at": "2026-07-19T09:09:07+00:00", "title": "Claim 1: SurvFD separates time-dependent and time-independent higher-order effects for interpreting survival models"}
-->
**Setup.** Two independent synthetic features `X1, X2 ~ U(-1,1)`, constant baseline hazard `h0=0.1`, hazard model `h(t|x)=h0*exp(G(t|x))` (Eq. 3). SurvFD pure effects computed via marginal functional decomposition (Eq. 4-5, inclusion-exclusion) on `F(t|x)=log h(t|x)` (Eq. 8), using **common random numbers**: ONE fixed 60,000-sample marginal reference set reused for every timepoint and subset (fixing a bug in a first version where independently-redrawn samples per timepoint made pure MC noise look like time-dependence). A subset is classified time-dependent (TD) if its pure effect varies across `t in {0.5, 2.0, 5.0}` by more than 6x the empirical MC standard error. Code `verify_survfd.py`.

| scenario | ground truth | SurvFD verdict | Theorem 3.2 exact match |
| --- | --- | --- | --- |
| A: additive, no interaction, TI (`1.2x1+0.9x2`) | x1 TI, x2 TI, x1x2 TI | x1 TI, x2 TI, x1x2 TI | **yes** |
| B: additive, no interaction, x2 TD (`1.2x1+0.9x2*log(1+t)`) | x1 TI, x2 TD, x1x2 TI | x1 TI, x2 TD, x1x2 TI | **yes** |
| C: linear + interaction, TI (`1.2x1+0.9x2+0.8x1x2`) | x1 TI, x2 TI, x1x2 TI | x1 TI, x2 TI, x1x2 TI | **yes** |
| D: nonlinear main + interaction (boundary, outside Thm 3.2) | ambiguous by construction | x1 TI, x2 TI, x1x2 TI | n/a (theorem doesn't cover this case) |

**Verdict — Claim 1 reproduced exactly on the covered cases (3/3).** For every scenario the paper's Theorem 3.2 guarantees exact recovery under (independent features, linear-incl-interactions or purely-additive G), SurvFD's estimated partition matches the ground truth to the letter — including correctly flagging exactly one main effect as time-dependent in scenario B while leaving everything else time-independent, and correctly keeping the whole system time-independent in scenario C despite the added interaction term. This is a strong, literal confirmation of the theorem, not just a qualitative trend.

**Process note (honesty).** The first implementation attempt failed this test completely — every effect in every scenario, including the trivially time-independent scenario A, came out 'time-dependent'. The cause was drawing a fresh Monte-Carlo reference sample independently at each evaluated timepoint: with independent draws, the MC estimation noise at different t is itself indistinguishable from a genuine time trend. Reusing one fixed reference sample across all t (common random numbers) removed that confound entirely.

**Seed robustness (`audit_seed_fragility.py`, 12 independent seeds, `N_ref=60,000`).** The 3/3 result is not a lucky draw: it holds on all 12 seeds tested. The margin is not marginal either — time-independent pure effects have a max-min spread across t of exactly `0.0` or `~1e-15` (floating-point noise), while the genuine time-dependent effect in scenario B has a spread of `7.48e-01`, a gap of about 15 orders of magnitude. The classification tolerance (6x the Monte-Carlo standard error) is not doing load-bearing work either: sweeping the tolerance multiplier from 0.001x to 100x still gives 3/3 on all 12 seeds; only at an implausible 1000x does the verdict change (2/3). The result is stable across five orders of magnitude of tolerance choice.
