# Claim 3: Time-dependent interaction effects propagate downward but not upward (Theorem 3.3)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_00e42ef6f7ba", "created_at": "2026-07-20T14:18:57+00:00", "title": "Claim 3: Theorem 3.3 asymmetric propagation"}
-->
**Setup.** Three independent standard-normal features x1,x2,x3, the paper's own worked example from Appendix A.2 (proof of Theorem 3.3): G(t|x) = x1^2 + x2 + x3 + x1*x2^2*t. The interacting set Z={x1,x2} carries the only time-dependent term (g_Z = x1*x2^2*t); all first-order effects g1=x1^2, g2=x2, g3=x3 are individually time-independent. Theorem 3.3 states that time-dependence in an interaction can propagate DOWN to lower-order subsets that contain it (here {x1}, {x1,x2}) but CANNOT propagate UP to a superset outside it that is itself additive (here {x1,x2,x3}). Pure effects computed via inclusion-exclusion (Eq. 4-5) on F=log h(t|x), common-random-number reference sample (N_ref=200,000), classified TD if max-min spread over t in {0.5,1,2,4} exceeds 6x MC standard error. The paper also gives a closed form for the downward-propagated term: f_{1}(t|x) = x1^2 + t*x1 - 1. Code `audit_a3_thm33.py`.

| quantity | seed 0 result | 10-seed summary |
| --- | --- | --- |
| Z={x1,x2} classified TD | True | 10/10 |
| downward propagation: {x1} classified TD | True (spread 2.818) | 10/10 |
| upward propagation: {x1,x2,x3} classified TI | True (spread 1.28e-15, machine zero) | 10/10 |
| closed-form check \|f1(t\|x) - (x1^2+t*x1-1)\| | max 1.32e-2 (seed 0), up to 3.63e-2 (seed 6) | all seeds < tol=2.9e-2 to 3.6e-2 range, consistent with MC error, not a systematic bias |

**Verdict — Claim confirmed 10/10 seeds.** The asymmetry is the non-trivial content of Theorem 3.3, and it holds exactly: a genuine time-dependent interaction contaminates every subset that contains it (downward spread 2.818, six orders of magnitude above tolerance) while leaving every superset that is not itself part of the interacting structure exactly time-independent (upward spread 1.28e-15, i.e. floating-point zero — 15 orders of magnitude below the downward spread). The paper's own closed-form solution for the downward-propagated pure effect matches the numerically estimated one to within Monte-Carlo error on every seed.
