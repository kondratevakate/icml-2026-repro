## Claim 5 — Theorem 4, poly-log-in-T bound for self-bounded losses

**Source recorded:** `"Theorem B.1 + Remark B.2 (App. B); contrast against Theorem 2.3"`
**Route:** `"bound functionals evaluated on the measured GD trajectory (logistic loss)"`. Config: `d=12, K=3, k=1, m=1152, n=300, η=0.18, T_max=400, reps=5`, loss `"logistic (self-bounded, 1-Lipschitz, 1-smooth)"`.

**Verdict: `verified`.**

**Evidence (T grid `[25, 50, 100, 200, 400]`):**

| T | Thm 2.3 bound | Thm B.1 bound |
|---|---|---|
| 25 | 0.022327023601958818 | 0.009238054026995872 |
| 50 | 0.06646613105632347 | 0.011260007205850691 |
| 100 | 0.2945164385064245 | 0.013238968420032674 |
| 200 | 2.8913310850169514 | 0.015261516970516333 |
| 400 | 139.32992405308832 | 0.017380270445999442 |

`loglog_slope_in_T`: `thmB_1 = 0.22622699057348847` (predicted `"<< 1 (polylog)"`) vs `thm2_3 = 3.0657823793584993`. `B1_tighter_than_2_3_at_Tmax = true`, `ratio_B1_over_23_at_Tmax = 0.00012474183535316584`.

Dependence on later tasks' cumulative loss: `c_kK_cumulative_loss_of_later_tasks = 37.41150126889778`; measured cumulative task-1 train loss `[12.626, 15.389, 18.094, 20.858, 23.754]` fits `log³T` better than `T` (`pearson_r_vs_log3_T = 0.9868994223630535` vs `pearson_r_vs_T = 0.9371800467149037`, `log3_fits_better = true`).

**Mutation — frozen non-decaying loss:** `"property_breaks": true`; B.1's T-exponent jumps to exactly `loglog_slope_in_T = 1.0` (bound `[0.02543 … 0.40682]`), i.e. the poly-log advantage disappears precisely when the self-bounding hypothesis is removed.

Evidence boundary, verbatim: `"This verifies the two bounds' T-dependence as functionals of a real GD trajectory; it does not re-derive the stability proof of Thm B.1. The exp(·/√m) factors are ≈1 at the widths used, so the comparison isolates the T-dependence, which is the claim."`

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 Theorem 4, poly-log-in-T bound for self-bounded losses"}\n-->
