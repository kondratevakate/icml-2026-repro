## Claim 4 — Theorem 3, delayed generalization gap `ηT·exp(ηT(K−k+1)/√m)/n`

**Source recorded:** `"Theorem 2.3 (Sec 2.2); restated as Theorem B.6 (App. B)"`
**Route:** `"simulation of Algorithm 1 + Monte-Carlo over D_k; bound evaluated literally"`. Config: `d=12, K=3, T=40, k=1, ηT=72.0, m=1152, n_test=4000, reps_per_cell=12`, loss `"Huberised hinge (1-Lipschitz, 1-smooth), as the theorem assumes"`.

**Verdict: `verified`.**

**Evidence (n sweep, gap vs literal RHS):**

| n | mean gap | stderr | bound RHS | bound holds |
|---|---|---|---|---|
| 50 | 0.002740975027481212 | 0.000672 | 835.97954431853 | true |
| 100 | 0.0017280548807244174 | 0.000816 | 417.989772159265 | true |
| 200 | 0.0006088137099770553 | 0.000171 | 208.9948860796325 | true |
| 400 | 0.0002555535816231669 | 6.60e-05 | 104.49744303981625 | true |
| 800 | 0.00019016911846579361 | 0.000134 | 52.24872151990812 | true |

`"bound_holds_all_cells": true`, `"decays_with_n": true`, `gap_loglog_slope_in_n = -1.0456119905699754` vs `predicted_slope = -1.0`.

Honest evidence boundary, verbatim:
> `"The RHS of Thm 2.3 is an order bound with hidden constants; at these (feasible) widths it is numerically very loose, so 'bound holds' is a weak test and the informative evidence is the recovered 1/n exponent."`

**Mutation — shrink n 8× (`mutation_shrink_n_8x`):** `"property_breaks": true`; gap moves `0.0002555535816231669 (n=400) → 0.002740975027481212 (n=50)`, `ratio = 10.725637301076794` against the predicted ~8×.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 Theorem 3, delayed generalization gap `\u03b7T\u00b7exp(\u03b7T(K\u2212k+1)/\u221am)/n`"}\n-->
