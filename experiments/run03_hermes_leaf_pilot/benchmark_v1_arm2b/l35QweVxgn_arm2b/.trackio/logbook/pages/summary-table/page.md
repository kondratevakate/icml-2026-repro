## Summary table

| # | Anchored claim (source in paper) | Route | Verdict | Mutation broke property? | Key executed number |
|---|---|---|---|---|---|
| 1 | Thm 1 closed-form train-time forgetting bound, order `O~(ηT√(K−k)/(d√n) + …)` — *Theorem 2.1 / Eq.4 (Sec 2.2), via Eq.17* | analytic functional + Monte-Carlo exponent recovery | **verified** (scoped) | yes | log-log slopes `0.4890` in (K−k) vs predicted `0.5`; `−0.4880` in n vs predicted `−0.5` |
| 2 | Regime `n=Θ~(d²K)`, `m=Ω~(d⁸K⁴)`, `ηT=Θ(d²)` — *Thm 2.1 / Thm C.1* | simulation of Algorithm 1 with prescribed scalings, width scaled down | **toy** (n, ηT verified; full m regime inconclusive) | **no** (hardened mutant still decreased) | forgetting `0.1565 → 0.00716` over `d=8…20`, ratio `21.87` |
| 3 | Thm 2 — misclassification error uniformly small across all K tasks after KT steps — *Theorem 2.2 / Thm C.1* | simulation of Algorithm 1, KT iterations, uniform-over-tasks stats | **verified (qualitative)**, `verdict_cap: toy` | yes | mean max-task err `0.00130` at d=8, exactly `0.0` for d=12,16,20 |
| 4 | Thm 3 — delayed generalization gap `≤ ηT·exp(ηT(K−k+1)/√m)/n`, decays in n — *Theorem 2.3 / Thm B.6* | simulation + Monte-Carlo over `D_k`, bound evaluated literally | **verified** | yes | gap log-log slope in n `−1.0456` vs predicted `−1.0`; bound held in **all** cells |
| 5 | Thm 4 — self-bounded-loss gap scales poly-log, not linearly, in T — *Theorem B.1 + Remark B.2 vs Thm 2.3* | bound functionals on the measured GD trajectory (logistic loss) | **verified** | yes | T-exponent `0.2263` (B.1) vs `3.0658` (Thm 2.3); ratio at `T_max` `1.247e-4` |
| 6 | Test-time forgetting = train-time forgetting + delayed gen gap; width and sample size control forgetting **jointly** — *Eq. 3 (Sec 2.1.3) + Thm 2.1 discussion* | direct measurement of all four quantities + 2-D (m,n) sweep | **verified** | yes | identity exact, `max_identity_residual = 1.084e-19`; `both_ratio 0.00546` vs `m_only 0.0478`, `n_only 0.7366` |

**Tally:** 4 verified, 1 verified-qualitative (capped *toy* on width), 1 *toy* (one sub-condition computationally unreachable). 0 falsified.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Summary table"}\n-->
