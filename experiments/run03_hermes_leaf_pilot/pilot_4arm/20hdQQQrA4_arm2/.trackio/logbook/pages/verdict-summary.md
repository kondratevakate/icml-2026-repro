## Verdict summary

| Claim | Source | Verdict | Headline evidence |
|---|---|---|---|
| 1 | Theorem 3.5 (UAT + error bound `(3+3√n_out)K`) | **verified** | 0 / 2400 instances exceeded the bound; max ratio **1.71** vs the smallest applicable bound 8.196; 0 infeasible outputs; **706** instances exercised the projection branch |
| 2 | Section 3, trainable null-space term `w_phi` | **verified** | invariance 7.9e-14, reachability 2.2e-14, `w=0` reproduces the fixed orthogonal projection with diff **0.0**, `w` beats it on 100% of **600** instances (median loss reduction 0.517) |
| 3 | Section 3, no full-row-rank + cardinality `min(m,n_out)` | **verified** | 0 violations over **705** cases, all **705** rank-deficient; HardNet-Aff violates **57** of them (8.09%, max 17.67); truncating the enumeration to `min(m,n_out)−1` fails on **114** of 450 (25.33%) |
| 4 | Experiments, piecewise benchmark: 73.33% MSE reduction, zero violations | **verified (structural) / toy (magnitude)** | CAffNet 0 violations on 1200 test points, soft baseline **413**; measured reduction only **9.22%** at CPU scale vs the paper's 73.33% (which is internally consistent: 1 − 0.0012/0.0045 = 73.333%) |
| 5 | Experiments, safety-critical control | **partial: verified / inconclusive** | CAffNet: 0 collisions and ≤7.1e-15 constraint violation in all 3 regimes; baselines lose the hard guarantee (HardNet-Aff 0.620, soft 1.364) but **no baseline collided**, so the collision contrast is untested |
