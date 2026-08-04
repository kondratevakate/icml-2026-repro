## Summary table

| Claim | Verdict | One-line evidence |
|---|---|---|
| 1 | verified | max \|\|P*-f_t\|\|/K = 1.6869 vs bound 6.0–9.71 over 1600 instances; mutation (argmax) breaks it 8/8 (up to 677.6) |
| 2 | verified | sub-constraint residual 1.18e-13 for any w; trainable w objective 1.859 vs 4.501 for w=0, better on 60% of instances |
| 3 | verified | 0/2100 violations (max 1.11e-12) incl. rank-deficient A, while HardNet undefined on 1961/2100; k-cap mutation loses feasibility 478/2100 |
| 4 | inconclusive | zero violations reproduced (0.0 both CAffNets, 5/5 seeds), but reduction 50.70% vs the claimed 73.33% at 20000/50000 epochs; CAffNet-FF worse than NN |
| 5 | inconclusive | CAffNet 1.0 vs soft NN 20.33 collisions / 49, but CAffNet not collision-free (4 empty-feasible-set steps) and HardNet undefined (A is 13x2) |

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Summary table"}\n-->
