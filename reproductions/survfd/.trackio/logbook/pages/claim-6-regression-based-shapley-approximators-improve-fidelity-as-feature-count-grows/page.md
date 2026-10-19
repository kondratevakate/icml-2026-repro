# Claim 6: regression-based Shapley approximators improve fidelity as feature count grows


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_survfd_c6_scalability_20260729", "created_at": "2026-07-29T00:00:00+00:00", "title": "Claim 6: synthetic scalability proxy"}
-->
## Claim

Figure 7 compares Shapley interaction approximation methods on four additional
datasets with 10-16 features and reports the best fidelity for
regression-based approximators.

## Independent proxy

`repro_survfd_v2/audit_scalability_proxy.py` constructs four seeded synthetic
coalition games with 10, 12, 14, and 16 features. Each game contains known
main and pair coefficients. It compares:

- joint least-squares regression, which reuses every coalition evaluation
  across all coefficients;
- a pairwise permutation estimator based on noisy second differences.

Both methods receive exactly the same coalition-evaluation budget. Results are
aggregated over 20 seeds at two budgets per feature count.

| Features | Budget | Regression RMSE | Permutation RMSE | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 112 | 0.1987 | 0.6466 | 0.307 |
| 10 | 224 | 0.1055 | 0.6516 | 0.162 |
| 12 | 158 | 0.1585 | 0.6590 | 0.241 |
| 12 | 316 | 0.0936 | 0.6631 | 0.141 |
| 14 | 212 | 0.1375 | 0.6575 | 0.209 |
| 14 | 424 | 0.0767 | 0.6592 | 0.116 |
| 16 | 274 | 0.1210 | 0.6594 | 0.183 |
| 16 | 548 | 0.0706 | 0.6835 | 0.103 |

Regression has lower coefficient RMSE in all `4 x 2` task-budget cells. The
direction is consistent with Figure 7 and demonstrates the evaluation-reuse
mechanism at the same 10-16 feature counts.

## Verdict

**TOY, not verified.** This is a synthetic mechanism check, not a reproduction
of Figure 7. It does not use the paper's four datasets, does not run the
complete approximation-method panel, and does not establish the paper's exact
empirical ranking. Those omissions are intentional and should not be scored as
full verification.
