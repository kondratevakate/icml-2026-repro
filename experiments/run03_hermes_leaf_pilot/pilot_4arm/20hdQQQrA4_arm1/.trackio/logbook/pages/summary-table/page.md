## Summary table

| # | Claim (source) | Verdict | One-line evidence |
|---|----------------|---------|-------------------|
| 1 | Thm 3.5: ‖P*−f_t‖_p < (3+3√n_out)K | **verified** | 0 / 9241 violations on genuine Case-2 instances, worst ratio 0.400; mutation of Eq. 12 → 3545 / 9241 violations |
| 2 | Eq. 8 trainable null-space w_φ | **verified** | A_γP_γ−b_γ ≤ 1.9e−11; output invariant to w_φ iff full column rank (2.9e−12 vs median spread 0.851); trained w_φ beats w_φ=0 on 8/8 seeds (0.458 vs 0.896) |
| 3 | No full-row-rank; cardinality ≤ min(m,n_out) | **verified** | 0 / 2160 violations (max 2.7e−14) where a HardNet-style single projection violates on 999 / 2160 (max 28.47); truncating Γ breaks feasibility (506 / 194 failures) |
| 4 | 73.33% MSE reduction + zero violations (§4.1, Table 2) | **inconclusive** | Aggregated 5-seed run: CAffNet-TF MSE 0.000570 vs NN 0.001728 → ~67% reduction (paper claims 73.33%); direction matches but estimate is noisy (per-seed −118%…+77%); zero-violation half holds (viol 3.3e−1 → 9.2e−8) |
| 5 | CAffNet avoids obstacles, HardNet/soft fail (§4.3, Table 4) | **inconclusive** | The paper does not specify the per-step affine safety constraint, reference, dt or loss; Table 4 is not reproducible from the paper. A self-designed mechanism test (labelled *toy*) gives 0/5 vs 5/5 collided |

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Summary table"}\n-->
