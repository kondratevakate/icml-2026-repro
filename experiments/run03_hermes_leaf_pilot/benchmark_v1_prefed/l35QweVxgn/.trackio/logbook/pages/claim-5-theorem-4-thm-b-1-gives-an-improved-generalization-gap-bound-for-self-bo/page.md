## Claim 5 — — Theorem 4 (Thm B.1) gives an improved generalization gap bound for self-bounded losses that scales poly-logarithmically rather than linearly in T, depending on the cumulative training loss of later tasks.

- **Verdict:** `verified`
- **Source:** Theorem B.1 (improved gen. gap, the paper's 'Theorem 4'), Remark B.2, Eqs (6), arXiv:2510.05573v2.
- **Seed:** 20261307  (master 20260802)
- **Mutation test:** Per-step training loss held constant (not self-bounded / net not in kernel regime).  [mutation breaks]  With a non-decaying per-step loss the cumulative loss ~ T, so the improved exponent grows with T and the bound reverts to >= linear in T, destroying the poly-log advantage.
- **Key numerics:**
  - `T_scan`: {"T_grid": [50, 100, 200, 400, 800], "G_improved": [0.07197762180584784, 0.11741682859618946, 0.17881633903861277, 0.258578418202709, 0.3591053311580691], "G_Thm23": [1.4675852886269917e-05, 2.9351705797826993e-05, 5.870341169680262e-05, 0.00011740682379819979, 0.00023481364921477772], "improved_growth_ratios": [1.6312963064118617, 1.5229191690535568, 1.4460558782990898, 1.3887676073436006], "predicted_polylog_ratios": [1.6312963064118615, 1.522919169053557, 1.44605587829909, 1.3887676073436002], "Thm23_growth_ratios": [2.0000000017230453, 2.0000000034460905, 2.000000006892181, 2.000000013784362], "polylog_in_T_confirmed": true, "linear_grows_faster": true}
  - `later_task_dependence`: {"later_loss_scale": [0.2, 0.5, 1.0, 2.0], "G_improved_by_scale": [3.041589853573734e-07, 3.2084967927197954e-07, 3.507275439982487e-07, 4.1908913634621457e-07], "depends_on_later_loss": true}

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 \u2014 Theorem 4 (Thm B.1) gives an improved generalization gap bound for self-bounded losses that scales poly-logarithmically rather than linearly in T, depending on the cumulative training loss of later tasks."}\n-->
