## Claim 3 — Theorem 5.2 / Corollary 5.3 — **verified**
`verify_claim3.py` → `results/claim3.json`. 2,000 instances × 300 random multiplier profiles + all one-bidder unilateral deviations.
- tCPA feasibility ⟺ `mu_i ≤ 1`: **0** mismatches (analytically `spend_i = mu_i t_i conv_i`).
- Feasible profiles beating mu=1 on revenue: **0**; on welfare (v=t): **0**; unilateral deviations beating mu=1 on the deviator's own conversions: **0**.
- Corollary 5.3 welfare-monotonicity violations under refinement: **0**.
- **Mutation** (v_i ≠ t_i, drawn as t·U[0.3,3]): the mu=1 allocation is *not* welfare-optimal in 1,732/2,000 instances — the `v = t` hypothesis is necessary.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 Theorem 5.2 / Corollary 5.3 \u2014 **verified**"}\n-->
