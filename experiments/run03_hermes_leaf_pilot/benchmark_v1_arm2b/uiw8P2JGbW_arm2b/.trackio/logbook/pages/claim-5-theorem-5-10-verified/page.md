## Claim 5 — Theorem 5.10 — **verified**
`verify_claim5.py` → `results/claim5.json`. FPA, uniform bidding, each bidder both tCPA (`mu ≤ 1`) and budget-constrained; equilibrium multipliers by damped iterated best response.
- 150 instances; **93** exhibit a revenue drop under refinement; max loss 99.85%; the closest sampled loss is **16.82%**, matching the paper's reported **16.8%** to two significant figures (coincidental instance, not the paper's construction).
- Equilibrium multipliers do shift downward on the refined partition (recorded in `best_instance.mu_fine` / `mu_coarse`), which is the stated mechanism.
- **Mutation** (budgets set to +∞): **0** revenue drops — recovers Theorem 5.1, confirming budgets are the cause.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 5 \u2014 Theorem 5.10 \u2014 **verified**"}\n-->
