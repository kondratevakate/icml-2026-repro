## Claim 6 — Theorem 5.11 + Table 1 — **inconclusive**
`verify_claim6.py` → `results/claim6.json`. Centralized non-strategic LP benchmark solved with `scipy.optimize.linprog` (HiGHS): maximize `Σ m_C v_i hatp_i(C) x_{C,i}` s.t. one slot per cluster and per-bidder budgets on `Σ m_C t_i hatp_i(C) x`.
- Welfare monotonicity violations across 400 instances: **0**.
- Explicit **lifting** check (copy the coarse solution to every subcluster): feasible in 400/400; objective error **4.4e-16**; budget overshoot **3.3e-16** — the lift is exactly value- and feasibility-preserving, which is the theorem's proof device.
- **Mutation** (miscalibrated coarse predictions): 19 monotonicity violations — calibration is required.
- **Why inconclusive:** the claim is a conjunction, and its second conjunct ("Table 1 identifies exactly three settings where monotonicity holds") requires the paper's table, which could not be retrieved. The Theorem 5.11 half is reproduced cleanly; the completeness half is untested.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 Theorem 5.11 + Table 1 \u2014 **inconclusive**"}\n-->
