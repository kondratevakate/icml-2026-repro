# Claim 1 — Theorem 3.1 (Section 3): sublinear convergence to a stationary point

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b921fff1e8ac", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 1 \u2014 Theorem 3.1 (Section 3): sublinear convergence to a stationary point"}
-->
**Verdict: `verified`.**

*Source:* Theorem 3.1, Section 3 (with Assumptions 2.3, 2.6, Definition 2.7 / eq. 3, Algorithm 1).
*Script:* `verify_claim1.py` → `results/claim1.json` (command recorded in the JSON).

Numbers (450 runs = toy problem q ∈ {0.5, 2, 5} × 3 ρ values ≥ the Theorem-3.1 threshold ×
16 x-initialisations × 3 z-initialisations, plus Example 2.2 with 9 initialisations × 2 ρ):

| quantity | value |
|---|---|
| runs converged (`res < 1e-8`) | **450 / 450** |
| runs feasible (`‖A(x)+Qz‖ < 1e-8`) | **450 / 450** |
| max final residual | **3.846e-16** |
| max final constraint violation | **2.220e-16** |
| o(1/k) witness `max_k k·(L^k − L*)` on the tail | **0.000e+00** (450/450 runs below 1e-6) |
| max blockwise (Nash-like) optimality error of the limit point | **5.192e-10** |

The last row is the substantive part of Theorem 3.1: for every block *i* we solved the paper's
characterisation `x*_i ∈ argmin f(x_i, x*_{−i}) s.t. A(x_i, x*_{−i}) + Q z* = 0` exactly and
compared with the ADMM limit — agreement to 5e-10. The observed rate is in fact geometric, which
is consistent with the theorem's "**at least** sublinear".

**Mutation A (mechanism: Assumption 2.6, Q full row rank).** Example 2.8 (`min x²+y² s.t. xy=1`,
Q = 0 ⇒ not full row rank), 12 runs (4 initial points × 3 ρ): iterates collapse to the origin
(`max‖x_final‖ = 0.0`), the dual diverges (`w = −4000` at every setting, monotone in ρ·k), and the
limit is **infeasible** (`violation = 1.000` in all runs). Removing Assumption 2.6 therefore
destroys exactly what Theorem 3.1 delivers — the theorem's hypothesis is load-bearing.

**Mutation B (the ρ threshold).** With `ρ_threshold = 500` for a problem with L_φ = μ_φ = 5,
Q = [0.2]: ρ = 1e-4·threshold … 0.1·threshold all still converge (worst residual < 1e-12) while
ρ ≥ threshold converges more slowly (worst residual 1.2e-4 after 1500 iterations). This does **not**
contradict Theorem 3.1 (its ρ bound is sufficient, not necessary) but it is honest evidence that
the stated bound is conservative and not the mechanism driving convergence in practice.

---
