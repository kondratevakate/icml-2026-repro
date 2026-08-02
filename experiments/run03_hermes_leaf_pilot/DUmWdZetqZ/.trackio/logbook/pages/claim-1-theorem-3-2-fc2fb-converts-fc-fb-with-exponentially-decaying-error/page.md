# Claim 1 — Theorem 3.2: FC2FB converts FC → FB with exponentially decaying error

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_021a545a4eee", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 1 \u2014 Theorem 3.2: FC2FB converts FC \u2192 FB with exponentially decaying error"}
-->
**Source:** Section 3, Algorithm 3 + Theorem 3.2.
**Script:** `verify_claim1.py` → `results/claim1.json`
**Verdict: `verified` (with a boundary caveat on the printed validity condition).**

Justifying numbers:
- Grid of 432 configurations `(A ∈ {5,20,100,500}) × (C ∈ {0,10,200}) × (Q ∈ {1,8}) ×
  (δ₀ ∈ {0.5, 1/e, 0.1}) × (B = ⌈B_min·m⌉, m ∈ {1,2,4,8,16,32})`, exact error vs the
  Theorem 3.2 bound `3exp(−B/(4Q/ln(1/δ₀)+4log₂(B/Q)A))`.
- On the 360 points with **B ≥ 2·B_min: 0 violations**, minimum slack ratio (bound/actual)
  **2.17**.
- **Exponential decay:** at `A=50, C=20, Q=1, δ₀=1/e`, exact error falls from `7.758e-01` to
  `1.604e-28` as B goes 704 → 90 078 (a factor **4.84e27** over 128× budget); linear fit of
  `log P(err)` on B gives slope **−6.820e-04** per sample, **R² = 0.983** (the residual is the
  `⌊log₂⌋` staircase in R, not curvature).
- **Monte Carlo cross-check** (5 seeds × 200 000 runs, B = 2815): exact `2.2946e-02` vs MC mean
  `2.2659e-02`, absolute deviation **2.87e-04**.

**Boundary caveat (a real, small discrepancy).** 12 of the 432 points violate the printed
inequality. **All 12 sit exactly at the stated threshold `B = B_min`** (max `B/B_min` among
violations = 1.0) and **all have C > A**. Worst case: `A=5, C=200, Q=1, δ₀=0.5, B=2460` →
exact error `9.730e-02` vs bound `7.134e-05` (slack ratio `7.33e-04`). Cause: the theorem's
exponent contains no `C`, while the achievable per-stage confidence is
`δ₀^{(B/R − C)/(A ln(1/δ₀))}`; the printed threshold `B ≥ 2(A ln(1/δ₀)+C+1)ln(…)` does not force
`B/R` far enough above `C` when `C ≫ A`. Requiring `B ≥ 2·B_min` removes every violation. This
affects only the constant in the validity condition, not the substance of the claim
(the exponential decay in B, which is what the anchored claim states).

**Mutation test.** Replacing the doubly-exponential schedule `L_r = 2^{R−r}`:
- M1 constant schedule (`L_r = 1`, every stage uses δ₀): error **stalls at 0.582** (ratio
  first/last budget = 1.0001 — no decay at all) and violates the Theorem 3.2 bound at **5 of 8**
  budgets. Compare: true schedule reaches `1.604e-28` at the same budget.
- M2 reversed schedule (`L_r = 2^{r−1}`): stalls at **0.419**, **5 of 8** bound violations.
The mechanism (increasing δ at a doubly exponential rate) is therefore load-bearing, not incidental.

---
