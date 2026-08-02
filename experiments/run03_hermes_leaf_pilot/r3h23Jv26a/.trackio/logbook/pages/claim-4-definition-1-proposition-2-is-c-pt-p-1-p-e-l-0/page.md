# Claim 4 — Definition 1 + Proposition 2: IS(C_PT) = p(1−p)(E L)² > 0

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_7d3f90a8e6e2", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 4 \u2014 Definition 1 + Proposition 2: IS(C_PT) = p(1\u2212p)(E L)\u00b2 > 0"}
-->
**Source:** Definition 1 and Proposition 2, §4. **Script:** `verify_claim4.py` → `results/claim4.json`.
**Verdict: `verified`.**

Justifying numbers:
- Symbolic: for the two-point interval-length law {0 w.p. 1−p, L w.p. p}, sympy returns Var = `-L**2*p*(p-1)`
  = p(1−p)L², matching Proposition 2 exactly (`symbolic_matches_prop2: true`).
- Monte-Carlo IS (Definition 1: mean over 500 test points of the variance of |C| over 400 repeated runs, fixed
  calibration set), 5 seeds × p ∈ {0.91,…,0.99} at α=0.10: empirical IS matches the closed form to within
  **2.47 % worst case** across all 35 cells, e.g. p=0.95 → IS 26.27 vs 26.22; p=0.98 → 10.38 vs 10.40;
  p=0.99 → 5.16 vs 5.18. IS is strictly positive everywhere (minimum **4.96**), i.e. the same input really does
  receive different intervals across runs.
- Deterministic VCP on the identical inputs: IS = **3.0e-26** (floating-point zero).

**Mutation test:** set p = 1 (remove the null branch, PT degenerates to its base). Empirical IS drops to
**3.0e-26 ≈ 0**, matching p(1−p)L² = 0. The non-zero stability is therefore produced by the PT randomization
itself, not by the estimator or the calibration noise.

---
