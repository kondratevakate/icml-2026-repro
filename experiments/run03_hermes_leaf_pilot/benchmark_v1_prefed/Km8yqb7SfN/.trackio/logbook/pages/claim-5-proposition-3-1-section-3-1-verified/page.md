# Claim 5 — Proposition 3.1, Section 3.1 → **verified**

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a9f69762d65c", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 5 \u2014 Proposition 3.1, Section 3.1 \u2192 **verified**"}
-->
Script `verify_claim5.py`, data `results/claim5.json`.

* Normalization: sympy confirms d/dy [y^α/(y^α+(1−y)^α)] equals the claimed density exactly, so the
  CDF is closed-form and F(1)−F(0) = 1; numeric quadrature gives mass 1.000, 1.002, 1.000, 1.000 for
  N = 2,4,8,16 (the 2e-3 deviation at N = 4 is quadrature error at the endpoint singularity).
* Optimality: for β = 0 an interior simplex maximiser must have a constant functional derivative.
  Using Prop. 4.2 with r₁(y) = 1−y², r₂(y) = 1−(1−y)²:
  ∂R₁/∂π(y) = −∫₀^y 2Nz(1−F(z))^{N−1}dz, ∂R₂/∂π(y) = −∫_y^1 2N(1−z)F(z)^{N−1}dz.
  At the claimed π*, the relative span of 0.5(∂R₁/∂π + ∂R₂/∂π) on [0.02, 0.98] is
  **0.0 / 4.16e-16 / 4.54e-16 / 5.84e-16 / 5.74e-16** for N = 2,3,4,8,16 — machine precision.
* Grid optimisation (K = 400 bins, exact mirror ascent, 5 inits): objective matches the closed form to
  ≤ **1.45e-5**; from the uniform init at N = 2 the TV to the closed form is 6.3e-15. From random
  inits the TV can stay as large as 0.56 while the objective gap is < 1.5e-5 — the β = 0 objective is
  extremely flat near the optimum on a 400-bin grid, so the TV agreement is weak evidence and the
  verdict rests on the exact first-order-optimality test above.
* Naive optimum: 0.5E[r₁]+0.5E[r₂] optimisation puts mass **1.0** within 0.01 of y = 0.5
  (argmax bin 0.49875, mean 0.5, std 1.25e-3) — the claimed δ_0.5 collapse — and **0.0** mass outside
  [0.25, 0.75], versus **0.922** for IAMA with N = 8 (the claimed bimodality).
* **Mutation A**: N = 2 ⇒ α = 1 ⇒ the formula must be exactly uniform; max|π*(y) − 1| = **0.0**.
* **Mutation B**: use α = 1/N instead of 1/(N−1); the optimality span jumps to **0.162, 0.122, 0.062,
  0.031** for N = 3,4,8,16 — i.e. 14–15 orders of magnitude above the correct exponent.
