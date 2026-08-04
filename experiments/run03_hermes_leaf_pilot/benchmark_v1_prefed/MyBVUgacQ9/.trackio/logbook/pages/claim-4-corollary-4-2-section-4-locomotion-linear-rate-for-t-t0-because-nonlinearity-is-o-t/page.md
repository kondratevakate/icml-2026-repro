# Claim 4 — Corollary 4.2 (Section 4): locomotion, linear rate for Δt ≤ t0 because nonlinearity is O(Δt³)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_42dea7744d90", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 4 \u2014 Corollary 4.2 (Section 4): locomotion, linear rate for \u0394t \u2264 t0 because nonlinearity is O(\u0394t\u00b3)"}
-->
**Verdict: `verified`** for the O(Δt³) mechanism and for linear convergence of eq. 6 at small Δt;
the *specific* threshold t0 (the paper's own "Δt = 0.005 s, and the bound is conservative") is not
pinned down — see Evidence boundary.

*Source:* Corollary 4.2 and eq. 6, Section 4 (with Fig. 5's parameters m = 2 kg,
f(f)=½Σ‖f_i‖², φ(z)=5Σ‖k'_i‖²). *Script:* `verify_claim4.py` → `results/claim4.json`.

**(A) Symbolic (sympy).** Independently deriving eq. 6 from the eq.-5 recursion for a planar
instance (T = 4, N = 2 contacts) and splitting the expansion by degree in the forces:

* rows 1–2: 0 quadratic terms; rows 3–4: 8 and 16 quadratic terms, and **every** quadratic
  coefficient has Δt-exponent exactly **3** (`all_quadratic_dt_pow3 = true` for all rows), while
  linear terms carry Δt-exponents {1, 2, 3}.
* the C_i / d_i matrices built by our numeric eq.-6 model match the sympy expansion to
  **2.78e-17** (max abs error over 5 random force vectors).
* fitting ‖C‖ ∝ Δt^p over Δt ∈ {0.002 … 0.05} gives **p = 3.0000**.

This is the paper's stated reason ("the nonlinear term is proportional to (Δt)³"), reproduced exactly.

**(B) Simulation.** Algorithm 1 on eq. 6 (T = 6, N = 2, m = 2 kg, polyhedral box force
constraints, ρ = 4, 4 random initialisations each):

| Δt | ‖C‖ | worst contraction factor | worst final violation |
|---|---|---|---|
| 0.002 | 4.38e-08 | **0.7143** | 7.1e-18 |
| 0.005 | 6.85e-07 | 0.7143 | 8.0e-18 |
| 0.010 | 5.48e-06 | 0.7143 | 1.1e-17 |
| 0.020 | 4.38e-05 | 0.7142 | 1.1e-17 |
| 0.050 | 6.85e-04 | 0.7138 | 2.2e-17 |
| 0.100 | 5.48e-03 | 0.7106 | 4.5e-17 |

Linear convergence at every Δt tested, including well above the paper's suggested 0.005 s —
consistent with the paper's own remark that the corollary's bound is conservative.

**Mutation (mechanism: the Δt³ decay).** Rescaling *only* the quadratic block by Δt⁻³ so that
‖C‖ = 5.477 independently of Δt: the contraction factor jumps to **0.9997 at every Δt** and
feasibility degrades to 4e-05…2e-04. So it is genuinely the O(Δt³) decay of the nonlinearity —
not the smallness of Δt in the linear terms — that produces linear convergence, exactly as
Corollary 4.2 argues.

---
