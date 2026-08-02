# Claim 3 — Theorem 5.2, Section 5.1 → **verified** (up to a numerically estimated L)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_165172086830", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 3 \u2014 Theorem 5.2, Section 5.1 \u2192 **verified** (up to a numerically estimated L)"}
-->
Script `verify_claim3.py`, data `results/claim3.json`.

* Grid: K ∈ {5,8} × N ∈ {2,4,8} × β ∈ {0.1,0.5,1.0} × 5 reward seeds = **90 instances**, each checked
  at every T = 1…25 (**2250** bound evaluations). π₀ = π_ref = uniform; π* and L[π*] from L-BFGS.
* L is not given in closed form in the paper's main text, so it is estimated as the smallest constant
  satisfying Assumption 5.1 over 4000 random policy pairs per instance, then inflated ×1.25.
  The concavity half of Assumption 5.1 was violated **0** times (max violation exactly 0.0).
* Result: **0** violations with excess loss above 1e-12. The 44 raw ratio>1 rows all occur at T ≥ 18
  with LHS ≤ **3.33e-16**, i.e. double-precision noise where both sides are already ≲1e-16.
  Max excess loss at T = 25 across all instances: **2.04e-6**, consistent with linear convergence.
* **Mutation A** (η = 20/L, violating the theorem's step size): 21/36 ratio violations, **11**
  macroscopic (final excess loss > 1e-3), worst final excess loss **0.593**.
* **Mutation B** (replace ∂R/∂π by the plain reward): **36/36** violations, 35 macroscopic,
  final excess loss 9.0e-4…8.9e-2 — the guarantee is specific to the functional-derivative update.
