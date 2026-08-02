# Claim 3 — FCW2S (Algorithm 4) converts weak FC (Def 4.1) → strong FC (Def 3.1)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_80506f3db91d", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 3 \u2014 FCW2S (Algorithm 4) converts weak FC (Def 4.1) \u2192 strong FC (Def 3.1)"}
-->
**Source:** Section 4, Algorithm 4, Propositions 4.2 and 4.3 (proofs in Appendix C).
**Script:** `verify_claim3.py` → `results/claim3.json`
**Verdict: `verified`.**

Setting: `δ₀ = 1/(8e)` (the value the paper itself uses in Corollary 5.7), `f(δ₀) = 100`,
`L = ⌈4 ln(1/δ)/ln(1/(4e δ₀))⌉`. Error/stopping probabilities computed by **exhaustive
enumeration of the whole multinomial support** `(n_wrong, n_correct, n_never)`; all wrong
instances vote for the *same* wrong arm and ties go to the wrong arm (worst case).

Justifying numbers over 8 target δ from 0.5 to 1e-8:
- **Prop 4.2**: `0/8` violations of `(4eδ₀)^{L/4}` and `0/8` violations of `≤ δ`.
  e.g. δ=0.1 → L=14, exact error `2.321e-06` ≤ bound `8.839e-02` ≤ δ.
- **Prop 4.3**: `0/8` violations of `(2eδ₀)^{L/2}` and `0/8` of `≤ δ`.
  e.g. δ=0.1 → exact tail `1.120e-06` ≤ bound `6.104e-05`.
- **Strongness (Def 3.1)**: `T*_δ = L·f(δ₀)` fitted on `ln(1/δ)` gives slope
  **578.07**, against the predicted `4f(δ₀)/ln(1/(4eδ₀)) = **577.08**`, with **R² = 0.99993** —
  i.e. `T*_δ = A ln(1/δ) + C` with A ≈ 578, exactly the form Definition 3.1 requires.
- Literal event simulation of Algorithm 4 (5 seeds × 20 000 runs, L=54) matches the exact
  computation to **2.4e-20** (error) and **4.5e-22** (tail).

**Mutation test.** Replace majority voting by "output the arm of the first instance that
terminated" (wrong instances terminate first by construction): error jumps to **0.9935** at the
largest L, does **not** decay with L (ratio first/last = 0.486, i.e. it *worsens*), and violates
Prop 4.2 at **6/6** grid points. Majority voting is the mechanism, not decoration.

---
