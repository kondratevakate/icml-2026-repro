## Claim 2 — Theorem 10 (linear-pool impossibility)
**Source:** Sec. 3.1 Thm 10 (App. E Thm 37).
**Method (`verify_claim2.py`):** re-derived the impossibility as a two-step inequality
`Σ_i β_i Δ_i ≤ −H(P) + Σ_i β_i H(P_i) ≤ 0` (Jensen on `Σβ_i log P_i ≤ log Σβ_i P_i`, then concavity of
entropy). Checked both steps and the conclusion over 6000 random configurations
(\|O\| ∈ 2..6, n ∈ 2..5, Dirichlet concentrations 0.3/1/3), plus 60 Nelder–Mead restarts directly
maximising `min_i Δ_i` over beliefs and weights.
**Result:** max Jensen slack `0.0`, max entropy slack `0.0`, max β-weighted Δ-sum `−2.3e−7`,
max `min_i Δ_i` over random trials `−3.6e−6`, optimiser best `1.7e−21` (numerically zero, attained only
at the degenerate all-agents-identical point). Impossibility holds.
**Mutation:** identical search under the **logarithmic** pool finds 9 strictly unanimous configurations,
best `min_i Δ_i = 0.0571` — the property is specific to linear pooling. The "random dictatorship"
intuition is exactly the Jensen step; note the *mechanism* wording is interpretive, what is verified is
the impossibility itself.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 Theorem 10 (linear-pool impossibility)"}\n-->
