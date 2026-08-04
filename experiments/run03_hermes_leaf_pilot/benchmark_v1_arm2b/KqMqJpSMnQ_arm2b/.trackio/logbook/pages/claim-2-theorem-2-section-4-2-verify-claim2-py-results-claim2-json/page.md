## Claim 2 — Theorem 2, Section 4.2 (`verify_claim2.py` → `results/claim2.json`)

Threshold family implemented as the parametric selection problem
`K(λ) = argmax_S Σ_{e⊂S} w_e − λ|S|`, solved exactly by min-cut on the project-selection
network (source→edge, edge→endpoints ∞, vertex→sink λ), with `λ = λ_max(1−τ)`.

Executed: 10 random graphs (n=10–17) × 21-point τ grid.

* Nesting violations with parametric min-cut: **0/200** consecutive pairs; the family is a
  perfect chain.
* **Mutation:** replacing λ by a non-monotone price `λ(1+0.9 sin 6λ)` yields **44**
  nesting violations. A local-search greedy peeling mutation happened *not* to break
  nestedness (it inherits monotone structure), which is recorded as-is.

Verdict: **verified** (the monotone/parametric-min-cut structure is exactly what
generates nestedness).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 Theorem 2, Section 4.2 (`verify_claim2.py` \u2192 `results/claim2.json`)"}\n-->
