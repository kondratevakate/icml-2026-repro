## Claim 1 — Theorem 1, Section 4.1 (`verify_claim1.py` → `results/claim1.json`)

Rounding rule reconstructed as `K = {v : x_v ≥ θ}` with `θ = κ/(1+κ)`. Then
`|K| ≤ r/θ = (1+1/κ)r`, and for every lost edge `w−y > w(1−θ)`, so
`Σ_lost w < εW/(1−θ) = (1+κ)εW`.

Executed: 24 random weighted graphs (n=12–21, p=0.2–0.5) × κ∈{0.25,0.5,1,2,4} = **120
instances**, ε taken as the LP-certified loss `(W−LP_opt(r))/W` at budget `r=⌊n/3⌋`.

* Loss-bound violations: **0/120**; size-bound violations: **0/120**.
* Bounds are *tight*: max loss ratio `1.000`, max size ratio `1.000` — i.e. the constants
  (1+κ) and (1+1/κ) cannot be improved by the analysis as stated.
* **Mutation:** rounding at θ/3 violates the size bound in **70/120** cases; rounding at
  an inflated θ violates the loss bound in **32/120** cases. The guarantee is specific to
  θ = κ/(1+κ), as claimed.

Verdict: **verified**.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 Theorem 1, Section 4.1 (`verify_claim1.py` \u2192 `results/claim1.json`)"}\n-->
