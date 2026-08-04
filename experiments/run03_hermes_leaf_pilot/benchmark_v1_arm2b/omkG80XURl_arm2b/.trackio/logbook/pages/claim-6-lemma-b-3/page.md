## Claim 6 — Lemma B.3 (η₁ ≥ ½η₃)
`verify_claim6.py` → `results/claim6.json`. 300 random instances, n∈[4,20), d∈[1,8), varying mixing.
- **0 violations** of η₁ ≥ ½η₃. min ratio η₁/η₃ = **1.042**, median 1.393, max 4.181 — the bound holds with slack, so the constant ½ is not tight on this ensemble (the empirically supported constant is ≥1.04).
- Proof steps re-checked: Dirichlet identity ‖v‖²_Dir = v'D(I−P)v holds to **8.9e-16**; the step λ ≤ 2 holds (max observed λ = 0.783).
- **Mutation A** (non-stationary μ substituted into both definitions): no violations found — the inequality is too loose for this perturbation to break it (reported honestly, min ratio 0.674).
- **Mutation B** (drop the (μ'Φx)² correction term from Eq. 3, with e ∈ span(Φ)): **200/200 instances violate** the bound, min ratio ≈ −2.8e-13 (the Dirichlet-only quantity collapses to 0). The correction term is exactly what makes Lemma B.3 true in the tabular-inclusive case.
- Verdict: **verified**.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 Lemma B.3 (\u03b7\u2081 \u2265 \u00bd\u03b7\u2083)"}\n-->
