## Claim 3 — Theorem 4.3 (decaying stepsize, no explicit dimension dependence)
`verify_claim3.py` → `results/claim3.json`. n=64, α_t = a/(t+c₀) with a = 4/η (a = Θ(1/η)), ξ=1, Markov sampling, T=60 000, 5 seeds, d ∈ {2,4,8,16,32}.
- All d converged (final/initial error ratio ≤ 0.027 in every case). Error normalized by ‖θ*‖² — the only quantity the theorem allows to carry d — is flat: 0.0072, 0.0268, 0.0130, 0.0240, 0.0205 for d=2…32; regression **d-exponent 0.285 with R²=0.34**, i.e. no systematic growth (the fit is dominated by seed noise, not a trend), while raw error grows only because ‖θ*‖² grows 0.065→72.9.
- **Mutation:** drop the paper's normalization assumption (entries of φ are O(1) so ‖φ(s)‖=O(√d), the Table-1 note-(4) regime) → normalized error explodes with d, **d-exponent 5.65**, with outright divergence at d=32. So the dimension-freedom is genuinely tied to the stated assumption, exactly as the paper says.
- Verdict: **verified** (claim as stated, under its own normalization assumption).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 3 \u2014 Theorem 4.3 (decaying stepsize, no explicit dimension dependence)"}\n-->
