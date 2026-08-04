## Claim 4 — Lemma 1 (`verify_claim4.py` → `results/claim4.json`)

Split-conformal calibration on the nested family: score `s_i = min{τ : B_i ⊆ K_τ(A_i)}`
(well defined *because* of Claim 2), `τ* = ⌈(n+1)φ⌉/n` empirical quantile, n_cal=50,
4000 trials per φ, heavy-tailed mixture scores with ties (no model probabilities used).

| φ | empirical coverage | δ (binomial 95%) | ≥ φ−δ |
|---|---|---|---|
| 0.70 | 0.712 | 0.014 | yes |
| 0.75 | 0.759 | 0.013 | yes |
| 0.80 | 0.809 | 0.012 | yes |
| 0.90 | 0.898 | 0.009 | yes |
| 0.95 | 0.961 | 0.006 | yes |

* **Mutation (a):** test-time distribution shift (+0.8) breaks exchangeability and drops
  coverage to 0.355 / 0.470 / 0.829 / 0.943 at φ = 0.75 / 0.8 / 0.9 / 0.95 — below φ−δ in
  every case.
* **Mutation (b):** dropping the finite-sample `+1` correction lowers coverage at every φ
  (e.g. 0.919 vs 0.961 at φ=0.95), confirming the correction is load-bearing.

Verdict: **verified**.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 Lemma 1 (`verify_claim4.py` \u2192 `results/claim4.json`)"}\n-->
