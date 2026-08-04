## Claim 6 — RLS confidence sets with log-det width (`verify_claim6.py` → `results/claim6.json`)

**Anchor:** v2 Eqs. (5)–(7) and Lemma C.2, with Λ = ½‖·‖², DΛ = Id.

**Numbers.** (a) The closed-form RLS estimator f̂ = (V+λI)⁻¹M*C is a stationary point of
L_t + λΛ: numerical gradient norm `3.8e−10`. (b) Sylvester identity underlying the width,
log det(Id + M*M/λ) = log det(Id + MM*/λ): gap `1.8e−15` — the width really is controlled by the
log-determinant of the design operator. (c) Uniform-in-time coverage over 400 independent runs
(d = 8, T = 300, λ = 1, σ = 0.3, δ = 0.05): **1.000** ≥ 1 − δ = 0.95, with β_T = `2.812`.

**Mutation.** Shrinking β_t by a factor 6 destroys validity: coverage `0.000`.

**Verdict: verified.** The construction is an OFUL-style ellipsoid and is (conservatively) valid;
the empirical coverage of 1.00 vs the nominal 0.95 shows the Eq. (7) width is loose, as expected
from a union-bound/self-normalised construction.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 RLS confidence sets with log-det width (`verify_claim6.py` \u2192 `results/claim6.json`)"}\n-->
