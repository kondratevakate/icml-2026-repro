## Claim 4 — Θ(1/p) variables receive variance Ω(p) (Theorem 1.6, §1.3)

**Verdict: inconclusive.**

The theorem is asymptotic (n,m→∞) and was tested at finite n=8,12,16 across p ∈ {0.125,…,1.0}. The optimal allocations are qualitatively consistent with concentration — few high-variance variables at large p, mass spreading across more variables as p shrinks — but the precise asymptotic *signature* (count k of Ω(p)-variance variables with k·p in a constant band, and k non-increasing in p) is **not** uniformly reproduced at these sizes.

Quoted evidence (`results/claim4.json`):
- `"all_signatures_ok": false`
- n=8, k_c25 summary: `"k_times_p": [0.667, 1.0, 1.5, 1.833, 1.5, 0.0]`, `"in_constant_band": false`, `"non_increasing_in_p": true` — k·p wanders outside a constant band.
- n=12, k_c25 summary: `"in_constant_band": true` but `"non_increasing_in_p": false` — the monotonicity signature fails here.
- Qualitative concentration is visible, e.g. n=8, p=1.0, rep=0: allocation `[0.247, 0.247, 0.247, 0.247, 5.6e-3, 5.6e-3, ~0, ~0]` with `k_c25=0, k_c50=0` (only ~4 vars carry meaningful variance), versus n=8, p=0.125, rep=0: `k_c25=5` of 8 variables above threshold.

**Mutation test** (uniform allocation instead of optimal): `"property_breaks": true` — uniform spreads variance across all n variables (k_c50 = n for small p), destroying the concentration. The direction of the theorem is supported, but because the exact constant-band asymptotic is not recovered at finite n, the verdict is **inconclusive**.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 \u0398(1/p) variables receive variance \u03a9(p) (Theorem 1.6, \u00a71.3)"}\n-->
