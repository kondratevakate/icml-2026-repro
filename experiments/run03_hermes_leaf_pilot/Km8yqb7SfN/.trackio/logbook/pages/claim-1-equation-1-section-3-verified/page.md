# Claim 1 — Equation (1), Section 3 → **verified**

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_eb1b90d7710f", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 1 \u2014 Equation (1), Section 3 \u2192 **verified**"}
-->
Script `verify_claim1.py`, data `results/claim1.json`, command `.venv/bin/python verify_claim1.py`.

* Non-linearity (the paper's stated distinguishing property of Eq. (1)): for random policies π_a, π_b
  and t on a 19-point grid, `max_t |R[(1−t)π_a+tπ_b] − ((1−t)R[π_a]+tR[π_b])|` over 45 instances
  (K ∈ {4,6,10} × N ∈ {2,4,8} × 5 seeds) lies in **[1.037e-4, 8.139e-2]**, whereas the same measure
  for the untransformed E_π[r] is **2.22e-16** (exactly linear, as the paper states).
* Multi-criterion adaptation: 60 instances (K ∈ {6,10} × N ∈ {2,4,8} × m ∈ {2,3} × 5 seeds) with
  conflicting rewards (r₂ = 1 − r₁), β = 0.05. The base policy trained on Eq. (1) achieves a higher
  Eq.-(1) value than the transform-unaware ("naive") base in **60/60** cases, min gain **5.726e-3**,
  median **4.096e-2**.
* **Mutation** (N = 1, i.e. BoN degenerates to the identity transform, removing the mechanism):
  chord gap collapses to **3.33e-16** (linear again) and the IAMA and naive optima become identical
  (max TV = **0.0** over 10 instances) — exactly as predicted if the non-linearity is caused by T_i.
* Cross-check: mirror descent vs the independent L-BFGS solver agree to TV = 5.16e-9.
