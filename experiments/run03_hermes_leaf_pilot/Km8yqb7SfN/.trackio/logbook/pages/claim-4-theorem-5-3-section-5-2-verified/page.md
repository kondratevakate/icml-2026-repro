# Claim 4 — Theorem 5.3, Section 5.2 → **verified**

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2df3e7f1802a", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 4 \u2014 Theorem 5.3, Section 5.2 \u2192 **verified**"}
-->
Script `verify_claim4.py`, data `results/claim4.json`.

* 60 configs = K ∈ {5,8} × N ∈ {2,4} × β ∈ {0.2,0.5,1.0} × 5 (noise, residual) settings
  ((0,0), (0.05,0), (0.2,0), (0.2,0.05), (0.5,0.1)); T = 20 steps; **200 independent runs each**;
  the LHS is averaged with the theorem's index distribution Prob(t̂ = t) ∝ ((L+β/2)/L)^t.
* ε is measured as the empirical mean of ‖noise‖²_sp and δ as the mean of ‖r_t‖²_sp with
  r_t = −r̃_t + β log(π_{t+1}/π_ref) + (1/η) log(π_{t+1}/π_t) computed from the actual iterates —
  no assumed values.
* Result: the full bound `(β/2)KL[π*‖π₀]/(((L+β/2)/L)^T − 1) + 2(ε+δ)/β` is **never** violated;
  max LHS/RHS ratio **0.2611** over all 60 configs.
* **Mutation** (drop the additive 2(ε+δ)/β term, i.e. test the exact Theorem-5.2 form): violated in
  **36/36** noisy configs, max ratio **1.65e10** — the bias term is necessary, not decorative.
* The measured plateau grows monotonically with the injected noise in **12/12** (K,N,β) cells.
