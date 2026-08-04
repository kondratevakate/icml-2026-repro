# Claim 2 — Algorithm 1 / Proposition 4.2, Section 4 → **verified**

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e8cee45739d7", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 2 \u2014 Algorithm 1 / Proposition 4.2, Section 4 \u2192 **verified**"}
-->
Script `verify_claim2.py`, data `results/claim2.json`.

* Definition 4.1 check: `<∂R/∂π, π′−π>` vs central finite differences of `R[π+ε(π′−π)]`, 80 cases
  (K ∈ {4,6,10,20} × N ∈ {2,3,4,8} × 5 seeds): max abs error **8.285e-10**, max rel error **7.33e-8**.
* "Drop-in on top of standard GRPO": the standard mirror-descent/GRPO update is left untouched and
  only the reward vector is replaced by r̃ = ∂R/∂π. Over 30 instances it reaches the true optimum of
  the non-linear objective: max suboptimality **5.55e-16**, max TV to the independent optimum **1.01e-8**.
* Empirical π̂ (Algorithm 1 line 4, M sampled responses): E‖∂R/∂π[π̂] − ∂R/∂π[π]‖²_sp measured for
  M = 8…1024 (400 reps each) gives a log-log slope of **−1.024** vs M (theory: −1, the Section 5.3
  O(1/M) lemma). At M = 8 (the paper's setting) the mean squared span error is 4.02e-2.
* **Mutation** (feed the plain scalar reward, i.e. plain GRPO): the fixed point is strictly worse on
  the IAMA objective in **30/30** instances, min excess loss **5.471e-3**, median **3.168e-2**.
