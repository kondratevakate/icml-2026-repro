## Claim 1 — Theorem 3.5 universal approximation bound

**Source:** Theorem 3.5 + proof in Appendix C (chain: Eq. 23 → Eq. 28 → Eq. 31 →
(1+3√n_out)K → (2+3√n_out)K → (3+3√n_out)K = ε, with K = ε/(3+3√n_out)).

**Scripts:** `verify_claim1.py` (broad sweep) and `verify_claim1b.py` (targeted; this is the
primary evidence — see the caveat below). Raw: `results/claim1.json`, `results/claim1b.json`.

Instances are constructed to satisfy exactly the hypotheses of the proof: f_t feasible,
‖f_θ−f_t‖_p < K = 0.01, ‖w_φ‖_p < 2K (Eq. 28). Grid: n_out ∈ {1..5}, m ∈ {2,3,5,7},
p ∈ {1,2,3}, rank-deficient A, 200 seeds per cell (exhaustive over the grid, seeds only as
the inner fallback).

**Numbers (`claim1b.json`)**
- Genuine Case-2 instances (f_θ infeasible, the non-trivial branch): **9241**
- `main_bound_violations` = **0**; `main_worst_ratio` = **0.4003** (realised error is at most
  40% of the claimed bound → the bound holds and is not tight)
- Intermediate bound (1+3√n_out)K at the intersection γ of the segment [f_θ, f_t]:
  **0 / 8866** violations
- Eq. 31 matrix-norm bounds ‖A_γ^†A_γ‖_p, ‖I−A_γ^†A_γ‖_p ≤ √n_out (`claim1.json`):
  **0** violations over 6000 instances

**Mutation test.** Replacing the selection rule of Eq. 12 (argmin over the feasible candidate
set) with argmax — keeping every other component identical — produces
**3545 / 9241 bound violations**, worst ratio **3.09e4**. The bound therefore depends on the
mechanism the theorem specifies, not on the sampling regime.

**Two honest caveats.**
1. In the first sweep (`claim1.json`) almost all instances had f_θ still feasible (Case 1,
   where the bound is trivial), so its "0 / 6000 violations" is weak evidence on its own;
   `claim1b.py` was written to force Case 2 and is what the verdict rests on.
2. A second mutation in `claim1.json` (inflating ‖w_φ‖ to 50K, i.e. breaking Eq. 28) did
   **not** break the final bound (0 violations). Reason: Eq. 12 discards distant candidates,
   so the overall bound survives a violated Eq. 28 in this regime. That is a finding about
   the proof's slack, not a counterexample to the theorem.

**Verdict: verified** (numerically, over the sampled instance space; this is not a machine-checked proof).

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 1 \u2014 Theorem 3.5 universal approximation bound"}\n-->
