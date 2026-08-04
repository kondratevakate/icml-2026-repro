## Claim 1 — Theorem 3.5

Route: structural guarantee (skill §2l) — the theorem quantifies over the layer, not over a
trained network, so no training is needed. A feasible target `f_t` is perturbed by
`||η||_∞ ≤ K` to emulate an unconstrained approximator, pushed through the CAffine layer,
and `||P(f_θ) − f_t||₂ / K` is compared against `3 + 3√n_out`.

Fixed modest setting (budget rule): 200 trials × 3 seeds × 4 `(m, n_out)` cells × 3 values
of K = **2400** instances. No sweep.

- `n_bound_violations = 0`, `max_ratio_over_all = 1.7137897456544158`
- `n_infeasible_caffnet = 0`
- Branch coverage (P25): `n_case2_projected = 706` — half the instances place `f_t` exactly
  on an active constraint so the projection branch is genuinely exercised.
- **Mutation** (HardNet-Aff single pseudo-inverse on the identical instances): **92**
  infeasible outputs, max violation 5.762. Its error ratio never exceeded the bound, which
  is recorded rather than glossed: the mutation separates the arms on *feasibility*, not on
  the approximation constant.

Verdict: **verified**.
