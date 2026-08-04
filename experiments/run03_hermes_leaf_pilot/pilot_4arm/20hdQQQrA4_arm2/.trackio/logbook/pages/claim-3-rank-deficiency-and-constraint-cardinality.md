## Claim 3 — rank deficiency and constraint cardinality

Route: exhaustive enumeration over a deterministic dictionary of constraint geometries
containing duplicated / negated / rescaled rows, plus a random rank-deficient sweep.

- `n_cases = 705`, `n_rank_deficient = 705`, `n_violations_caffnet = 0` (max violation 0.0).
  The degeneracy count is reported alongside the violation count: zero violations would be
  vacuous if no instance were rank-deficient.
- **Mutation — the prior operator must fail:** HardNet-Aff violates on **57 / 705** cases
  (rate **0.0809**, max violation **17.67**) on the identical instances.
- Cardinality clause: the enumeration size equals `Σ_{k=0..min(m,n_out)} C(m,k)` and
  `kmax == min(m, n_out)` on every cell of a 6 × 5 grid.
- **Mutation — truncate to `min(m,n_out) − 1`:** feasibility failures appear on
  **114 / 450** cases (**25.33%**), so the cardinality bound is load-bearing, not decoration.

Verdict: **verified**.
