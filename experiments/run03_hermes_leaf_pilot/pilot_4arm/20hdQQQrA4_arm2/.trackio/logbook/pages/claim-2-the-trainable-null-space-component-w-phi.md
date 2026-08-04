## Claim 2 — the trainable null-space component `w_phi`

Route: four-property decomposition (skill P26), in the equality geometry where the formula
is stated exactly; 600 under-determined instances (3 seeds × 200).

| Property | Measured |
|---|---|
| (a) invariance — `A P = b` for arbitrary `w` | max residual **7.85e-14** |
| (b) reachability — solve for the `w` hitting an arbitrary feasible target | max error **2.18e-14** |
| (c) degenerate — `w = 0` equals the fixed orthogonal projection | max diff **0.0** (exact) |
| (d) benefit — optimising over the feasible set beats the fixed foot | **100%** of instances, median relative loss reduction **0.517** |

**Mutation** (delete the term, `w := 0`): reproduces the fixed orthogonal projection
*exactly* on **600 / 600** instances — an exact equality, which is what proves nothing else
contributes. Verdict: **verified**.
