# Claim 2 — Eq. (8) trainable null-space component

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_915b4a3d7739", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 2 \u2014 Eq. (8) trainable null-space component"}
-->
*Source:* Section 3.2, Eq. (8); text after Theorem 3.4; Remark 3.6.
*Type:* theory + small optimisation. *Script:* `verify_claim2.py` -> `results/claim2.json`.

**Verdict: verified.**

Over 280 instances (7 shapes x 40 seeds, a third with a forced linearly dependent row, mean
null-space dimension 1.23):
- **T1** `||A_g P_gamma − b_g||_inf <= 1.8e-13` for arbitrary `w_phi` — the null-space term never
  breaks sub-constraint satisfaction.
- **T2** reachability: with `w_phi = y* − f_theta`, `P_gamma = y*` for any `y*` in
  `{y : A_g y = b_g}`, max error **1.7e-14** — the trainable term spans the *entire* set of
  feasible projections on that face, not one point.
- **T3** `w_phi = 0` reproduces the fixed orthogonal (minimum-norm-correction) projection exactly,
  max deviation **1.0e-14**.
- **T4** joint optimisation: on 38 instances with an objective whose optimum lies on a face,
  optimising `w_phi` beat the fixed orthogonal projection on **29/38 = 76.3%**, median relative
  loss reduction **0.847**.

Mutation: deleting the term `(I − A_g^+ A_g) w_phi`. The attainable loss then equals the
`w_phi = 0` loss **exactly on every instance** (`MUT_no_nullspace_equals_w0 = true`) and is strictly
worse than the optimised loss on all 29 improved instances — i.e. the observed gain is produced by
the null-space component and nothing else.
