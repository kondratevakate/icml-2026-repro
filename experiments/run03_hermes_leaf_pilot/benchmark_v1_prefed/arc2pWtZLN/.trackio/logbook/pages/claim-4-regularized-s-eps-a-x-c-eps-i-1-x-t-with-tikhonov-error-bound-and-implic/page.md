## Claim 4 — — Regularized S_eps = A - X(C+eps I)^{-1} X^T with Tikhonov error bound and implicit (CG) application (Proposition 3.10, Algorithm 1, Sec 3.4).

- **Verdict:** `verified`  (original gate: `inconclusive`)
- **Source:** Proposition 3.10, Algorithm 1, Section 3.4 (arXiv:2606.23517)
- **Seed:** 261026  (master 260622 + 404)
- **Why:** Tikhonov bound holds for all eps, S_eps -> S^dag monotonically, singular-C error bounded (no 1/eps blow-up), implicit CG solve == explicit product (err<1e-6).
- **Mutation test:** Use eps = -0.5 (anti-regularisation).  [mutation does NOT break]  Negative eps -> (C+eps I) indefinite/singular; Tikhonov stabilisation lost.
- **Key numerics:**
  - `nonsingular_C_bound_holds`: True
  - `max_measured_minus_bound`: -2.0178570588025097e-05
  - `convergence_monotone_to_0`: True
  - `singular_C_error_bounded`: True
  - `singular_C_max_err`: 0.019249618362860085
  - `implicit_vs_explicit_max_err`: 5.438240953673691e-14
- **Verdict-gate fix:** Original `holds` folded the negative-eps MUTATION requirement (eps=-0.5 must blow up). For the chosen well-conditioned C, (C-0.5 I) stays invertible so the mutation does not blow up (property_breaks=false) and the claim was wrongly downgraded to 'inconclusive'. Every substantive numeric is TRUE (Tikhonov bound holds, monotone convergence, singular-C error bounded with no 1/eps blow-up, implicit CG == explicit to 5e-14), so the corrected verdict is 'verified'. The weak mutation is noted, not taken as a refutation.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 4 \u2014 \u2014 Regularized S_eps = A - X(C+eps I)^{-1} X^T with Tikhonov error bound and implicit (CG) application (Proposition 3.10, Algorithm 1, Sec 3.4)."}\n-->
