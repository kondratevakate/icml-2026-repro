## Claim 2 — Proof mechanism (convexity + mean-preserving spread)

**What the paper claims:** the proof relies on convexity of
`f(p)=max_i(t_i·p_i)` (pointwise max of linear functions) via Jensen's
inequality, because refinement is a mean-preserving spread of this convex
function.

**Reproduction.** (1) Random Jensen tests on `f` over 5000 points: **0**
convexity violations. (2) Per coarse cluster, verified the sub-cluster
predictions form a distribution whose mean equals the coarse prediction
(calibration / mean-preserving spread): **0** failures. (3) Jensen per cluster
`Σ_j λ_j f(p_sub_j) ≥ f(Σ_j λ_j p_sub_j) = f(p_coarse)`: **0** failures, min
margin ≈ 0.

**Mutation test.** (a) Replaced `f` by the non-convex function `min_i(t_i·p_i)`
→ Jensen fails (observed). (b) Broke the mean-preserving spread (sub
predictions no longer average to coarse) → the Jensen inequality fails
(observed). So convexity *and* calibration are both required.

**Verdict: verified.** Source: Theorem 5.1 proof, §5.2.1 / Appendix A.2.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 2 \u2014 Proof mechanism (convexity + mean-preserving spread)"}\n-->
