# Claim 4 — zero-cost robustness at T_c = 0 (Theorem 4.1)

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d95f94d291ec", "created_at": "2026-08-02T04:30:00+00:00", "title": "Claim 4 \u2014 zero-cost robustness at T_c = 0 (Theorem 4.1)"}
-->
**Source:** Theorem 4.1, Sec 4.3; Appendix F; mechanism Definition 3.1/Sec 3.
**Verdict: verified.**

(a) Symbolic: Ψ(0) = **1.0**, and both Theorem 4.2 and Theorem 4.3 bounds collapse at T_c = 0 to
`sqrt(T)*sqrt(betaprime)*sqrt(gamma)` = O(√(T β'_T γ_T)), the GP-UCB rate (both equalities exact).

(b) Exact numeric, uncorrupted Forrester benchmark (Sec 5.3 / Fig 3 setting; 5 initial points,
30 iterations, 10 seeds, σ_noise² = 1): with the plateau condition satisfied (L = 10) and T_c = 0,
FC-RCGP-UCB and A2-RCGP-UCB reproduce GP-UCB **bit-identically**:
max |query difference| over all seeds/iterations = **0.0** for both, max |cumulative-regret
difference| = **0.0** for both; mean cumulative regret **34.9909** for all three algorithms.

**Mutation test:** violating the plateau condition (L = 0.05) makes the FC trajectory diverge from
GP-UCB on **10/10 seeds**, with max |cumulative-regret difference| **34.7054** — i.e. the
equivalence is caused by the plateau, not by coincidence.
