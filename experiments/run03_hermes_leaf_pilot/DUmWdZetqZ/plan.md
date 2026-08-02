# plan.md

Paper: "Fixed Budget is No Harder Than Fixed Confidence in BAI up to Logarithmic Factors"
(OpenReview DUmWdZetqZ / arXiv 2602.03972). **Pure theory paper — no experiments, no datasets,
no GPU anywhere in it.** Therefore every claim is either analytic/symbolic or checkable by an
exact/Monte-Carlo bandit simulation on CPU. No data claim exists to refuse.

Classification and method per anchored claim:

| # | Claim (short) | Type | CPU plan |
|---|---|---|---|
| 1 | Thm 3.2: FC2FB converts FC->FB with error decaying exponentially in B | theory + simulation | Implement Algorithm 3 verbatim. Drive it with (a) an **adversarial worst-case strong FC oracle** that saturates Definition 3.1 exactly, for which FC2FB's error probability is computable in **closed form (exact enumeration over stages, no seeds)**, and (b) Monte Carlo over many seeds. Assert empirical/exact error <= Thm 3.2 bound over a grid of (A, C, Q, delta0, B), and assert log-linearity in B (exponential decay). Mutation: destroy the doubly-exponential delta schedule. |
| 2 | Inverted complexity O(A ln(1/d) ln(A ln(1/d)/Q) + C) matches FC up to log factors | theory (numeric inversion) | Numerically invert the Thm 3.2 bound: B_req(delta) = min B with bound <= delta. Check B_req <= c * (Q ln(1/d) + A ln(1/d) ln(A ln(1/d)/Q) + C) with one constant c over a wide (A, C, Q, delta) grid, and check the ratio B_req / T*_delta is polylog (not polynomial) in T*_delta. Mutation: drop the inner ln factor from the claimed form -> ratio must blow up. |
| 3 | FCW2S (Alg 4) converts weak FC (Def 4.1) -> strong FC (Def 3.1) by parallel runs + majority vote | theory + simulation | Implement Algorithm 4 verbatim over a weak FC oracle saturating Def 4.1. Exactly compute (binomial closed form) and Monte-Carlo P(err) and P(tau > L f(delta0)); assert Prop 4.2 / 4.3 bounds and that the resulting T*_delta is linear in ln(1/delta) (= Def 3.1 strong). Mutation: replace majority vote by "first terminated instance wins". |
| 4 | FC2AT (Alg 6) anytime variant via doubling, no budget knowledge | theory + simulation | Implement Algorithm 6 verbatim. (a) Exhaustively verify Props D.3/D.4 (T_{I_f} >= B*, >= T/4) over integer grids. (b) Simulate anytime error at every T and assert Thm D.5 bound. Mutation: replace doubling by constant phase length. |
| 5 | Improvements shown on heterogeneous-noise (Cor 5.2), linear (Cor 5.4), unimodal (Cor 5.7) bandits | mixed | **5a heterogeneous noise: fully reproducible.** Implement Algorithm 5 (PE-KHN) on Gaussian bandits, verify Theorem 5.1's high-probability sample complexity, plug into FC2FB, check the Cor 5.2 error bound, and check the K^7 vs K^9 separation instance symbolically (sympy). **5b linear / 5c unimodal: the corollaries are analytic consequences of Thm 3.2 given constants (gamma*, rho*, T_mu(delta)) defined in *other* papers** (Katz-Samuels et al. 2020; Poiani et al. 2024) — reproducing them needs those algorithms/constants, out of budget here; verdict recorded honestly rather than substituted with a toy. |

Budget policy: all claims are cheap (seconds to low minutes of CPU). Order: 1, 2, 3, 4, 5.
Every `verified` verdict carries a mutation test in the same script.
