# plan.md — reproduction plan (zlnoC4YPQ1)

Paper: Robust Bayesian Optimisation with Unbounded Corruptions (arXiv 2511.15315v2).
Budget: hard 8h / soft 4h / 2h per claim. Start 2026-08-02 02:14 (+04).

| # | Claim (short) | Type | CPU feasible? | Approach |
|---|---|---|---|---|
| 1 | Thm 4.2 FC bound ⇒ sublinear for α<1/4 | theory (symbolic) | yes | sympy: substitute Ψ(T_c), γ_T=Õ(1), β'_T=Õ(1), T_c=T^α into the stated bound; extract the exponent of T; solve exponent<1. Mutation: drop Ψ ⇒ must give the paper's "Ideal" α<1/2. |
| 2 | Thm 4.3 A2 bound ⇒ sublinear for α<1/7 | theory (symbolic) | yes | same machinery; also Matérn rows of Table 1 as an extra consistency check. Mutation: drop Ψ ⇒ "Ideal" α<1/3. |
| 3 | Unbounded-magnitude corruptions tolerated (Def 2.1) | theory + numeric simulation | yes | numerically evaluate |μ^R(x) − μ_uc(x)| as corruption magnitude |c| → ∞ (up to 1e12) for the P-IMQ RCGP vs a standard GP; check RCGP deviation stays bounded (and the Lemma D.5 bound C_w√T_c σ_uc holds) while GP deviation grows linearly in c. Exhaustive over a grid of magnitudes × several designs/seeds. Mutation: replace P-IMQ weights by constant weights (=plain GP) ⇒ boundedness must fail. |
| 4 | T_c = 0 ⇒ zero-cost robustness, GP-UCB rate (Thm 4.1) | theory + exact numeric | yes | (a) symbolic: Ψ(0)=1 and β_t=β'_t ⇒ both bounds collapse to O(√(Tβ'_Tγ_T)); (b) exact: when the plateau condition holds, J_w=I, m_w=0 ⇒ RCGP posterior == GP posterior to machine precision, so FC/A2-RCGP-UCB produce the *identical* query sequence and identical regret as GP-UCB. Test over many seeds. Mutation: shrink L below the residuals (plateau violated) ⇒ trajectories must diverge. |
| 5 | Forrester experiment: RCGP methods beat GP/Student-t/DiagnosticsGP under O(T^{1/3}) corruption | simulation | yes (own from-scratch CPU implementation; paper's BoTorch code not required) | 1-D Forrester, σ_noise²=1, 5 Sobol init points, 10 seeds; adversary exactly as Sec 5.3/App I.4.2; five methods (GP-UCB, Student-t-P-UCB, DiagnosticsGP, FC-RCGP-UCB, A2-RCGP-UCB); compare mean cumulative regret ± s.e. Also uncorrupted run (Fig 3 setting). Paper gives NO numbers (figures only) ⇒ the reproducible content is the *ordering* of the methods, which is what will be checked. |

Non-anchored experiments (CIFAR-10 HPO ≈8h GPU, Lunar Lander ≈8–12h) are out of scope and out of budget.

Refusal policy: any claim I cannot close with real numbers gets `inconclusive` + reason, never a toy stand-in.
