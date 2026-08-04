# plan.md — reproduction plan for Ir6N7U5Kea (MSNN)

Environment: `.venv` (python 3.12, numpy/scipy/sympy), CPU only. Start 2026-08-01 23:37 +04.

| # | Claim (short) | Type | Feasible on CPU? | Approach |
|---|---|---|---|---|
| 1 | Thm 4.5 bound preserved + Thm 4.6 asymptotic normality, error ~ K^{-1/2} | theory + simulation | yes | implement Alg 2 (MSNN) directly with constructed mixed anchors; Monte-Carlo over many seeds; check (a) studentized statistic ~ N(0,1) (KS test), (b) log-log slope of error sd vs K = -1/2, (c) finite-sample bound of Thm 4.5 is not violated. Mutation: drop the weights w=1/f(d) / break the column-treatment matching. |
| 2 | Cor 4.10 gain factor [sum_d' (p_d'/p_d)^{r+1}]^c | theory (combinatorial probability) | yes | EXHAUSTIVE enumeration of all treatment assignments of the (r+1)x(c+1) anchor block for several (r,c,L,p); exact SNN/MSNN feasibility probabilities vs closed forms of Thm 4.8; ratio vs Cor 4.10. Plus Monte-Carlo count of feasible anchor tuples on a small matrix. Mutation: force column treatments to be identical (MSNN degenerates to SNN) => ratio must become 1. |
| 3 | Cor 4.11 gap reduced from quadratic (rc) to linear (r) order | theory (symbolic) | yes | sympy: derive both ratios from Thm 4.8 expressions, check exponents of (p_d/p_max) are rc+r+c and r; confirm numerically against the exhaustive enumeration of claim 2. Mutation: SNN-vs-SNN ratio must keep exponent rc+r+c. |
| 4 | Table 1 MCAR numbers (p=0.01: MSNN FR 4.69%, MRE 3.91e-2; SNN FR 0.03%, MRE 0.806) | simulation | partially | full setting m=300,n=100,r=3,K=1,10 reps; Alg 3 needs `maxBiclique` (NP-hard, unspecified heuristic in paper). Implement a documented greedy biclique search; report reproduced FR/MRE and compare. Verdict depends on match; the biclique heuristic is not specified in the paper, so a mismatch is `inconclusive`/`toy`, not `falsified`, unless the qualitative ordering fails. |
| 5 | Tables 2-3 MNAR numbers | simulation | same machinery as 4, budget permitting | run only if 4 finishes inside its 2h cap. |
| 6 | Mixed anchors via bipartite cliques preserving target-row same-treatment data; relies on A2.5 | theory + simulation | yes | test the identification mechanism (Lemma 2.6 / Thm 2.7): beta estimated from OTHER treatment levels recovers A_ij^{(d)}; verify constructed anchors satisfy B = 1{D_ab=D_ib, D_aj=d}. Mutation: violate A2.5 (independent row factors per treatment) => estimate must break. |

Priority order: 2, 3 (cheap, exact) -> 1, 6 (simulation) -> 4 -> 5.
No data download is required by any anchored claim (all synthetic; Prop-99 case study is not anchored).
