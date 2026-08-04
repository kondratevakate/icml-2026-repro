# plan.md — CAffNet (20hdQQQrA4), arm2b

| Claim | Type | CPU feasible? | Approach |
|---|---|---|---|
| 1 — Thm 3.5 UA bound (3+3√n_out)K | theory | yes | sympy check that K=ε/(3+3√n_out) closes the constant + exhaustive numeric stress of the conclusion with the real Eq (12) algorithm on random polyhedra (target on the boundary). Mutation: argmax instead of argmin candidate selection. |
| 2 — trainable null-space term in Eq (8) | theory + small optimisation | yes | structural identity A_γ P_γ = b_γ for any w; null-space non-degeneracy when rank(A_γ)<n_out; objective value optimised over w vs fixed orthogonal projection (w=0). Mutation: force w=0. |
| 3 — hard satisfaction without full row rank; cardinality ≤ min(m,n_out) | theory | yes | 2100 random instances incl. rank-deficient / duplicated-row A with m≫n_out; compare with HardNet formula; exhaustive check of Γ cardinality/count (Eq 2). Mutation: cap k at min(m,n_out)−1. |
| 4 — 73.33% MSE reduction, zero violations (Table 2) | experiment | yes, reduced | Reimplement Appendix D.1 benchmark in torch CPU (50 train / 400 test points, 5 seeds, Adam 1e-4). Deviation: 20000 epochs instead of 50000 (CPU budget). |
| 5 — safety-critical control: CAffNet avoids obstacles, HardNet & soft NN fail | experiment | partially | Full Appendix D.3 setup (3 polytopes, smooth-union CBF, state/control boxes, unicycle, PID + net correction) with reduced training (12 initial states, 8 epochs, 3 seeds) vs the paper's 300 states. HardNet arm NOT reproducible: A(x) is 13×2, not full row rank → its projection is undefined. |
