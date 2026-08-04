# Reproduction task — Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time

OpenReview: https://openreview.net/forum?id=69IOkVkTQX
Area: Theory
Anchored claims (6):

1. Under the Lévy Mirror Flow model with centered Lévy noise having finite p-th moments (1<p≤2), time-averaged orbits achieve ε-optimality within Õ(ε^(-p/(p-1))) time for convex objectives (Theorem 1).

2. For strongly convex objectives, the process converges geometrically to an uncertainty ball whose radius scales as O(ησ²_tame + η^(p-1)σ^p_heavy), despite jump discontinuities of arbitrary magnitude (Theorem 4).

3. The first-passage (hitting) time to within δ of the optimum for strongly convex objectives is Õ(δ^(-2p/(p-1))) (Theorem 3).

4. The discrete-time Stochastic Dual Averaging analogue achieves an O(T^(-(p-1)/p)) ergodic convergence rate on convex functions, matching the continuous-time rate (Theorem 5).

5. For strongly convex functions, the discrete-time method requires Ω((σ^p/ε)^(1/(p-1)) log(1/ε)) iterations to reach ε-accuracy (Corollary 1).

6. A novel 'weak Itô formula' for Lipschitz-smooth convex functions under Lévy integrators is proved to enable the analysis despite discontinuous sample paths and potentially infinite noise variance when p<2 (Section 2).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
