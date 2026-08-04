# Reproduction task — Accelerating Regression Tasks with Quantum Algorithms

OpenReview: https://openreview.net/forum?id=TBSyYj4VV6
Area: Theory
Anchored claims (6):

1. The proposed quantum algorithm constructs epsilon-approximate GLM sparsifiers in time O~(r*sqrt(mn)/epsilon + poly(n))*log(s_max/s_min), giving a quadratic speedup in sample count m over the classical O~(mr) algorithm (Theorem 10).

2. For linear regression, the quantum algorithm runs in O~(r*sqrt(mn)/epsilon + n^3) time versus O~(mr + n^3) classically (Corollary 23).

3. The paper gives the first quantum algorithm for Lasso regression, running in O~(r*sqrt(mn)/epsilon + poly(n,1/epsilon)) time versus O~(mn^2 + n^3) classically (Corollary 26).

4. Ridge regression is solved in O~(r*sqrt(mn)/epsilon + n^3) quantum time versus O~(mr + poly(n,1/epsilon)) classically (Corollary 25).

5. Huber regression is handled via a gamma_p-loss framework in O~(r*sqrt(mn)/epsilon + poly(n,1/epsilon)) quantum time (Corollary 12).

6. For l_p regression with p in (0,2], the algorithm achieves quadratic speedup in the sample parameter m, which the authors note dominates runtime when m >> n (Corollary 11).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
