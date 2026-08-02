# Reproduction task — Linear Bandits beyond Inner Product Spaces, the case of Bandit Optimal Transport

OpenReview: https://openreview.net/forum?id=ugjBMARbyt
Area: Theory
Anchored claims (6):

1. The EntUCB algorithm embeds transport plans into a Hilbert space via the Fourier isometry F on L²(ℝᵈ;ϱ), reducing the bandit optimal transport problem to linear bandit estimation in that space (Section 4.1, Equation 7).

2. Theorem 5.1 bounds the entropic regret as R_T^{H,ε}(A) ≤ σ√(2T log(2/δ)) + 2Cβ_T(δ)√(T log det(...)) with probability at least 1-δ (Theorem 5.1).

3. Theorem 5.2 shows that for Lipschitz costs with an entropy penalty decaying as ε_t = αt^{-α}, EntUCB achieves sublinear regret matching classical linear-bandit rates for the Kantorovich optimal transport problem (Theorem 5.2).

4. Corollary 5.3 establishes that in finite-dimensional/discrete settings with N basis coefficients, the algorithm attains Õ(√(NT)) regret, recovering the parametric OFUL-style rate (Corollary 5.3).

5. Corollary 5.4 shows that when basis coefficients decay at rate 1-n^{-q}, regret interpolates between Õ(√T) and Õ(T) depending on the decay exponent q (Corollary 5.4).

6. The confidence sets used by the algorithm are constructed via regularized least-squares in L²(ℝᵈ;ϱ), with width controlled by the log-determinant of the design operator, mirroring the classical OFUL confidence ellipsoid construction (Equations 11-12).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
