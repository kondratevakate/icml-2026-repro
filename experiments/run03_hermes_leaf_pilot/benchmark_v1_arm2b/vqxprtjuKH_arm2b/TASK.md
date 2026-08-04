# Reproduction task — Allocating Variance to Maximize Expectation

OpenReview: https://openreview.net/forum?id=vqxprtjuKH
Area: Probabilistic Methods
Anchored claims (6):

1. For the independent Gaussian variance allocation problem, the paper gives a PTAS achieving E[max_i X_i] ≥ OPT - ε in polynomial time (Theorem 1.1, Section 1.2).

2. For correlated Gaussian variables, a PTAS with the same additive ε guarantee is established (Theorem 1.2, Section 1.2).

3. For the GraphVarAlloc problem with multiple constraint sets (general m>1), the paper gives an O(log n) multiplicative approximation guaranteeing Ω(1/log n)·OPT (Theorem 1.3, Section 1.2).

4. Theorem 1.6 proves that in the optimal allocation, only Θ(1/p) variables receive variance Ω(p), i.e., the allocation concentrates on a shrinking subset as the constraint parameter p grows (Theorem 1.6, Section 1.3).

5. Lemma 2.1 bounds the contribution of small-variance variables by O(ε√ln(1/ε)), which is used to limit the number of high-variance variables to O(1/ε²) and underlies the PTAS construction (Lemma 2.1, Section 2.1).

6. Monte Carlo simulations on Erdős–Rényi random graphs with n=8 nodes and edge probabilities p ranging from 1/8 to 8/8 are used to illustrate the concentration and concavity results across independent, positively, and negatively correlated settings (Figures 1-2, Section 1.3).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
