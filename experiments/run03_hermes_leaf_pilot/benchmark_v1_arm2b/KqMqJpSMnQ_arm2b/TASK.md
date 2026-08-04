# Reproduction task — Compact Conformal Subgraphs

OpenReview: https://openreview.net/forum?id=KqMqJpSMnQ
Area: Optimization
Anchored claims (6):

1. Theorem 1 gives a bicriteria (1+κ, 1+1/κ) approximation via LP rounding: the returned subgraph K satisfies coverage loss W−e(K)≤(1+κ)·ε·W and size |K|≤(1+1/κ)·r (Theorem 1, Section 4.1).

2. Theorem 2 proves a monotonicity/nestedness property (Kτ1⊆Kτ2 for τ1<τ2), derived from a connection to parametric minimum cuts, which is required for valid conformal calibration (Theorem 2, Section 4.2).

3. Corollary 1 shows the entire nested sequence of subgraphs across thresholds can be computed in Õ(γ(m+n)²) time (Corollary 1, Section 4.2).

4. Lemma 1 establishes a distribution-free marginal coverage guarantee ℙ(B*⊆Kτ*(A*))≥φ−δ under exchangeability, without requiring model probability estimates (Lemma 1).

5. Theorem 4 proves NP-hardness of the conformal subgraph problem even for constant ε via reduction from clique detection, in Appendix B (Theorem 4, Appendix B).

6. On a synthetic 6x6 grid navigation experiment with 50 train/test routes, the LP-based method compresses the calibration set to 52 edges at φ=0.75 coverage and outperforms greedy baselines for φ≤0.8 (Section 5).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
