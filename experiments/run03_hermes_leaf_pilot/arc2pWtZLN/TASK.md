# Reproduction task — Collapsed Effective Operators for Higher-order Structures

OpenReview: https://openreview.net/forum?id=arc2pWtZLN
Area: General Machine Learning
Anchored claims (6):

1. The Collapsed Effective Operator S is defined via Schur complement as S := A − X C⁻¹ Xᵀ, marginalizing higher-order cells onto vertices while encoding topology-mediated long-range interactions (Definition 3.3, Section 3.2).

2. The collapsed operator is spectrally bounded between 0 and the rank-0 Laplacian A, i.e. 0 ⪯ S ⪯ A, guaranteeing positive semi-definiteness (Proposition 3.5, Section 3.2).

3. Eigenvalue compression holds for every index k, with λ_k(S) ≤ λ_k(A), meaning the collapsed operator never exceeds the rank-0 Laplacian's spectrum (Corollary 3.6).

4. A regularized variant S_ε := A − X(C + εI)⁻¹Xᵀ is introduced with Tikhonov regularization to bound the collapse error while preserving efficient implicit computation (Proposition 3.10, Algorithm 1, Section 3.4).

5. The graded Laplacian L⋆ remains positive semi-definite only when the coupling weight γ_k satisfies γ_k ≤ β_{k+1}·σ_min⁺(B_{k+1}), a condition on the boundary matrices (Proposition 3.1, Theorem 3.2, Section 3.1).

6. Spectral clustering using the collapsed operator improves accuracy from 46.9% to 70.9% on a protein secondary structure task compared to the baseline rank-0 Laplacian (Section 4).

## Your job (autonomous)
Reproduce each anchored claim on CPU with numpy/scipy/sympy (no GPU). For every claim:
- Write `verify_claim<N>.py` that computes the claim's quantity from first principles.
- Save numeric result to `results/claim<N>.json`.
- Run a MUTATION test (perturb the setup; the claimed property must break or shift) for every verified claim.
- Write verdict: verified / falsified / toy / inconclusive (honest if data/GPU blocks full repro).
- Record exact source (section/equation/theorem) and a reproducible seed.

Hard budget: 8h total, 4h soft, 2h per claim. Write `logbook.md` when done.
