# Reproduction plan — Compact Conformal Subgraphs (KqMqJpSMnQ)

Paper: Gollapudi, Kollias, Munagala, Vijayaraghavan, *Compact Conformal Subgraphs*, ICML 2026 (arXiv:2602.07530).
Toolchain: CPU-only, numpy/scipy/sympy (no GPU). All seeds pinned.

## Anchored claims to reproduce (6)
1. **Theorem 1** (§4.1, proof App.C): LP rounding → bicriteria `(1+κ, 1+1/κ)` approximation:
   `W - e(K) ≤ (1+κ)·ε·W` and `|K| ≤ (1+1/κ)·r`.
2. **Theorem 2** (§4.2, proof App.D): monotonicity `K_{τ1} ⊆ K_{τ2}` for `τ1<τ2`, derived from
   parametric min-cuts; and the parametric algorithm equals the LP-rounding algorithm.
3. **Corollary 1** (§4.2, Lemma 2 / Gallo-Grigoriadis-Tarjan): the whole nested sequence is
   computed in `Õ(γ(m+n)²)` time. (Verified structurally + empirical scaling; asymptotic bound is the
   cited parametric-max-flow theorem.)
4. **Lemma 1** (§2.1, proof App.A): distribution-free marginal coverage
   `ℙ(B* ⊆ K_{τ*}(A*)) ≥ φ − δ` under exchangeability.
5. **Theorem 4** (App.B): NP-hardness of conformal-subgraph even for constant ε, via reduction from
   CLIQUE. Reproduced as a constructive, instance-checked reduction (sound + complete on test cases).
6. **§5 / App.G.1** synthetic 6×6 grid: LP compresses calibration set to 52 edges at φ=0.75 and beats
   greedy baselines for φ≤0.8.

## Method per claim
- Each claim gets `verify_claim<N>.py` that computes the quantity from first principles, saves
  `results/claim<N>.json`, and runs a MUTATION test (perturb the setup → claimed property must break
  or shift). Verdict ∈ {verified, falsified, toy, inconclusive}, honest.
- `check_reproducibility.py` re-runs all 6 verifiers end-to-end from a pinned seed and asserts the
  recorded verdicts reproduce.

## Libraries
- `lib_hg.py` — hypergraph container, `e(S)`, LP (1) solver via `scipy.optimize.linprog`.
- `lib_flow.py` — Dinic max-flow / min-cut + parametric min-cut network `D_λ` of §4.2.
- `lib_conformal.py` — split-conformal calibration (η scores, τ* quantile), greedy baselines.

## Sources
TASK.md + input_bundle.json (this dir). Paper fetched from arxiv.org/abs/2602.07530 (v2, 27 Mar 2026).
