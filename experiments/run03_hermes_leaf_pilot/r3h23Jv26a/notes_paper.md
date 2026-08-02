# notes_paper.md — "Questioning the Coverage-Length Metric in Conformal Prediction" (arXiv 2601.21455, OpenReview r3h23Jv26a)

Read ONCE from `results/paper.txt` (rendered from arxiv.org/html/2601.21455v1). Do not re-read.

## Core construction
**Algorithm 1 (Prejudicial Trick, PT)** — input: base CP algo A_{1-α}(·;μ̂), test point x', probability p.
- Draw U ~ Unif[0,1].
- If U > p: C(x') = μ̂(x')  (regression: single point, measure 0) or ∅ (classification).
- Else: α' = 1 − (1−α)/p; C(x') = A_{1-α'}(x'; μ̂).
- Requires p ∈ (1−α, 1).
Eq (2) is the same statement. Remark 5 generalizes: (1−p)α1' + p α2' = α.

**Notation / VCP (Algorithm 2, App. A.4)**: split D into D_tr, D_ca; score s(x,y;μ̂)=|y−μ̂(x)|;
empirical quantile defined with DECREASING order stats: Q̂_τ({Z_i}) := Z_(⌈(n+1)(1−τ)⌉).
Equivalently the ⌈(n+1)(1−α)⌉-th SMALLEST score for miscoverage α. Interval = μ̂(x) ± q → length 2q (constant in x for VCP).

## Theory anchors
- **Theorem 6 (marginal coverage)**: under exchangeability, PT with α' = 1 − (1−α)/p gives
  P(y' ∈ C^PT_{1-α}(X')) ≥ 1 − α. Intuition given in paper: p(1−α') = 1−α.
- Theorem 7: PT preserves conditional coverage of the base.
- Theorem 9: F(p) = p f_A(1−(1−α)/p); if F(1) − F(p) ≥ 1−p then PT's conditional coverage ≥ base's.
- **Lemma 1**: if p·E(L(x, (1−α̃)/p ; s)) < E(L(x, 1−α̃; s)) then E|C^PT| < E|C^CP|.
- Theorem 10 (first-order): E(L/(1−α̃)) > E(∂L/∂α |_{α=1−α̃}) ⇒ ∃p with shorter PT length.
- Corollary 1: holds if E L(x,α) locally concave on α ∈ [0, 1−α̃]. Corollary 2: VCP case (expectation degenerates).
- Corollary 3 (secant): ∃u ∈ (1−α̃,1) with E L(1−α̃)/(1−α̃) > [E L(u) − E L(1−α̃)]/[u−(1−α̃)] ⇒ p = (1−α̃)/u works.
- **Example 3 (failure case)**: if VCP scores are Gaussian, then for ALL α, all p ∈ (1−α,1),
  E|C^PT| > E|C^CP|. Proof reduces to Φ^{-1}(1−α/2) < p Φ^{-1}( (1/2)(1 + (1−α)/p) ) (Eq 54).
- **Proposition 1 (PT ⊂ localized CP)**: localized CP uses Ŝ_norm(x,y) = Ŝ(x,y)/σ̂(x). If σ̂ takes
  only 0+ (w.p. 1−p) and 1 (w.p. p), localized-CP intervals are equivalent to Eq (2), i.e. PT.
- **Definition 1 (Interval Stability)**: IS(C_{1-α}(X)) ≜ E_X[ Var_{A | X, D_ca}( |C_{1-α}(X)| ) ].
  Zero for deterministic methods by design.
- **Proposition 2**: IS(C^PT_{1-α}(X)) = p(1−p) ( E( L(x, (1−α̃)/p ; s) ) )^2 > 0.

## Reported numbers
**Table 1 (synthetic, Example 2 / App. D.1.1)**, 5 seeds ± std err:
| α | p | VCP cov | VCP len | PT cov | PT len |
|---|---|---|---|---|---|
|0.10|0.96|0.906±0.004|22.894±0.138|0.909±0.005|**22.614±0.254**|
|0.10|0.98|same|22.894±0.138|0.904±0.004|22.714±0.165|
|0.20|0.96|0.792±0.011|21.886±0.125|0.799±0.011|21.255±0.136|
|0.20|0.98|same|21.886±0.125|0.796±0.010|21.589±0.149|

Synthetic DGP (D.1.1): Y = Xᵀβ + ε, X ~ N(0, I_2); ε ~ N(μ,1) w.p. 0.5 and N(−μ,1) w.p. 0.5.
Fit training fold with a linear model (Gaussian-noise assumption ⇒ misspecification).
μ = 20, α ∈ {0.1, 0.2}, p ∈ {0.96, 0.98}, 5 seeds. β, n_train/n_cal/n_test NOT specified in the paper.

**Table 2 (real regression, α=0.1, p=0.95, 5 seeds, 3-layer MLP 64-64, Adam lr 5e-4, bs 64, wd 1e-6,
dropout keep 0.1, early stop ≤1000 epochs; bias added to the network output to force misspecification)**:
meps-19 bias20: 42.34±0.228 → 41.92±0.389; meps-20 bias20: 41.98 → 41.41; meps-21 bias20: 42.28 → 41.90;
bike bias10: 20.46 → 19.59; blog-data bias20: 41.67 → 41.13; bio bias10: 21.13 → 20.44;
facebook-1 bias10: 20.81 → 20.80; **facebook-2 bias10: 20.97 → 21.01 (the 1 of 10 that gets WORSE)**;
concrete bias5: 10.32 → 9.87; star bias5: 10.14 → 9.63. Coverage 0.89–0.91 everywhere (nominal 0.90).

**Table 3 (interval stability, same setting)**: VCP = 0.00±0.000 on all 10 datasets;
PT-VCP: meps19 1.26, meps20 1.24, meps21 1.26, bike 0.58, blog 1.23, bio 0.61, facebook-1 0.62,
facebook-2 1.19, concrete 1.14, star 1.14.

## Anchored-claim → source map
1. Algorithm 1 + Theorem 6 (§3.2, §3.3) [+ length claim: Lemma 1 / Thm 10 / Cor 3, §3.4]
2. Table 1, row α=0.10, p=0.96 (22.894 → 22.614); setting App. D.1.1
3. Table 2 (9 of 10 datasets shorter, coverage ≈0.90); setting App. D.2.1/D.2.3
4. Definition 1 + Proposition 2 (§4)
5. Definition 1 + Proposition 1 (§3.2, §4), Remark 3, Remark 12
