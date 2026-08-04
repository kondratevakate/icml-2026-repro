# notes_paper.md — anchored claims ↔ theorems/equations (paper 418BWmKIzX)

Paper: "Efficiently Learning Drifting Halfspaces with Massart Noise"
Ma, Cao, Diakonikolas, Diakonikolas. ICML 2026. arXiv:2606.11149.

Setting: online learner sees history of independent examples whose labels are noisy
versions of a target concept that changes round-to-round (distribution drift rate Δ).
Halfspace h_w(x)=sign(w·x), w∈𝕊^{d-1}. Massart noise: label flipped with prob η(x)≤η<1/2
(indep. across examples). RCN = η(x)≡η. Margin γ: |w·x|≥γ for all x in support.

True error of hypothesis ŵ at round t (target normal w^(t), constant η=RCN):
  err^(t)(ŵ) = η + (1-2η)·P_x(sign(ŵ·x)≠sign(w^(t)·x)).
For x uniform on sphere, P(disagree)=angle(w^(t),ŵ)/π. So excess beyond η = (1-2η)·angle/π.

## Claim 1  ↔ Theorem 1.1 (Sec 1.1; Alg 1 + Alg 2 in Sec 3)
- Guarantee: err^(T)(ĥ) ≤ η + Õ(Δ^{1/3}/γ), for T=Ω̃(Δ^{-2/3}), prob ≥ 0.9.
- Algorithm 1 DriftedMassart: epoch length W = 2⌊Δ^{-2/3} log(Δ^{-1})⌋. Each epoch collects
  W examples, then updates hypothesis via DriftPerceptron, discards buffer.
- Algorithm 2 DriftPerceptron(S,η,γ,μ): w^(0)=e1. Split S (size 2m) into S1 (first m), S2.
  For i=1..m: g = ((1-2η)·sign(w·x) − y)·x / max(|w·x|, γ);
  w ← proj_{𝔹(1)}(w − μ g); h_i = sign(w·x). Return argmin_{h_i} empirical err on S2.
  μ = γ/√T (T=W/2) per proof (Eq. proof step: μ=γ/√T).
- Proof balance (Sec 3, proof of Thm 1.1): optimization error 1/(√T γ) + drift error O(TΔ/γ)
  + concentration. Min over T → T=Θ(Δ^{-2/3}) gives excess Õ(Δ^{1/3}/γ).

## Claim 6  ↔ Theorem 3.2 (Sec 3, realizable case)
- Guarantee: err^(T)(ĥ) = Õ(√Δ · γ^{-3/2}), for T=Ω̃((γΔ)^{-1/2}), prob ≥ 0.9.
- Realizable gradient (no label noise): g=(sign(w·x)−y)·x (NO (1-2η) factor, NO margin weight).
- Epoch length W=O((γΔ)^{-1/2}) (vs HL94 W=O(Δ^{-1/2})). Improvement: γ-exponent −3/2
  instead of −2. (Paper consistently states √Δ γ^{-3/2}; task paraphrase "Δ γ^{-3/2}" drops √.)

## Claim 3  ↔ Theorem 2.1 (Sec 2; Alg 3 in App B.1)
- Info-theoretic UPPER bound (any VC-dim d Boolean class, general Massart): excess
  ≤ opt_t + Õ(√((dΔ)/(1-2η))). Epoch length W=Θ̃(√(d·Δ^{-1}·(1-2η)^{-1})).
- Comparison: for dΔ<1, √(dΔ) < (dΔ)^{1/3} → Massart rate BETTER than adversarial Θ((dΔ)^{1/3}).

## Claim 4  ↔ Theorem 2.2 (Sec 2; App B.2)
- LOWER bound for halfspaces with η-RCN: provided (1-2η)^3 > dΔ, NO algorithm achieves
  err ≤ opt_T + o(√(dΔ/(1-2η))) with prob ≥ 1/2 for every instance in the family.
- Construction (B.2): marginal D_x: pick i~Unif[d], g~Unif(0,1), x=g·e_i.
  h^(z)_t(x)=sign((Σ_i z_i e_i·x) − (1 − Δ t d^{-1}(1-2η)^{-1})), z∈{0,1}^d uniform.
  m = √(d/(Δ(1-2η)))/20. Two regret bounds used:
  * For general γ-margin lower bound (Thm B.3): d=Θ(1/γ²), excess Ω(√Δ/γ), cond (1-2η)²>dΔ.
- Lemma B.1 (coin-flipping minimax): to predict a coin with bias 1/2+ε using
  ≤1/(100ε²) flips per coin, Pr[‖Ŝ−S‖₁≥d/4] ≥ 1/2. This forces ≥d/8 wrong bits →
  excess error Ω(√(dΔ/(1-2η))).

## Claim 5  ↔ Theorem 4.1 (Sec 4)
- Trajectory-testing problem (Def 4.1/4.3). Hard instance: examples on {±1}^d, γ=Θ(1/d),
  η=1/3. H0: labels independent, P(y=1)=1−η. H1: for last m=T−(T−m) steps, target drifts
  along random direction v, P(h=−1)=Δ j, noise (η−Δj)/(1−2Δj) < η.  m=γ^{-1/6}Δ^{-2/3}.
- Low-degree distinguisher: p(z) polynomial; 1-distinguisher if
  |E_{H0}p − E_{H1}p| ≥ √(Var_{H0} p).
- Theorem 4.1: for c∈(0,1/2), η=1/3, Δ>2^{-1/γ^c}, NO polynomial of degree < O(γ^{-c/4})
  is a 1-distinguisher. Proof: ‖E_D(D̄^{≤∞,k})−1‖² = Σ_{A,|A|≤k} E_{u,v}[∏ I_j] ≤
  Σ_{ℓ=1}^k C(m,ℓ)(1/(100m))^ℓ ≤ 1 for k=γ^{-c/4}. Lemma 4.2 bounds
  I_j(u,v) ≤ Õ(d^{-(1/2-c)}(Δj)²) (off-diag ‖u·v‖≤d^{1-c}), I_j(u,u)≤O(Δj) (diag).

## Claim 2  ↔ Theorem 1.2 (Sec 1.1; implication of Theorem 4.1)
- Under the low-degree hardness conjecture, no poly-time algorithm achieves error
  < opt_T + Δ^{1/3}γ^{-1/6} for the RCN family → excess Ω(Δ^{1/3}), matching the algorithm.
- Information-computation gap: info-theoretically optimal error scales Δ^{1/2} (Claim 3/4),
  but computationally efficient algorithms need Δ^{1/3} (Claim 5/Thm 4.1). The conjecture
  itself is assumed; the Δ^{1/3} computational barrier is independently reproduced in Claim 5.
