# notes_paper.md — "Robust Bayesian Optimisation with Unbounded Corruptions" (arXiv 2511.15315v2)
Paper read ONCE (arxiv HTML v2 -> paper.txt). This file is the only paper reference from now on.

## Setting
- Maximise f on compact X ⊂ R^d. R_T = Σ_t (f(x*) − f(x_t)). Obs y_t = f(x_t)+ε_t+c_t, ε~N(0,σ_noise²).
- κ = sup_x k(x,x) (kernel boundedness, Assumption 2).
- GP-UCB (Srinivas 2010): x_t = argmax μ_{t−1}+√β'_t σ_{t−1}; R_T = O(√(T β'_T γ_T)). (Sec 2.1, Eq 3)

## Definition 2.1 (Frequency-Constrained Corruption), Sec 2.2 [claim 3]
Adversary picks c_t ∈ R ∪ {∞} AFTER seeing x_t; magnitudes unbounded, only the count is bounded:
T_c = |{t : c_t ≠ 0}|. Appendix H: for a standard GP a single corruption c shifts the posterior mean by
W_t(x,x_c)·c exactly linearly, W_t independent of y ⇒ unbounded derailment (Appendix H).

## RCGP (Sec 2.3, Eqs 5–6)
μ^R_t(x) = k_t(x)^T (K_t + σ_n² J_w)^{-1} (y_t − m_w)
σ^R_t(x)² = k(x,x) − k_t(x)^T (K_t + σ_n² J_w)^{-1} k_t(x)
J_w = diag(σ_n²/(2 w_i²)),  m_w = [σ_n² ∇_y log(w_i²)]_i .
Huber robustness iff sup_y |y|·w(x,y)² < ∞.

## Definition 3.1 (P-IMQ weight), Sec 3
W = σ_noise/√2 ; for centering g and plateau half-width L>0:
w_{L,g}(x,y) = W                                   if |y−g(x)| ≤ L
             = W (1 + (|y−g(x)|−L)²/c²)^{−1/2}      otherwise.
Key identity: within the plateau J_w = I and m_w = 0 ⇒ RCGP update ≡ exact GP update. ("plateau condition")

## Algorithms (Sec 4.2)
Alg 1 FC-RCGP-UCB: g ≡ 0, fixed L_T = √(B_f κ) + N_T(δ/3); β_t = (√β'_t(δ/3) + C_{w,T}√T_c)²;
 x_t = argmax μ^R_{t−1} + √β_t Ψ(T_c) σ^R_{t−1}.
Alg 2 A2-RCGP-UCB: anchor model M_A (g=0, L_A,T = √(B_f κ)+N_T(δ/3)) + acquisition model M_R with
 g_R,t = μ_{A,t−1}, L_R,T = √(β_A,T)√κ Ψ(T_c) + N_T(δ/3); acquisition uses μ_R, β_R,t, Ψ(T_c) σ^R_{t−1}.

## Ψ (Sec 4.1, from Lemma D.11)
Ψ(n) ≜ sqrt( 1 + (nκ/σ_n²)(1 + nκ/σ_n²) )  ⇒ Ψ(0)=1, Ψ(n)=Θ(n) for large n.
Robust multiplier: √β_t = √β'_t + C_w √T_c ; |f−μ^R| ≤ √β_t Ψ(T_c) σ^R_{t−1}. Lemma D.5: |μ^R−μ_uc| ≤ C_w √T_c σ_uc.

## Theorems (Sec 4.3) — the anchored theory claims
- Thm 4.1 (Zero-Cost Robustness) [claim 4]: T_c = 0 ⇒ both algos R_T = O(√(T β'_T γ_T)) (= GP-UCB).
  Proof idea (App F): T_c=0 ⇒ Ψ(0)=1, β_t = β'_t, plateau condition holds ⇒ RCGP ≡ GP.
- Thm 4.2 (FC) [claim 1]: R_T = Õ( Ψ(T_c)(1+√T_c) √(β'_T T (γ_T+T_c)) ).
- Thm 4.3 (A2) [claim 2]: R_T = Õ( (1 + T_c Ψ(T_c)²) √(β'_T T (γ_T+T_c)) ).
- Sublinearity (App D.4.3 / D.5.5 / Table 1, App E), RBF (γ_T=Õ(1), β'_T=Õ(1)):
  FC: R_T = Õ(T_c² √T) sublinear iff 0<α<1/4 ; A2: R_T = Õ(T_c^{7/2} √T) sublinear iff 0<α<1/7.
  "Ideal" (if σ_uc observable, i.e. drop Ψ): FC α<1/2, A2 α<1/3.
  Matérn (γ_T=Õ(T^η)): FC cases1&2 min(1/4,(1−η)/3), case3 min((1−η)/4,(1−2η)/3);
  A2 cases1&2 min(1/7,(1−η)/6), case3 min((1−η)/7,(1−2η)/6).
- C_{w,T} = √2 C_{1,T}/σ_n²; C_1 = O(L + sup_x Δ(x)); FC: C_{w,T}=Õ(1); A2 acquisition model C_{w,R,T}=O(T_c).

## Experiments (Sec 5 + Appendix I) — claim 5
- Sec 5.3 Forrester (1D), σ_noise² = 1, 10 runs/seeds, 5 initial uncorrupted Sobol points,
  30 iterations uncorrupted (Fig 3) and 100 iterations corrupted (Fig 1) with T_c = O(T^{1/3}).
- Adversary (Sec 5.3 main text): clairvoyant, knows x*; ‖x−x*‖ < 0.2 ⇒ corrupt to fixed LOW value;
  ‖x−x*‖ > 0.5 ⇒ corrupt to fixed HIGH value; greedy = applied to the earliest qualifying queries
  until budget exhausted.
- Appendix I.4.2 config (DIFFERENT numbers than Sec 5.3): N_ITERATIONS 30, N_INITIAL 5,
  near_threshold 0.1, far_threshold 0.4, high_value 25.0, low_value −10.0, TIME_BUDGET_ALPHA 1/3,
  USE_ROBUST_HEURISTICS True.
- Baselines (Sec 5.2): GP-UCB; Student-t Process UCB (Shah et al. 2014 closed form); DiagnosticsGP
  (Martinez-Cantin 2018: detect+discard outliers, then standard GP).
- Practical (App I.1): standardize Y; c = 1, L = 1.96 or 95% quantile of |y − median(y)|;
  σ_uc proxied by σ^R; T_c estimated as #observations outside the plateau; authors note setting
  T_c = 0 inside β_t also works well.
- Claimed qualitative outcome (Fig 1): RCGP methods < baselines in cumulative regret under corruption;
  GP-UCB worst; DiagnosticsGP can beat A2 on individual extreme-corruption runs; in uncorrupted setting
  (Fig 3) FC/A2 ≈ GP-UCB and slightly better than Student-t / DiagnosticsGP.
- NO numeric regret values are reported in the text; Figures 1/3 only (no digit-level anchor available).
- Other experiments: CIFAR-10 HPO (140 iters, 4 epochs each, ~8h, GPU), Lunar Lander 36D (300 iters,
  8–12h). Both outside CPU/time budget; not part of the anchored claims.
