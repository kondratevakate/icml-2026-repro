# notes_paper.md — MSNN (arXiv 2603.11942), read ONCE

## Model / assumptions
- A2.1 low-rank per layer: Y^{(d)} = U^{(d)} V^{(d)T} + E^{(d)}, latent dim r.
- A2.2 E[E|U,V,D]=0.
- A2.4 linear span inclusion on row factors: |I(i)|>=mu => u_i = sum_{l in I} beta_l u_l.
- A2.5 SHARED latent ROW factors across treatments: u_i^{(d)} == u_i. Covers tensor model
  Y_ij^{(d)} = sum_l u_il v_jl lambda_dl, i.e. V^{(d)} = V diag(lambda_d).
- Lemma 2.6: under A2.4+A2.5, I(i) and beta are treatment-irrelevant:
  u_i^{(d')} = sum_l beta_l u_l^{(d')} for ALL d'.
- Thm 2.7: A_ij^{(d)} = sum_{l in I(i)} beta_l E[Ytilde_lj | U,V,D], with D_lj = d for l in I(i).
  => beta may be learned from OTHER treatment levels; only x(d) must be at level d.

## Algorithms
- Alg 1 SNN(i,j,d): anchor rows AR(d), cols AC(d) with D=d on AR x AC, on (AR,j), on (i,AC).
  S^{(k)}=Ytilde[AR x AC]; SVD; beta_hat = (sum_{l<=lambda} 1/tau_l u_l ⊗ v_l) q; Ahat=<x,beta_hat>;
  average over K_SNN disjoint subgroups.
- Alg 2 MSNN(i,j,d): same but MIXED: S_w^{(k)} = [w(b,d(b)) Ytilde_ab], q_w = [w(b,d(b)) Ytilde_ib].
  Requirement: for column b, treatment of the whole column of S equals the treatment D_ib of q_b;
  the target column x(d)=[Ytilde_aj]_{a in MAR} must all be at level d. Weight w(b,d(b)) = 1/f(d(b)).
- Alg 3 MixedAnchorSubMatrix: B = [1{D_ab = D_ib, D_aj = d, a!=i, b!=j}]; createGraph(B);
  maxBiclique -> MAR, MAC. MAC^{(k)}=MAC, MAR split into K equal random subgroups.
  Note: "computationally almost unaffordable when |MAR|,|MAC| exceed 10".

## Theory (Section 4.2)
- Thm 4.5 (finite-sample): with lambda^{(k)}=rank(E[S_w]), w=1/f(d(b)):
  Ahat - A = f(d) * O_p( (1/K) { error + [sum_k ||beta~^{(k)}||_2^2 ]^{1/2} } ),
  error_k1 = r^{1/2}/|MAC|^{1/4}; error_k2 = r^{3/2}||beta~||_1 log^{1/2}(|MAC||MAR|)/min{|MAC|,|MAR|}^{1/2}.
- Thm 4.6 (asymptotic normality): K(Ahat-A)/[sum_k (sigma~^{(k)})^2]^{1/2} -> N(0,1),
  (sigma~^{(k)})^2 = sum_{l in MAR} (beta~_l sigma_lj)^2. Since the denominator grows like sqrt(K),
  the error scales as K^{-1/2}.
- Remark 4.7: identical in form to SNN with MAR/MAC replacing AR/AC.

## Sample efficiency (Section 4.3, MCAR)
- P(D_ij=d)=p_d, P(D_ij=0)=1-sum p_d. p_max = max_d' p_d'. gamma := sum_{d'} (p_d'/p_max)^{r+1}.
- Thm 4.8 (sparsity conditions m r p_d p_max^{alpha c}=o(1), n c gamma p_max^{(1-alpha)r+1+alpha}=o(1)):
  E[K_SNN(d)]  / [ C(m-1,r) C(n-1,c) p_d^{rc+r+c} ]                 in [1-o(1),1]
  E[K_MSNN(d)] / [ C(m-1,r) C(n-1,c) gamma^c p_d^r p_max^{(r+1)c} ] in [1-o(1),1]
- Cor 4.10: E[K_MSNN(d)]/E[K_SNN(d)] = (1+o(1)) [ sum_{d'} (p_d'/p_d)^{r+1} ]^c.
- Cor 4.11: E[K_SNN(d)]/E[K_MSNN(dmax)]  = O( gamma^{-c} (p_d/p_max)^{rc+r+c} )
            E[K_MSNN(d)]/E[K_MSNN(dmax)] = O( (p_d/p_max)^r )
  Remark 4.12: dependence on p_d/p_max reduced from quadratic order rc to linear order r.

## Experiments (Sec 5.1 + Appendix B)
- m=300, n=100, r=3, relative noise sigma=0.001, K=1, 10 repetitions (mean±sd).
- L = {low, medium, high, very high}, scales f(d) = 1, 5, 25, 625.
- MCAR: p(0)=0.115 (unobserved), p(low)=0.01, p(med)=0.025, p(high)=0.05, p(vhigh)=0.8.
- MNAR: |A| used; p_MNAR(d) = exp(lambda A_ij^d)/sum_{d'!=0} exp(lambda A_ij^{d'}), lambda in {0.05,0.02}.
- Feasibility filter: approx x in Col(S) and q in Row(S); drop anchor matrices of size (1,1).
- FR := #feasible / (m n);  MRE := mean over feasible of |(Ahat-A)/A|.

## Reported numbers
- Table 1 (MCAR): SNN low(p=0.01) FR 0.03±0.00, MRE 0.806±0.240; med(0.025) FR 1.20±0.36, MRE 0.577±0.110;
  high(0.05) FR 11.34±0.82, MRE 0.515±0.046.
  MSNN low FR 4.69±1.11, MRE 3.91±1.09e-2; med FR 63.73±5.67, MRE 1.18±0.33e-3;
  high FR 99.29±0.81, MRE 7.05±0.21e-4.
- Table 2 (MNAR, lambda=0.05): SNN FR 0.19/0.38/4.17, MRE 0.349/0.390/0.351;
  MSNN FR 3.13/3.26/4.52, MRE 0.117/0.114/0.106.
- Table 3 (MNAR, lambda=0.02): SNN FR 9.57/11.70/22.66, MRE 0.366/0.379/0.383;
  MSNN FR 26.96/33.88/54.16, MRE 0.129/0.135/0.118.
- Sec 5.2: California Proposition 99 case study (qualitative figure only).
