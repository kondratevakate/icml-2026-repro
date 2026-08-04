# notes_paper.md — "Fixed Budget is No Harder Than Fixed Confidence in BAI up to Log Factors" (OpenReview DUmWdZetqZ, arXiv 2602.03972)

Read ONCE from `results/paper.txt` (text extracted from arXiv HTML v1). This file is the only
source about the paper from here on.

## Setting
- K arms, arm 1 is the unique best. FC: input delta, stop at random tau, output Jhat.
- (1) delta-correctness: P(Jhat != 1, tau < inf) <= delta.
- (2) high-probability sample complexity T*_delta: P(tau > T*_delta) <= delta.
- FB: fixed budget B, output Jhat, metric P(Jhat != 1).

## Definition 3.1 (Strong FC algorithm)
For any delta in (0,1): delta-correct AND P(tau > T*_delta) <= delta with
**T*_delta = A ln(1/delta) + C**, A, C independent of delta.

## Algorithm 3 (FC2FB) — inputs: budget B, algorithm A, base failure rate delta0, Q <= B/2
- R := floor(log2(B/Q)),  B' := floor(B/R)
- Jhat <- arbitrary arm
- for r = 1..R:  L_r = 2^(R-r);  run A(delta0^{L_r}) with budget cap B'
  - if it did not self-terminate: Jhat_r <- 0 ; else Jhat_r <- its output
  - if Jhat_r != 0: Jhat <- Jhat_r ; break
- Recommended: delta0 = 1/2 or 1/e; Q = 2K ln K in general text, but Q=1 as generic
  recommendation later; guarantees hold as long as Q <= B/2.

## Theorem 3.2 (Correctness of FC2FB)   [claim 1]
For a strong FC algorithm A, given
  B >= 2 ( A ln(1/delta0) + (C+1) ) * ln( 2*(A/Q)*ln(1/delta0) + 2(C+1)/Q ),
  delta0 <= 0.5, Q <= B/2:
    **P(Jhat != 1) <= 3 exp( - B / ( 4Q/ln(1/delta0) + 4 log2(B/Q) * A ) )**
(note the denominator uses log_2(B/Q), and 4Q/ln(1/delta0).)

## Sample complexity corollary of Thm 3.2 (Section 3)   [claim 2]
With delta0 = 1/e, inverting the bound gives sample complexity
  **O( Q ln(1/delta) + A ln(1/delta) * ln( A ln(1/delta) / Q ) + C )**.
Q = A would cancel the extra ln(A); generic recommendation Q = 1, giving
A ln(1/delta) + C up to polylog(A, ln(1/delta)) = polylog(T*_delta) factors.
Practical rec: Q = minimal #samples for A to have any guarantee (Q=K unstructured, Q=d linear).

## Definition 4.1 (Weak FC algorithm)
For SOME delta in (0,1): delta-correct and T*_delta = f(delta), f not necessarily log in 1/delta.

## Algorithm 4 (FCW2S) — inputs: weak alg A^w, number of trials L, base failure rate delta0
- L independent instances; S = all instances (surviving/running set).
- while |S| >= floor(L/2): pick instance in S with smallest internal time step, give it one
  sample (its recommended arm J, reward R); if it terminates, record Jhat_l and remove from S.
- votes v_i = #{l not in S, Jhat_l = i}; output argmax_i v_i.
- Appendix C clarification: once an instance self-terminates no further samples go to it;
  V = set of terminated indices.

## Proposition 4.2 (Correctness of FCW2S)   [claim 3a]
For any delta, any weak FC A^w with delta0 (< 1/(4e)) and L >= 4 ln(1/delta) / ln(1/(4e delta0)):
  P(Jhat != 1, tau < inf) <= exp( -(L/4) ln( 1/(4 e delta0) ) ) = (4 e delta0)^(L/4) <= delta.
Proof route: error requires N >= L/4 incorrectly-terminated trials out of L (Lemma K.2).

## Proposition 4.3 (Strong stopping time of FCW2S)   [claim 3b]
P( tau > L * f(delta0) ) <= exp( -(L/2) ln( 1/(2 e delta0) ) ) = (2 e delta0)^(L/2) <= delta
for L >= 4 ln(1/delta)/ln(1/(4 e delta0)).
=> FCW2S turns a weak FC algorithm into a strong one (T*_delta linear in L, i.e. in ln(1/delta)).

## Algorithm 6 (FC2AT, Appendix D) — anytime, doubling trick   [claim 4]
Inputs: A, delta0, base budget Q. For phase i = 1,2,...: T_i := 2^i * Q; run a fresh
FC2FB(T_i, delta0, A, Q) for T_i samples; at the end of the phase update the standing
recommendation Jhat to that instance's output. No knowledge of the horizon required.
- Def D.1: B* := max{ 2(A ln(1/delta0) + (C+1)) ln( 2(A/Q)ln(1/delta0) + 2(C+1)/Q ), 2Q },
  i* := min{ i >= 1 : T_i >= B* }.
- Def D.2: for T >= max{4B* - 2Q, 2Q}: I_f := max{ k >= 1 : sum_{i<=k} T_i <= T }.
- Prop D.3: T_{I_f} >= B*.   Prop D.4: T_{I_f} >= T/4.
- **Theorem D.5**: for T >= max{4B* - 2Q, 2Q}:
    P(Jhat_T != 1) <= 3 exp( - T / ( 16Q/ln(1/delta0) + 16 log2(T/Q) * A ) ).

## Section 5 applications   [claim 5]
### 5.1 Known heterogeneous noise, Algorithm 5 (PE-KHN)
- S_1 = [K], l = 1; while |S_l| > 1: eps_l = 1/2^l, delta_l = delta/(l(l+1));
  each arm i in S_l sampled to a total of ceil( 2 sigma_i^2 / eps_l^2 * ln(K/delta_l) );
  compute means; S_{l+1} = S_l \ { i : muhat_i <= max_j muhat_j - 2 eps_l }; l++.
- **Theorem 5.1**: with Delta_j <= 1 for all j,
  T*_delta := 64 sigma_1^2/Delta_2^2 * ln( 4K(ln2)^2 ln^2(4/Delta_2) / delta )
            + sum_{j != 1} 64 sigma_j^2/Delta_j^2 * ln( 4K(ln2)^2 ln^2(4/Delta_j) / delta ),
  then P(tau > T*_delta) <= delta.
  => A = 64( sigma_1^2/Delta_2^2 + sum_{j!=1} sigma_j^2/Delta_j^2 ),
     C = sum of the same coefficients times ln(4K(ln2)^2 ln^2(4/Delta_j)).
  Order: O( (sigma_1^2/Delta_2^2 + sum_{i>=2} sigma_i^2/Delta_i^2) ln(K/delta) ).
- **Corollary 5.2** (FC2FB(PE-KHN), delta0 = 1/e, Q = 1): for B >= 2A ln(4A),
    P(Jhat != 1) <= 3 exp( - B / ( 4 + 4 A **ln** B ) ).
  (NB: Theorem 3.2 with delta0=1/e, Q=1 gives denominator 4 + 4 A **log2** B.)
- Separation instance: Delta_2 = K^-5, Delta_i = K^-4 (i>=3), sigma_1^2 = sigma_2^2 = K^-5,
  sigma_i^2 = K^-2 (i>=3). Ours O(K^7); SHVar O(K^9).
- Table 1: SHVar O~(s1^2/D2^2 + sum_i s_i^2/D_2^2); SH O~(s1^2/D2^2 + sum_i (s1^2+s_i^2)/D_i^2);
  FC2FB(PE-KHN) O~(s1^2/D2^2 + sum_i s_i^2/D_i^2).

### 5.2 Unknown variance: Corollary 5.3, FC2FB(VD-BESTARMID),
  A = O( sum_i (sigma_i^2/Delta_i^2 + 1/Delta_i) ln(1/delta) ), same 3exp(-B/(4+4A ln B)) form.

### 5.3 Linear bandits: FC best known complexity gamma* + rho* ln(1/delta) + d
  (Katz-Samuels et al. 2020); their FB alg gives (gamma* + rho*) ln(1/delta) + d.
  **Corollary 5.4**: FC2FB + Fixed Budget Peace (KS20 Alg 3), delta0=1/e, Q=1: exists
  B_0 = Theta~(gamma* + rho* + d) s.t. B >= B_0 => P(Jhat != 1) <= c exp( -B / (Theta~(rho*) ln B) ).
  Also rho* = O~(H_{2,lin}), H_{2,lin} := max_{2<=i<=d} i Delta_i^-2 (Appendix G).

### 5.4 Unimodal bandits
  FB literature best: order Delta^-2 with Delta = min_i |mu_i - mu_{i-1}| (Ghosh et al. 2024).
  FC optimal: T*(mu) ~ sum_{i in N(*)} (mu_* - mu_i)^-2 (Poiani et al. 2024); their Thm 3.7
  gives E[tau_delta] <= T_mu(delta) + 15K.
  Cor 5.5 (Markov): UniTT with constant delta has tau <= (T_mu(delta) + 15K)/delta w.p. >= 1-delta.
  Prop 5.6: FCW2S over UniTT gives strong guarantees.
  **Corollary 5.7**: FC2FB( FCW2S(UniTT, delta1 = 1/(8e), L = ceil(4 ln(1/delta)/ln(1/(4e delta1))) ),
  delta0 = 1/e, Q = 1, budget B ): for some B_0 = Theta~(T_mu(delta1) + K),
    B >= B_0 => P(Jhat != 1) <= 3 exp( - B / ( Theta~(T_mu(delta1) + K) ln B ) ).
  Prop 5.8: Delta^-2 is of higher order than T_mu(delta1) + K up to log factors.

## Reported numbers available for checking
- No experimental section / no empirical tables: this is a pure theory paper. The only
  "numbers" are the bound expressions above plus the K^7 vs K^9 separation instance.
