# notes_paper.md — Km8yqb7SfN
"Inference-Aware Meta-Alignment of LLMs via Non-Linear GRPO" (arXiv 2602.01603)
Source read ONCE from arxiv.org/html/2602.01603v1 -> paper.txt. This file is now the only paper reference.

## Eq. (1), Section 3  (claim 1)
R[pi] = g(R_1[pi],...,R_m[pi]) - beta*KL[pi | pi_ref],
R_i[pi] = E_{y~T_i[pi], x~rho}[ r_i(x,y) ].
g may be weighted sum sum_i w_i R_i, worst-case min_i R_i, or smooth-min -1/gamma log(sum w_i exp(-gamma R_i)).
Key stated property: R[pi] is NON-LINEAR in pi even for weighted-sum g, because of the inference-time
transformations T_i; contrast with E_pi[r] which IS linear in pi. A single base pi is meta-trained and
adapted at inference by T_i.

## Section 3.1 + Proposition 3.1  (claim 5)
Y=[0,1]; r_1(y)=1-y^2 (max at y=0), r_2(y)=1-(1-y)^2 (max at y=1). Naive objective
R_naive[pi] = sum_i w_i E_{y~pi}[r_i] - beta KL, with w1=w2=0.5, beta=0 -> optimum pi*_naive = delta_{0.5}.
Prop 3.1: with T_1,T_2 = BoN sampling, N>=2, w1=w2=0.5, beta=0, the IAMA optimum is
  pi*_IAMA(y) = alpha * y^(alpha-1) (1-y)^(alpha-1) / (y^alpha + (1-y)^alpha)^2,  alpha = 1/(N-1).
Proof in Appendix G.1 (not read). Fig 2 shows N=2,4,8; larger N -> stronger bimodality at y=0 and y=1.
Note: for N=2, alpha=1 -> pi* = 1/(y+(1-y))^2 = 1 (uniform).

## Section 4 / Algorithm 1  (claim 2)
Mirror-descent view of TRPO/PPO/GRPO:
  pi_{t+1} = argmin_pi Ltilde[pi] + (1/eta) KL[pi|pi_t],
  Ltilde[pi] = -E_{y~pi_t}[(pi/pi_t) r(y)] + beta KL[pi|pi_ref].
Non-linear GRPO (Alg. 1): identical but the scalar reward r(y) is replaced by the approximated
functional derivative  rtilde_t(y) = dR/dpi[pihat_t](y)  evaluated at the EMPIRICAL distribution pihat_t
of the M sampled responses (M=8 in all experiments, TRL default). "Drop-in reward function on top of
standard GRPO; rest of the algorithm unchanged."
Definition 4.1: first variation dR/dpi[pi](y) defined by d/deps R[pi+eps(pi'-pi)]|_0 = int dR/dpi (pi'-pi).
Defined up to a constant shift.
Chain rule: dR/dpi[pi](y) = sum_i (dg/dR_i) * dR_i/dpi[pi](y).
Proposition 4.2 (BoN): for R[pi]=E_{y~BoN_N[pi]}[r(y)],
  dR/dpi[pi](y) = - int_{r(y)}^{r_max} N * (C[pi](r))^(N-1) dr,   C = CDF of r(y) under pi.
Proposition 4.3 (soft BoN / exponential tilting):
  dR/dpi[pi](y) = r(y)e^{r(y)/tau}/Z - e^{r(y)/tau} * int r e^{r/tau} pi / Z^2,  Z=int e^{r(z)/tau} pi(z)dz.

## Section 5  (claims 3, 4)
Assumption 5.1: R concave and L-smooth relative to KL:
  R[pi'] <= R[pi] + <dR[pi], pi'-pi>            (concavity)
  R[pi'] >= R[pi] + <dR[pi], pi'-pi> - L*KL[pi'|pi]   (relative smoothness)
L[pi] := -R[pi] + beta KL[pi|pi_ref] is then strongly convex rel. KL for beta>0; unique optimum pi*.

Theorem 5.2 (EXACT proximal updates, eta = 1/L):
  L[pi_T] - L[pi*]  <=  beta * KL[pi* | pi_0] / ( ((L+beta)/L)^T - 1 ).

Theorem 5.3 (INEXACT): assume E[|| dR/dpi[pihat_t](y) - dR/dpi[pi_t](y) ||_sp^2] <= eps and
optimization residual r_t := dLhat/dpi[pi_{t+1}] + (1/eta) log(pi_{t+1}/pi_t) with E[||r_t||_sp^2] <= delta.
Span seminorm ||f||_sp = (sup f - inf f)/2. Random index that with Prob(that=t) prop. ((L+beta/2)/L)^t. Then
  E[ L[pi_that] - L[pi*] ]  <=  (beta/2)*KL[pi*|pi_0] / ( ((L+beta/2)/L)^T - 1 )  +  2(eps+delta)/beta.
Lemma (Sec 5.3): for BoN-type objectives E[||dR/dpi[pihat]-dR/dpi[pi]||_sp^2] <= L_f^2 r_max^2 / M
(dimension-independent, O(1/M)).

## Section 6 experiments
6.1 Length reward: r_i(y) = -(|y-L_i|/L_max)^2, L1=50, L2=150, L_max=256; UltraFeedback contexts;
Mistral-7B-Instruct; g = 0.5R1+0.5R2; beta=1e-4; M=8; BoN N=2,4 and soft BoN tau=0.09,0.1.
Baseline standard GRPO on E_pi[(r1+r2)/2]. Result: IAMA base policy bimodal in length, baseline unimodal-medium.
6.2 (claim 6) HH-RLHF (bai2022training), reproduced Alpaca-7B (dai2024safe), objective
w1*R_helpful + (1-w1)*R_harmless with BoN N=4 objectives; reward models = Qwen3-4B Bradley-Terry;
golden judge = Qwen 32B reward models; target KL fixed at 0.1 via log-space proportional controller on beta;
w1 varied. Result: WITH BoN at inference, IAMA pushes the helpfulness/harmlessness Pareto front forward vs
standard alignment; WITHOUT BoN, IAMA is equivalent to (does not outperform) standard alignment. N=8 in App J.3.
No numeric table in main text — result is a Pareto-front figure (Fig. 6b). No public code URL in the HTML
(code stated to be "in the supplementary material").
