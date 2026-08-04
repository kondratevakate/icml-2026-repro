# notes_paper.md — CAffNet (orid 20hdQQQrA4, arXiv 2605.24437)

## Core construction (Sec 3)
- Constraints: S(x) = {y in R^n_out | A(x) y <= b(x)}, m constraints.
- Decomposition (Sec 3.1): k in [1, min(m,n_out)]; Gamma = union_k Gamma_k of index
  combinations; |Gamma| = sum_{k=1}^{min(m,n_out)} C(m,k) <= 2^m - 1  (Eq 2).
  A_gamma in R^{k x n_out}, b_gamma in R^k (Eq 5).
- Projection (Eq 8):
  P_gamma(x) = f_theta(x) - A_gamma^+ (A_gamma f_theta - b_gamma) + (I - A_gamma^+ A_gamma) w_phi(x)
- Feasible candidate set (Eq 9): S_P(x) = {P_gamma | gamma in Gamma, A P_gamma <= b}.
- Output (Eq 12): P* = f_theta if A f_theta <= b, else argmin_{y in S_P} ||y - f_theta||_p (p=2 used).
- Lemma 3.3: if S(x) nonempty then S_P(x) nonempty (some P_gamma feasible).
- Theorem 3.4: A(x) P*(x) <= b(x) always (under Assumption 3.2). No full-row-rank or
  irredundancy requirement on A(x); HardNet requires full row rank (Sec 1, para "However, it
  assumes that A(x) has full row rank").
- If rank(A_gamma) = n_out (full column rank), I - A_gamma^+ A_gamma = 0 => null-space term
  vanishes, projection = extreme point. HardNet recovered as special case (m = n_out, full row rank).
- Remark 3.6: one shared w_phi across gammas (not per-gamma).

## Theorem 3.5 (Universal approximation) — Appendix C
Statement: X compact, A,b continuous, S(x) nonempty; if F_theta universally approximates F,
then F_CAffNet = {P*} universally approximates F_target = {f_t in F | A f_t <= b}.
Proof constants: given eps>0 set K = eps / (3 + 3 sqrt(n_out)); pick f_theta with
||f_theta - f_t||_p < K; also ||w_phi - w_t||_p < K with ||w_t||_p <= ||f_theta - f_t||_p < K.
Final chain gives ||P* - f_t||_p < (3 + 3 sqrt(n_out)) K = eps.

## Experiment 4.1 — piecewise constraints (Appendix D.1)
- Target f: [-2,2] -> R, n_in = n_out = 1:
  f(x) = -5 sin(pi/2 (x+1)) - 2,           x <= -1
       = -2,                                -1 < x <= 0
       = 2 - 9 (x - 2/3)^2,                 0 < x <= 1
       = 3/x^2 - 2,                         x > 1
- Upper bounds:
  g1u = -3 sin(pi/2 (x+1)) + 1/5 (x<=-1); -2 (-1<x<=0); 3 - 4(x-1/2)^2 (0<x<=1); 2 (x>1)
  g2u = -3 sin(pi/2 (x+1))^3 + 1 (x<=-1); 2 (-1<x<=0); 3 - 4(x-4/5)^2 (0<x<=1); 2.5 (x>1)
- Lower bounds:
  g1l = 5 sin(pi/2 (x+1))^2 - 3 (x<=-1); -2 (-1<x<=0); (4 - 9(x-2/3)^2)x - 5/2 (0<x<=1); 3/x^3 - 5/2 (x>1)
  g2l = 5 sin(pi/2 (x+1))^8 - 2 (x<=-1); -3 (-1<x<=0); (5 - 4(x-1/6)^2)x - 5/2 (0<x<=1); 3/(2x^3) - 16/9 (x>1)
- Constraints for non-HardNet methods: A(x) = [1,1,-1,-1]^T (m=4, n_out=1),
  b(x) = [g1u, g2u, -g1l, -g2l]^T.
- Protocol: 50 random training samples in [-2,2], 400 test samples linspace; MSE loss;
  50000 epochs, batch 500 (=full batch); Adam lr 1e-4; 5 seeds; soft penalty 100*ReLU(Ay-b).
- Architectures: FF = 3 hidden layers x 200 ReLU; TF = 3 attention heads size 40, hidden 120.
- Table 2 (mean (std)):
  NN:         MSE 0.0045 (0.0057) | viol max 0.0074 (0.0074) | viol mean 0.0019 (0.0019)
  HardNet:    MSE 0.0037 (0.0063) | viol max 0.0033 (0.0037) | viol mean 0.0008 (0.0009)
  CAffNet-FF: MSE 0.0020 (0.0032) | viol 0.0000 | 0.0000
  CAffNet-TF: MSE 0.0012 (0.0009) | viol 0.0000 | 0.0000
  Reduction claim: (0.0045 - 0.0012)/0.0045 = 73.33%.

## Experiment 4.3 — safety-critical control (Appendix D.3)
Unicycle robot, 3 convex polytopic obstacles O_j = {x | A_j x <= b_j}; per-edge CBF
h_j^i(x) = a_j^i x - b_j^i >= 0; collision-free = max_i h_j^i(x) >= 0 (smooth softmax used).
Claim: CAffNet-FF reaches goal with zero collisions / zero control-constraint violations;
NN(soft) and HardNet collide with obstacles O_1 and O_3. CAffNet-TF omitted.
Also: a-posteriori projection (no joint training) => robot gets stuck near an obstacle.
