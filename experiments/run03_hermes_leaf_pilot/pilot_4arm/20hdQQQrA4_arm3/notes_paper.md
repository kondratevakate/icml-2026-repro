# notes_paper.md — CAffNet (arXiv 2605.24437v1), read ONCE

Paper: "CAffNet: Hard Constraint-Affine Neural Networks", Zhao, Lee, Jeon, Yong. ICML 2026 submission (OpenReview 20hdQQQrA4).
Source text extracted to `paper/full.txt` (from arxiv.org/html/2605.24437v1). DO NOT re-read.

## Core construction (Section 3)
Constraint: A(x) y <= b(x), A(x) in R^{m x n_out}, no rank / cardinality restriction. Assumption 3.2: A,b continuous, S(x)={y: A(x)y<=b(x)} non-empty.

Decomposition (Sec 3.1): index sequences gamma=(j_1<...<j_k), 1<=k<=min(m,n_out);
Gamma = union_{k=1}^{min(m,n_out)} Gamma_k, |Gamma| = sum_{k=1}^{min(m,n_out)} C(m,k) <= 2^m - 1  (Eq. 2).
A_gamma in R^{k x n_out}, b_gamma in R^k (Eq. 5).

Projection (Eq. 8):
  P_gamma(x) = f_theta(x) - A_gamma^dagger(x) (A_gamma(x) f_theta(x) - b_gamma(x))
               + (I - A_gamma^dagger(x) A_gamma(x)) w_phi(x)
w_phi = trainable null-space component (a second NN), shared across gamma (Remark 3.6).

Candidate set (Eq. 9): S_P(x) = { P_gamma(x) : gamma in Gamma, A(x)P_gamma(x) <= b(x) }.
Output (Eq. 12): P*(x) = f_theta(x) if feasible, else argmin_{y in S_P(x)} ||y - f_theta(x)||_p.

Lemma 3.3 (App. A): S(x) non-empty => at least one P_gamma is feasible. Proof: a minimal face F={y: A_{g*}y=b_{g*}} exists with k*<=min(m,n_out) linearly independent active constraints, so g* in Gamma; A_{g*}P_{g*} = A_{g*}A_{g*}^dagger b_{g*} = b_{g*} (consistency: A A^dagger b = b); F subset S => feasible.
Theorem 3.4 (App. B): P*(x) always satisfies A(x)P*(x) <= b(x).
Notes after Thm 3.4: if rank(A_gamma)=n_out then I - A_gamma^dagger A_gamma = 0 (null term vanishes, vertex). w_phi=0 => orthogonal projection. HardNet recovered when m=n_out, A full row rank.

Theorem 3.5 (App. C, Universal Approximation). If F_theta universally approximates F, then F_CAffNet={P*} universally approximates F_target={f_t in F : A(x)f_t<=b(x)}.
Proof constants: choose f_theta with ||f_theta - f_t||_p < K, K = eps / (3 + 3 sqrt(n_out)).
Chain:
 (23) ||f_gamma - f_theta||_p <= ||f_t - f_theta||_p < K  (f_gamma = boundary intersection on segment)
 (28) ||w_phi||_p < 2K   (w_t = 0 admissible; ||w_phi - w_t||_p < K, ||w_t||_p <= ||w_{t,gamma}||_p < K)
 (31) ||A_g^dag A_g||_p <= sqrt(n_out), ||I - A_g^dag A_g||_p <= sqrt(n_out)   (from ||.||_2 <= 1 + norm equivalence)
 => ||f_t - P_gamma||_p < (1 + 3 sqrt(n_out)) K
 => ||P* - f_theta||_p <= ||P_gamma - f_theta||_p < (2 + 3 sqrt(n_out)) K
 => ||P* - f_t||_p < (3 + 3 sqrt(n_out)) K = eps.

## Experiments (Section 4)
Common: baseline soft-NN penalty 100*ReLU(A(x)y-b(x)); FF nets = 3 hidden layers x 200 ReLU; TF = 3 heads of size 40, hidden size 120; Adam lr=1e-4; 2-norm in (12); 5 random seeds; V100 GPU.
Metrics: r = ReLU(A(x)y - b(x)); report max r, mean r, % violated.

### 4.1 Piecewise constraints (App. D.1)  <-- claim 4
n_in = n_out = 1, x in [-2,2]; 50 random training samples (uniform), 400 test samples linspace; MSE loss; 50000 epochs, batch 500.
Target:
 f(x) = -5 sin(pi/2 (x+1)) - 2,      x <= -1
      = -2,                          -1 < x <= 0
      = 2 - 9 (x - 2/3)^2,           0 < x <= 1
      = 3/x^2 - 2,                   x > 1
Upper bounds:
 g1u = -3 sin(pi/2 (x+1)) + 1/5 (x<=-1); -2 (-1<x<=0); 3 - 4(x-1/2)^2 (0<x<=1); 2 (x>1)
 g2u = -3 sin(pi/2 (x+1))^3 + 1 (x<=-1); 2 (-1<x<=0); 3 - 4(x-4/5)^2 (0<x<=1); 2.5 (x>1)
Lower bounds:
 g1l = 5 sin(pi/2 (x+1))^2 - 3 (x<=-1); -2 (-1<x<=0); (4 - 9(x-2/3)^2) x - 5/2 (0<x<=1); 3/x^3 - 5/2 (x>1)
 g2l = 5 sin(pi/2 (x+1))^8 - 2 (x<=-1); -3 (-1<x<=0); (5 - 4(x-1/6)^2) x - 5/2 (0<x<=1); 3/(2x^3) - 16/9 (x>1)
Constraints for CAffNet/NN: A(x) = [1,1,-1,-1]^T (m=4, n_out=1), b(x) = [g1u, g2u, -g1l, -g2l]^T.
=> min(m,n_out) = 1, so Gamma = 4 singletons.
Table 2 reported (mean over 5 seeds, std in parens):
 NN         MSE 0.0045 (0.0057), viol max 0.0074, mean 0.0019, T_train 5.31 ms, T_test 3.37 ms
 HardNet    MSE 0.0037 (0.0063), viol max 0.0033, mean 0.0008, 8.00 / 3.70
 CAffNet-FF MSE 0.0020 (0.0032), viol 0 / 0,       11.93 / 7.11
 CAffNet-TF MSE 0.0012 (0.0009), viol 0 / 0,       15.57 / 14.71
Claimed reduction: 73.33% MSE reduction of CAffNet-TF vs NN (1 - 0.0012/0.0045 = 0.7333).

### 4.2 Optimization solver (not an anchored claim)
min 1/2 y'Qy + p' sin(y) s.t. Gy<=h, Cy=x; n_in=n_eq=3, n_out=n_ineq=5; 1000 train/1000 test; 10000 epochs. IPOPT obj -0.3494; NN -0.1194; HardNet -0.3000; CAffNet-FF -0.3418 (0 violations); CAffNet-TF -0.1576.

### 4.3 Safety-critical control (App. D.3)  <-- claim 5
Unicycle xdot = [[cos t,0],[sin t,0],[0,1]] u, x=[px,py,theta], u=[v,omega].
u = u_nom (PID) + u_net. Saturation on nominal & final command.
State constraints -5<=px<=1, -4<=py<=2, -pi<=theta<=pi:
 A_x = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], b_x = [1,5,2,4,pi,pi]
Control constraints: A_u = [[1,0],[-1,0],[0,1],[0,-1]], b_u = [1, 0.01, 0.5, 0.5]  (-0.01<=v<=1, |omega|<=0.5)
3 polytopic obstacles O_j = {x in R^2: A_j x <= b_j}:
 A1=[[0.4472,-0.8944],[0.7071,0.7071],[-0.2425,0.9701],[-0.7071,-0.7071],[-0.8944,-0.4472]], b1=[-0.2184,-0.5303,0.6219,1.1667,1.4368]
 A2=[[-0.9685,0.2489],[0.9417,0.3363],[-0.3714,0.9285],[0.3714,0.9285],[-0.9417,-0.3363],[-0.2976,-0.9547]], b2=[1.2755,-1.7670,-0.8511,-2.0249,2.3274,2.6868]
 A3=[[-0.9191,0.3939],[0.8944,0.4472],[0.9703,-0.2419],[-0.8701,-0.4930],[0.0000,-1.0000]], b3=[2.9916,-1.9975,-2.5305,2.4854,0.1000]
CBF: h_j^i(x) = a_j^i x - b_j^i >= 0 (outside edge i); smooth union (Molnar & Ames Eq.26)
 h_j(x) = (1/kappa) ln( sum_i exp(kappa h_j^i(x)) ) - eta_j/kappa, kappa=10, eta_j=ln(m_j).
 lambda_j^i = exp(kappa (h_j^i - h_j)); Lf h = sum lambda Lf h^i, Lg h = sum lambda Lg h^i.
CBF constraint: Lf h_j + Lg h_j u >= -alpha(h_j), alpha(h)=h.
Aggregated A(x) = [-Lg h_1; -Lg h_2; -Lg h_3; -Lg h_x; A_u], b(x) = [Lf h_1 + h_1; ...; Lf h_x + h_x; b_u].
Training: 300 random initial states inside state constraints, orientation toward origin; dt=0.1 s; simulate 15 s per epoch;
 J = sum_k (x_k' Q x_k + u_net' R u_net) + x_N' Q_N x_N, Q=diag(1000,1000,0), R=diag(1,1), Q_N=diag(1e6,1e6,0).
Goal (0,0), arrival radius 0.1 m. Test initial state x0 = [-4.5, 0, 0.5]'.
PID: u_nom = [[1,0,0],[0,1,1]] (Kp e + Ki int e + Kd edot), Kp=diag(0.01,0.2,0), Ki=diag(0.05,0.005,0), Kd=diag(0,0.01,0);
 e = R(theta)(x_ref - x) with R = [[cos,sin,0],[-sin,cos,0],[0,0,1]].
Table 4: NN cost 4.1411e5, viol max 0.1348, mean 0.0106, 2.60%; HardNet 4.5701e5, 0.1317, 0.0104, 2.60%; CAffNet-FF 7.2060e5, 0, 0, 0.00%.
Text: NN and HardNet collide with obstacles O1 and O3; CAffNet-FF reaches goal vicinity collision-free. A-posteriori projection (untrained w_phi) gets stuck near an obstacle.

## Anchored claim -> source map
1. Theorem 3.5 + Appendix C bound (3 + 3 sqrt(n_out)) K.
2. Eq. (8) in Section 3.2 (null-space term, joint optimization) + Remark 3.6, text after Thm 3.4.
3. Section 3 / Section 3.1 (Eq. 2, Gamma cardinality <= min(m,n_out)) + Lemma 3.3 + Table 1 row "CAffNet (Ours)"; HardNet full-row-rank limitation stated in Section 1 and Fig. 1.
4. Section 4.1, Table 2 (73.33% MSE reduction, zero violations).
5. Section 4.3, Table 4 + Fig. 6.
