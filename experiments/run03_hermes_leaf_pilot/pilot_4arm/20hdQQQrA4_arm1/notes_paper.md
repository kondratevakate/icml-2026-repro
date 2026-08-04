# notes_paper.md — CAffNet (arXiv 2605.24437, orid 20hdQQQrA4)

Read once from `paper/full.txt`. This file is the only paper source afterwards.

## Core objects
- Constraint: A(x) y <= b(x), A ∈ R^{m×n_out}, no rank / cardinality restriction (Eq. 1).
- Decomposition (§3.1): γ = (j_1<...<j_k), 1 <= k <= min(m, n_out); Γ = ∪_k Γ_k,
  |Γ| = Σ_{k=1}^{min(m,n_out)} C(m,k) <= 2^m - 1  (Eq. 2).
- Projection (Eq. 8):
  P_γ(x) = f_θ(x) − A_γ^†(A_γ f_θ − b_γ) + (I − A_γ^† A_γ) w_φ(x).
- Candidate set (Eq. 9): S_P(x) = {P_γ : A P_γ <= b}.
- Output (Eq. 12): P* = f_θ if feasible else argmin_{y∈S_P} ||y − f_θ||_p.
- Lemma 3.3 (App. A): S_P non-empty when S(x) non-empty (minimal face argument,
  k* <= n_out linearly independent active constraints; uses A A^† b = b consistency).
- Theorem 3.4 (App. B): A(x) P*(x) <= b(x) always.
- Theorem 3.5 (App. C): if F_θ universally approximates F then F_CAffNet approximates
  F_target = {f_t ∈ F : A f_t <= b}. Chain, with K = ε/(3+3√n_out):
  * ||f_θ − f_t||_p < K
  * ||w_φ||_p < 2K  (Eq. 28)
  * ||A_γ^† A_γ||_p <= √n_out, ||I − A_γ^† A_γ||_p <= √n_out  (Eq. 31)
  * ||f_t − P_γ||_p < (1 + 3√n_out) K
  * ||P* − f_θ||_p < (2 + 3√n_out) K
  * ||P* − f_t||_p < (3 + 3√n_out) K = ε
- §3.2 remarks: when rank(A_γ)=n_out, I − A_γ^†A_γ = 0 → null-space term vanishes,
  projection = extreme point; w_φ = 0 → orthogonal projection; HardNet recovered
  when m = n_out, A full row rank, all constraints violated.

## Experiments
- §4 setup: penalty 100·ReLU(Ay−b) for soft; FF = 3 hidden layers × 200, ReLU;
  TF = 3 heads of size 40, hidden 120; Adam lr 1e-4; p = 2; 5 seeds; V100 GPU.
- §4.1 piecewise (App. D.1): n_in = n_out = 1, 50 random train samples on [-2,2],
  400 linearly spaced test samples, MSE loss, 50000 epochs, batch 500.
  Target f(x): −5 sin(π/2 (x+1)) − 2 (x<=−1); −2 (−1<x<=0);
               2 − 9(x − 2/3)^2 (0<x<=1); 3/x^2 − 2 (x>1).
  Uppers g1u: −3 sin(π/2(x+1)) + 1/5; −2; 3 − 4(x−1/2)^2; 2.
          g2u: −3 sin(π/2(x+1))^3 + 1; 2; 3 − 4(x−4/5)^2; 2.5.
  Lowers g1l: 5 sin(π/2(x+1))^2 − 3; −2; (4 − 9(x−2/3)^2)x − 5/2; 3/x^3 − 5/2.
          g2l: 5 sin(π/2(x+1))^8 − 2; −3; (5 − 4(x−1/6)^2)x − 5/2; 3/(2x^3) − 16/9.
  A(x) = [1,1,−1,−1]^T, b(x) = [g1u, g2u, −g1l, −g2l]^T  (m = 4, n_out = 1).
  Table 2 (means over 5 seeds): NN MSE 0.0045 (max viol 0.0074), HardNet 0.0037
  (0.0033), CAffNet-FF 0.0020 (0.0000), CAffNet-TF 0.0012 (0.0000).
  Claimed reduction NN→CAffNet-TF: (0.0045−0.0012)/0.0045 = 73.33%.
- §4.3 control (App. D.3): unicycle, state box A_x/b_x = [1,5,2,4,π,π],
  input box A_u = [[1,0],[−1,0],[0,1],[0,−1]], b_u = [1, 0.01, 0.5, 0.5],
  3 polytopic obstacles O_j = {p ∈ R^2 : A_j p <= b_j} with A_1..A_3, b_1..b_3 as
  listed in D.3. PID nominal: u_nom = [[1,0,0],[0,1,1]](Kp e + Ki ∫e + Kd ė),
  Kp = diag(0.01,0.2,0), Ki = diag(0.05,0.005,0), Kd = diag(0,0.01,0),
  e = R(θ)(x_ref − x); x0 = [−4.5, 0, 0.5]. Table 4: NN cost 4.1411e5 (max viol
  0.1348, 2.60%), HardNet 4.5701e5 (0.1317, 2.60%), CAffNet-FF 7.2060e5 (0.0000, 0%).
  The paper does NOT specify the affine safety constraint used per timestep, the
  reference trajectory, the horizon, the dt, or the training loss for §4.3.

## Anchored claims (from input_bundle.json)
1. Theorem 3.5 universal approximation with bound (3+3√n_out)K.
2. Eq. 8 trainable null-space component w_φ (not a fixed orthogonal projection).
3. No full-row-rank requirement; arbitrary cardinality via sub-constraints of size <= min(m,n_out).
4. 73.33% MSE reduction of CAffNet-TF vs soft NN with zero violations (§4.1).
5. Safety-critical control: CAffNet avoids obstacles, HardNet and soft NN do not (§4.3).
