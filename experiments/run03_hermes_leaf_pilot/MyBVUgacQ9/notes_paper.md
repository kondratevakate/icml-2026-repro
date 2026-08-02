# notes_paper.md — arXiv 2603.11919 (read ONCE, 2026-08-02)

"Last-iterate Convergence of ADMM on Multi-affine Quadratic Equality Constrained Problem"
Chao, Ciebielski, Etesami, Khadiv (TUM). math.OC, 12 Mar 2026. 13 pages main + appendices.
No official code URL found anywhere in the PDF text (grep: no "github", no "code available").

## Problem (eq. 1)
min_{x,z} F(x) + phi(z)  s.t.  A(x) + Q z = 0,
x = (x_1..x_n), x_i in R^{n_i}; Q in R^{nc x nz}; z in R^{nz}.

Def 2.1 (multi-affine quadratic operator), eq. 2:
(A(x))_i = x^T C_i x / 2 + d_i^T x + e_i, and A(x_j; x_{-j}) affine in x_j for fixed x_{-j}
=> diagonal blocks of C_i are zero.

Example 2.2: min x1^2+x2^2+z1^2+z2^2 s.t. x1x2+x1+1+z1=0, -x1x2+x2+1+z2=0;
Q=I, C1=[[0,1],[1,0]], C2=[[0,-1],[-1,0]], d1=(1,0), d2=(0,1), e1=e2=1.

## Assumptions
- Asm 2.3: F subanalytic, F = f(x) + sum_i I_i(x_i); f is C2, mu_f-strongly convex;
  I_i indicator of convex closed X_i. phi is C2, mu_z-strongly convex.
- Def 2.4 L-smooth. Def 2.5 alpha-PL, alpha in (1,2].
- Asm 2.6: Q full ROW rank. (weaker than full column rank of prior work)
- Example 2.8 (Gao et al. 2020): min x^2+y^2 s.t. xy=1 violates Asm 2.6 (here "Q" absent /
  no z-block => row rank fails). From init (x0, 0, w0): (x^k,y^k) -> (0,0) and w^k -> -inf;
  limit point is INFEASIBLE. => used as the negative control for Asm 2.6.

## Algorithm 1 (ADMM), Def 2.7 augmented Lagrangian eq. 3
L(x,z,w) = F(x) + phi(z) + <w, A(x)+Qz> + (rho/2)||A(x)+Qz||^2
loop: for i=1..n: x_i^{k+1} = argmin_{x_i} L(x_{1:i-1}^{k+1}, x_i, x_{i+1:n}^k, z^k, w^k)
      z^{k+1} = argmin_z L(x^{k+1}, z, w^k)
      w^{k+1} = w^k + rho (A(x^{k+1}) + Q z^{k+1})

## Theorem 3.1 (Section 3) — CLAIM 1
Under Asm 2.3 + 2.6, phi L_phi-smooth, and
rho >= max{ 4 L_phi^2 / (mu_phi * lambda_min^+(Q^T Q)) ,
            4 L_phi^2 / (mu_phi * sqrt(lambda_min^+(Q Q^T))) }
ADMM converges with at least SUBLINEAR rate to a stationary point:
   L(x^k,z^k,w^k) - L(x*,z*,w*) in o(1/k).
Limit point is blockwise-optimal (Nash-like): x*_i in argmin_{x_i} f(x_i,x*_{-i}) + I_i(x_i)
s.t. A(x_i,x*_{-i}) + Q z* = 0; z* in argmin phi(z) s.t. A(x*)+Qz=0.

## Theorem 3.2 (Section 3) — CLAIM 2
Assumptions of Thm 3.1 + f is L_f-smooth + L second-order differentiable at limit point.
Condition eq. 4:
||C|| in O( ||(QQ^T)^{-1} Q||^{-1} * min{ m1, m2 lambda_min(QQ^T)^{1/2},
                                           m3 lambda_min(QQ^T)^{1/4} ||(QQ^T)^{-1}Q||^{1/2} } )
with ||C|| := max_i ||C_i||, constants m_i >= 0 depending on L_f, mu_f, ...
Then exists c1 > 1 with L(x^k,z^k,w^k) - L(x*,z*,w*) in O(c1^{-k}); (x*,z*) is a LOCAL MINIMUM.
Note: ||C||=0 recovers linear constraints => known linear convergence (Lin et al. 2015b).

## Theorem 3.3 (Section 3) — CLAIM 3
Assumptions of Thm 3.1 + eq. 4 + {I_i} indicators of POLYHEDRAL sets (no 2nd-order diff needed):
L(x^k,z^k,w^k) - min_{(x,z) in B(x^k,z^k;r)} L(x,z,w^k) in O(c2^{-k}), c2>1, r>0;
lim (x^k,z^k) = (x*,z*) is a local minimum.

## Section 4 — locomotion (eq. 5, eq. 6)
eq. 5: centroidal dynamics, c_{i+1}=c_i+ c_dot_i dt ; c_dot_{i+1} = c_dot_i + sum_j f_i^j/m dt + g dt;
k_{i+1} = k_i + sum_j (r_i^j - c_i) x f_i^j dt ; f_i^j in Omega_i^j (cone -> polyhedral approx).
Multi-affine because of c x f term.
eq. 6: eliminate c, c_dot; new var k'_{i+1} := k_{i+1}-k_i, k'_0 = k_init:
 k'_1 = sum_j ((r_0^j - c_init) x f_0^j) dt
 k'_2 = sum_j ((r_1^j - c_init - c_dot_init dt) x f_1^j) dt
 k'_{i+1} = sum_j (( r_i^j - c_init - c_dot_init i dt
                     - sum_{i'=0}^{i-2} (i-1-i') ( sum_l f_{i'}^l/m + g ) dt^2 ) x f_i^j ) dt , i>=2
=> z := [k'_0..k'_T], x := [x_0..x_T], x_i := [f_i^1..f_i^N]. Q = identity-like (coefficient of k').
KEY: the NONLINEAR (f x f) term carries the factor dt^2 * dt = (dt)^3 => ||C|| ∝ (dt)^3.
Corollary 4.1: sublinear o(1/k) for eq. 6 under Thm 3.1 assumptions with rho large.
Corollary 4.2 — CLAIM 4: with eq. 7 boundedness (||x0||^2, ||x*_f||^2 in O(n_x), ||z*_phi||^2 in O(n_z))
and 2nd-order differentiability at limit: exists c3>1 and t0>0 s.t. for dt <= t0 the iterates satisfy
L^k - L* in O(c3^{-k}); (x*,z*) local minimum. (Appendix E: polyhedral version w/o 2nd-order diff.)

## Section 5 — experiments — CLAIM 5, CLAIM 6
Toy problem (Figure 2):
  min_{x,z} (mu_x/2)(x1^2+x2^2+x3^2+x4^2) + (mu_z/2) z^2
  s.t.  x1 x2 - x3 x4 + q z + 1 = 0.
  "Condition in (4) suggests q >= 10 to ensure linear convergence." Fig 2 shows convergence for
  different q. (mu_x, mu_z, rho values NOT reported in the main text.)
Figure 4: comparison vs PADMM (Yashtini 2021), IPDS-ADMM (Yuan 2025), IADMM (Tang & Toh 2024)
  in three scenarios: (i) convex obj + multi-affine constraint (their method best),
  (ii) convex obj + linear constraint (comparable), (iii) nonconvex obj + linear constraint
  (comparable). 100 iterations, y-axis 1e-15..1e5 (left), 1e-15..1e0 (center), 1e-10..1e2 (right).
  No hyperparameters, no problem instance sizes, no seeds reported.
2D locomotion (Figure 5): m = 2 kg (Solo, Grimminger et al. 2020);
  cost f(f) = 0.5 sum_{i=0}^T ||f_i||^2 + I_i(f_i); phi(z) = 5 sum_{i=0}^T ||k'_i||^2.
  Constraints on f keep CoM in a target area. Corollary 4.2 suggests dt = 0.005 s;
  paper states the bound is CONSERVATIVE — linear convergence observed for much larger dt too.
  Fig 5(right): linear convergence for random initializations. (T, target area not reported.)
Figure 3: humanoid jump dynamic-violation mean/std over 10 trials, 3 values of dt.
Figure 6 — CLAIM 6: real hardware snapshots — humanoid vertical jump + quadruped bounding gait;
  kinematics tracking via DDP (Crocoddyl, Mastalli et al. 2020). No data, no code, no logs released.
