# notes_paper.md — FluxNet (arXiv 2602.01941, OpenReview 1KRpajnd6u)

Source read ONCE: arXiv HTML v1 -> text. All facts below are from that single read.

## Core update (Sec 3.2, Eq. 2 / Eq. 6)
u_i^{t+1} = u_i^t - sum_{j in N(i)} F_{i->j} + sum_{j in N(i)} F_{j->i}
- Regular periodic grid; fluxes parameterized per directional offset d in D
  (e.g. 8 neighbors of a 3x3 stencil in 2D); inflow computed by periodic shift (roll)
  of the outgoing flux fields.

## Proposition 1 (Sec 3.2 + Appendix A, Eq. 6-9) — Discrete Conservation
Periodic BC + symmetric stencil (j in N(i) <=> i in N(j)) => sum_i u^{t+1} = sum_i u^t,
for ANY flux field F (independent of how F is produced).
Proof: S_out = S_in by index relabeling.
Reported empirical magnitude: "machine precision", Table 2 max E_cons = 1.19e-7 / 2.38e-7 /
1.79e-7 (float32 conv-diff); Table 3 FluxNet-LAP 3.3e-8 (h), 6.0e-8 (mx), 3.2e-8 (my);
Table 4 FluxNet-D 7.7e-8. Claim 1 anchors "~1e-7 to 1e-8".

## Transport heads (Sec 3.3, Table 1)
- N-head: F = F_hat (signed). Conservation only.
- P-head: F = softplus(F_hat) >= 0. Conservation only.
- L-head (lower bound u >= l): a_i = u_i - l; alpha_i = sigmoid(.) in (0,1);
  pi_{i->i+d} >= 0, sum_d pi = 1 (softmax);  F_{i->(i+d)} = a_i * alpha_i * pi_{i->(i+d)}.
- U-head (upper bound u <= umax): b_i = umax - u_i; beta_i = sigmoid(.) in (0,1);
  rho_{j->i} >= 0, sum_{j in N(i)} rho_{j->i} = 1 (softmax); F_{j->i} = b_i * beta_i * rho_{j->i}.

## Proposition 2 (Sec 3.3) — L-head lower bound guarantee
If u_i^t >= l for all i, one conservative transport update with L-head gives u_i^{t+1} >= l.
Proof: total outflow = a_i*alpha_i < a_i = u_i - l, all inflows >= 0.

## Proposition 3 (Sec 3.3) — U-head upper bound guarantee
If u_i^t <= umax for all i, one update with U-head gives u_i^{t+1} <= umax (periodic BC).
Proof: total inflow = b_i*beta_i < b_i = umax - u_i, all outflows >= 0.

## D-head (Sec 3.4, Eq. 3-4)
Two branches: outflow branch (L-head structure, a_i = u_i - l),
inflow branch (U-head structure, b_i = umax - u_i).
Delta u^out_i = -sum_j F^out_{i->j} + sum_j F^out_{j->i}
Delta u^in_i  = -sum_j F^in_{i->j}  + sum_j F^in_{j->i}
u^{t+1} = u^t + 0.5 (Delta u^out + Delta u^in)                                  (Eq. 3)
L_DCL = (1/|Omega|) sum_i |Delta u^out_i - Delta u^in_i|^2                      (Eq. 4)
Paper text explicitly: "Each branch individually is conservative and satisfies a single-sided
hard bound. However, the averaged update does NOT theoretically guarantee satisfaction of both
bounds unless the two branches agree exactly." Violations reported as first-class metrics;
DCL drives them to near-zero empirically. Conclusion section repeats: D-head = "effective dual
bound control ... transparent violation reporting", not a strict guarantee.

## Training (Sec 3.6, Eq. 5)
Pushforward training, K-step unroll, L = sum_k lambda_k ||u_hat^k - u^k||_1 + gamma L_DCL,
AdamW. Metrics: rollout MAE at T = 2x training horizon, max conservation drift,
bound violation rate V (%), conditional violation magnitude M.

## Reported numbers relevant to the anchored claims
- Table 2 (conv-diff, T=1): N 4.49e-3 / P 3.46e-3 / L 1.82e-3 MAE; E_cons ~1e-7.
- Table 3 (shallow water, T=2, MAE x1e-3):
  ResNet-AR 123 / +SoftCons 104 / +Box+Mass Proj 22.1 / FluxNet-LAP 3.12 (h).
  FNO-AR 7.26 / +SoftCons 15.9 / +Box+Mass Proj 6.74 / FluxNet-LAP(FNO) 2.22.
  E_cons(h): Box+Mass Proj (FNO) 3.3e-7 ; FluxNet-LAP 3.3e-8. V_lb 0.00 for both.
  NOTE: 3.12 (ResNet backbone) vs 6.74 (FNO-backbone projection baseline) — claim 3 compares
  across backbone categories.
- Table 4 (traffic LWR, T=2): ResNet-AR MAE 15.9e-3, E_cons 2.1e-2, V_lb 0.32%, V_ub 3.20%;
  FluxNet-D MAE 3.48e-3, E_cons 7.7e-8, V_lb 0.52%, V_ub 2.87%, M_lb 1.13e-3, M_ub 0.84e-3.
  SigmoidBound+SoftCons: MAE 88.0e-3, 0 violations.
  NOTE: claim 4 quotes "1.87% vs 3.20%" for the upper-bound violation rate of FluxNet-D;
  Table 4 as printed gives V_ub = 2.87% for FluxNet-D (ResNet) and 2.01% for FluxNet-D (FNO).
- Table 5 (spinodal, T=2): 10dt r=3 RF 19^2 ERF 9.7^2 speedup 0.55x MAE 2.76e-2;
  100dt r=5 RF 37^2 ERF 12.2^2 speedup 3.8x MAE 2.16e-2;
  1000dt r=9 RF 79^2 ERF 19.0^2 speedup 17.3x MAE 8.39e-2.
  Speedup is "over GPU-accelerated explicit solvers". Two-point correlation S2bar(r) via FFT
  autocorrelation; all three models comparable to the two-independent-simulation baseline.

## Availability
No code URL, no dataset URL, no repository link appears anywhere in the paper text
(searched for github / zenodo / 4open / "code is available" — zero hits).
