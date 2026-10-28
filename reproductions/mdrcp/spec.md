# MDCP spec (arXiv 2601.02998, Yang & Jin) - transcribed

Common (Sec 5.1): K=3 sources, d=10, alpha=0.1, n_k=2000 each.
X ~ N(0, Sigma), Sigma_ij = 0.2 + 0.8*1{i=j}. Random signal set I, |I|=4.
Split pooled data: train 37.5%, calib 12.5%, test 50%. tau=2.5 in Linear. N=100 runs.

p-value (7): p^(k)(y) = (1 + #{i in calib_k: s_k(Xi,Yi) <= s_k(x,y)}) / (1 + n_calib_k)
Set: C = {y : max_k p^(k)(y) >= alpha}.  (Theorem 1: >= 1-alpha under any mixture)

Baselines:
- Baseline-src-k: standard conformal with calib from source k only.
  classification score = TPS = -phat_k(y|x); regression score = |y-mu_k(x)|/sigma_k(x).
- Baseline-agg (= "naive max-p"): union over k of Baseline-src-k sets.

MDCP: shared score s_k(x,y) = -sum_l lambda_l(x) * phat_l(y|x)  [density fhat_l for regression],
lambda_k(x) = softplus(Lambda(x)^T theta_k), Lambda = cubic B-spline basis (degree 3, 5 knots)
per coordinate. Theta fit by MAXIMIZING (13) on the training fold:
  E_train[ (1 - h(X,Y;Theta))_- / phat_pool(Y|X) ] + (1-alpha) * E_train[ sum_k lambda_k(X) ]
with (t)_- = min(t,0), h = sum_k lambda_k(x) phat_k(y|x). Adam, minibatch, early stopping.

Classification DGP (5.2): C=6 classes. f_k(y=c|x) prop exp(eta_kc),
eta_kc = xi_k*(b_kc + beta_kc^T x) + 1{c>1} g(x); g=0 in Linear.
xi_k = 2.5(1+0.25*tau*u_k), u_k~Unif[-1,1]; b_kc ~ N(0,(0.4 tau)^2);
beta_kc = betabar_c + tau*Delta_kc, (betabar_c)_j~N(0,1), (Delta_kc)_j~N(0,0.15^2) for j in I, else 0.

Regression DGP (5.3): Y = mu_k(X) + eps_k, eps_k~N(0,sigma_k^2).
mu_k(x) = beta_k^T x + b_k + g(x); g=0 in Linear.
beta_k = betabar + 0.2*tau*delta_k, betabar_j~N(0,1), (delta_k)_j~N(0,1) for j in I else 0.
b_k = b + tau*v_k, b~N(0,0.5^2), v_k~N(0,0.5^2).
SNR ~ Unif[5,10] per run, sets sigma_k.
Regression set: grid search Alg 2, M=100 grid on [yL,yU] of train+calib Y,
include j if p(y_j)>=alpha, merge into maximal consecutive blocks, extend each by Delta.
Set size = total length.

Claims: Fig 2 classification MDCP is 34.39% smaller than Baseline-agg.
Fig 5 regression MDCP is 22.44% smaller; MDCP worst-case coverage 90.25%;
Baseline-agg 97.32% avg / 95.94% worst-case.

DEVIATIONS (no sklearn/scipy available): paper uses gradient-boosted trees for
phat_k / mu_k / sigma_k. We use correctly-specified parametric fits instead
(multinomial logistic regression; ridge linear mean + log-residual-squared variance model),
which is appropriate since both settings are Linear. All methods share the same fits,
so the MDCP-vs-Baseline-agg comparison remains apples-to-apples.
