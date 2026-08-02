import numpy as np
from eoss_core import (THB_1d, rho_THB_1d, critical_curvature_1d,
                       sgd_eta_eff_operator_value, THB_ddim, slow_operator,
                       rho_slow, RandomQuadratic, sgdm_step, sgd_step, make_model_spectrum)

eta, beta = 0.1, 0.9
eta_eff = eta / (1 - beta)
print(f"eta_eff = eta/(1-beta) = {eta}/{1-beta} = {eta_eff:.4f}")

# (A) 1-D equivalence: SGDM slow-mode value == SGD(eta_eff) second-moment value (Theorem 4.1)
print("\n[A] Theorem 4.1 slow-mode == SGD(eta_eff) second-moment (1-D, noise-dominated):")
sigma, b = 2.0, 4
max_err = 0.0
for a in [0.5, 1.0, 2.0, 3.0, 5.0]:
    rho_sgdm = rho_THB_1d(a, sigma, b, eta, beta, n_mc=6000, rng=np.random.default_rng(0))
    sgd_val = sgd_eta_eff_operator_value(a, sigma, b, eta_eff)
    err = abs(rho_sgdm - sgd_val)
    max_err = max(max_err, err)
    print(f"   a={a:4.1f}  rho(THB_SGDM)={rho_sgdm:.4f}   SGD(eta_eff) op={sgd_val:.4f}   err={err:.4f}")
print(f"   -> max abs error between SGDM slow-mode and SGD(eta_eff) value: {max_err:.4f}")

# (B) d-dim: rho(THB) vs rho(slow=I-eta_eff K + eta_eff^2 G) in noise-dominated vs deterministic
print("\n[B] d-dim rho(THB_SGDM) vs rho(I-eta_eff K + eta_eff^2 G) across regimes:")
rng = np.random.default_rng(5)
d = 5
m = 3.0          # mean curvature per direction
s = 2.0          # per-sample curvature std
Hbar_diag = np.full(d, m)

def H_sampler_noise():
    # batch Hessian = diag(mean of b samples from N(m, s^2))
    return np.diag(rng.normal(m, s / np.sqrt(b_dim), size=d))

def H_sampler_det():
    return np.diag(np.full(d, m))

# build moments Hbar, G for the noise-dominated case
Hbar = np.diag(Hbar_diag)
# G = E[H o H]; diagonal H_i iid entries ~ N(m, s^2) -> G_{(i,j),(k,l)} = E[H_ii H_kk]...
G = np.zeros((d*d, d*d))
for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                if i == k and j == l:
                    G[i*d+j, k*d+l] = m*m + (s**2 if i == j else 0.0)
                elif i == k and j != l:
                    G[i*d+j, k*d+l] = m*m
                elif i != k and j == l:
                    G[i*d+j, k*d+l] = m*m
                # else 0

b_dim = 4
T_slow = slow_operator(Hbar, G, eta, beta)
rho_slow_val = np.max(np.abs(np.linalg.eigvals(T_slow)))
T_hb, _ = THB_ddim(H_sampler_noise, b_dim, eta, beta, n_mc=1200, rng=rng)
rho_thb = np.max(np.abs(np.linalg.eigvals(T_hb)))
print(f"   noise-dominated (b={b_dim}): rho(THB)={rho_thb:.4f}   rho(slow)={rho_slow_val:.4f}   diff={abs(rho_thb-rho_slow_val):.4f}")

b_dim = 1024
T_slow_det = slow_operator(Hbar, G, eta, beta)   # same G (constructed for noise regime)
rho_slow_det = np.max(np.abs(np.linalg.eigvals(T_slow_det)))
T_hb_det, _ = THB_ddim(H_sampler_det, b_dim, eta, beta, n_mc=1200, rng=rng)
rho_thb_det = np.max(np.abs(np.linalg.eigvals(T_hb_det)))
print(f"   deterministic-ish (b={b_dim}): rho(THB)={rho_thb_det:.4f}   rho(slow)={rho_slow_val:.4f}   diff={abs(rho_thb_det-rho_slow_val):.4f}")
print("   (Theorem 4.1 predicts rho(THB)~rho(slow) ONLY in noise-dominated regime; diff grows in deterministic regime)")

# (C) Small-batch SGDM critical curvature (the predicted 2(1-beta)/eta plateau regime)
print("\n[C] SGDM small-batch critical curvature (predicted ~ 2(1-beta)/eta = %g):" % (2*(1-beta)/eta))
for (sig, bb) in [(1.0, 2), (2.0, 2), (2.0, 4)]:
    a_star = critical_curvature_1d(sig, bb, eta, beta, n_mc=2000, rng=np.random.default_rng(9))
    print(f"   sigma={sig} b={bb}: a*={a_star:.3f}")
