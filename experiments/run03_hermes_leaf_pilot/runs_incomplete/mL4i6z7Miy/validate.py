import numpy as np
from eoss_core import (THB_1d, rho_THB_1d, critical_curvature_1d,
                       slow_operator, rho_slow)

rng = np.random.default_rng(123)
eta = 0.1
beta = 0.9
two_over_eta = 2.0 / eta
small_plateau = 2 * (1 - beta) / eta
large_plateau = 2 * (1 + beta) / eta
print(f"eta={eta} beta={beta}  2/eta={two_over_eta}  2(1-beta)/eta={small_plateau}  2(1+beta)/eta={large_plateau}")

# --- Deterministic (full-batch / large-batch) SGDM: expect 2(1+beta)/eta ---
a_star_det = critical_curvature_1d(sigma=0.0, b=1, eta=eta, beta=beta, n_mc=2000, rng=rng)
print(f"[large-batch SGDM, sigma=0]  a* = {a_star_det:.3f}   target 2(1+beta)/eta={large_plateau:.3f}")

# --- vanilla SGD (beta=0), large batch: expect 2/eta ---
a_star_sgd = critical_curvature_1d(sigma=0.0, b=1, eta=eta, beta=0.0, n_mc=2000, rng=rng)
print(f"[large-batch SGD,  beta=0]  a* = {a_star_sgd:.3f}   target 2/eta={two_over_eta:.3f}")

# --- small-batch SGDM: noise-dominated, sweep batch size, watch plateau ---
print("\n--- SGDM critical curvature a* vs batch b (sigma=2) ---")
for b in [2, 4, 8, 16, 64, 256, 1024]:
    a_star = critical_curvature_1d(sigma=2.0, b=b, eta=eta, beta=beta, n_mc=2500, rng=rng)
    print(f"  b={b:5d}  a*={a_star:.3f}")

# --- Theorem 4.1 operator match in noise-dominated regime ---
print("\n--- Theorem 4.1: rho(THB_SGDM) vs rho(I-eta_eff K+eta_eff^2 G) ---")
# 1-D case: K=2a, G=a^2+sigma2_b (scalar); build as 1x1 operators
sigma = 2.0
for b in [2, 8, 64]:
    a = 1.0
    rho_full = rho_THB_1d(a, sigma, b, eta, beta, n_mc=6000, rng=rng)
    sigma2_b = sigma**2 / b
    eta_eff = eta / (1 - beta)
    T_slow = 1.0 - 2 * eta_eff * a + eta_eff**2 * (a**2 + sigma2_b)
    rho_slow_1d = abs(T_slow)
    print(f"  b={b:4d}  rho(THB)={rho_full:.4f}   |1-2 eta_eff a + eta_eff^2(a^2+sigma2_b)|={rho_slow_1d:.4f}")
