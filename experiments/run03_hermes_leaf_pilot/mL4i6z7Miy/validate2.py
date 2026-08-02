import numpy as np
from eoss_core import (THB_1d, rho_THB_1d, critical_curvature_1d,
                       sgd_eta_eff_operator_value, rho_THB_sgd,
                       RandomQuadratic, sgdm_step, sgd_step, make_model_spectrum)

rng = np.random.default_rng(7)
eta, beta = 0.1, 0.9
two_over = 2/eta; small_p = 2*(1-beta)/eta; large_p = 2*(1+beta)/eta
print(f"eta={eta} beta={beta}: 2/eta={two_over}  2(1-beta)/eta={small_p}  2(1+beta)/eta={large_p}")

# 1) Deterministic SGDM (sigma=0) -> heavy-ball threshold 2(1+beta)/eta
a_det = critical_curvature_1d(sigma=0.0, b=1, eta=eta, beta=beta, n_mc=2000, rng=rng)
print(f"[C4] deterministic SGDM a* = {a_det:.2f}  target {large_p:.2f}")

# 2) Vanilla SGD deterministic -> 2/eta
a_sgd = critical_curvature_1d(sigma=0.0, b=1, eta=eta, beta=0.0, n_mc=2000, rng=rng)
print(f"[C1-SGD] vanilla SGD a* = {a_sgd:.2f}  target {two_over:.2f}")

# 3) SGDM critical curvature vs batch (interpolation) and SGD batch-independence
print("\n[C1] SGDM a* vs batch b (sigma=1.0):")
for b in [2, 8, 64, 1024, 4096]:
    a = critical_curvature_1d(sigma=1.0, b=b, eta=eta, beta=beta, n_mc=1200, rng=rng)
    print(f"   b={b:5d}  a*={a:7.3f}")
print("[C1] vanilla SGD a* vs batch b (sigma=1.0) [should stay ~2/eta]:")
for b in [2, 8, 64, 1024, 4096]:
    a = critical_curvature_1d(sigma=1.0, b=b, eta=eta, beta=0.0, n_mc=1200, rng=rng)
    print(f"   b={b:5d}  a*={a:7.3f}")

# 4) Theorem 4.1: SGDM small-batch slow-mode MSS == SGD with step eta_eff
print("\n[C2/C3] SGDM slow-mode value vs SGD(eta_eff) over a (noise-dominated, b small):")
eta_eff = eta/(1-beta)
sigma, b = 2.0, 4
for a in [0.5, 1.0, 1.5, 2.0, 3.0]:
    # SGDM via full THB (4x4) and the slow scalar
    rho_sgdm = rho_THB_1d(a, sigma, b, eta, beta, n_mc=8000, rng=rng)
    sgd_val = sgd_eta_eff_operator_value(a, sigma, b, eta_eff)  # = E[(1-eta_eff h_b)^2] = SGD MSS operator with step eta_eff
    print(f"   a={a:.2f}  rho(THB_SGDM)={rho_sgdm:.4f}   SGD(eta_eff) op value={sgd_val:.4f}")

# 5) Empirical dynamical check: SGD / SGDM on random quadratic, track Batch Sharpness
print("\n[C5/C6] dynamical Batch-Sharpness tracking")
eigs = np.array([35., 8., 4., 2., 1., 0.5, 0.3, 0.2, 0.1, 0.05])
model, Hbar = make_model_spectrum(eigs, seed=11)
print("   Hbar top eigenvalue:", np.sort(np.linalg.eigvalsh(Hbar))[-1])
for name, b, opt in [("SGD b=4", 4, "sgd"), ("SGDM b=4", 4, "sgdm"), ("SGDM b=512", 512, "sgdm")]:
    rng2 = np.random.default_rng(3)
    x = rng2.normal(scale=0.5, size=model.d)
    v = np.zeros(model.d)
    bs_hist = []
    for t in range(1500):
        if opt == "sgd":
            x, g = sgd_step(x, model, b, eta, rng2)
        else:
            x, v = sgdm_step(x, v, model, b, eta, beta, rng2)
        if t % 150 == 0:
            bs = model.batch_sharpness(x, b, n_mc=80, rng=rng2)
            bs_hist.append(bs)
    print(f"   {name}: BS plateaus ~ {np.mean(bs_hist[-5:]):.2f}  (predicted: SGD->2/eta={two_over}, SGDM small->2(1-beta)/eta={small_p}, SGDM large->2(1+beta)/eta={large_p})")
