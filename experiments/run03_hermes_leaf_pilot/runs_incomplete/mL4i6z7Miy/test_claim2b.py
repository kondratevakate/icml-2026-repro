import numpy as np
from eoss_core import (rho_THB_1d, Tslow_1d, Tslow_theorem41_1d)

eta, beta = 0.1, 0.9
eta_eff = eta/(1-beta)

def f(x):
    return float(np.asarray(x).item())

print("eta=%.3f beta=%.2f  eta_eff=%.4f  2(1-beta)/eta=%.3f  2(1+beta)/eta=%.3f" %
      (eta, beta, eta_eff, 2*(1-beta)/eta, 2*(1+beta)/eta))

print("\n[1] Schur reduction: rho(THB) == rho(T_slow)?  (should match)")
for (a, sig, b) in [(1.0, 2.0, 4), (2.0, 2.0, 4), (0.5, 1.0, 8)]:
    rho_thb = rho_THB_1d(a, sig, b, eta, beta, n_mc=6000, rng=np.random.default_rng(0))
    tslow = f(Tslow_1d(a, sig, b, eta, beta))
    print(f"   a={a} sig={sig} b={b}: rho(THB)={rho_thb:.4f}  rho(T_slow)={abs(tslow):.4f}")

print("\n[2] Theorem 4.1 closed form vs exact Schur T_slow (small-eta expansion, should match):")
for (a, sig, b, et) in [(1.0, 2.0, 4, 0.05), (2.0, 2.0, 4, 0.05), (1.0, 1.0, 8, 0.03)]:
    ts = f(Tslow_1d(a, sig, b, et, beta))
    t41 = Tslow_theorem41_1d(a, sig, b, et, beta)
    print(f"   a={a} sig={sig} b={b} eta={et}: T_slow={ts:.5f}  Thm4.1 form={t41:.5f}  diff={abs(ts-t41):.2e}")

print("\n[3] eta_eff rescaling: T_slow(SGDM, tiny eta) == SGD(eta_eff) MSS operator")
print("    SGD(eta_eff) 1-D op = 1 - 2*eta_eff*a + eta_eff^2*(a^2+sigma^2/b)")
for (a, sig, b) in [(1.5, 1.0, 16)]:
    ts_small = f(Tslow_1d(a, sig, b, 0.001, beta))
    sgd_form = 1.0 - 2*eta_eff*a + eta_eff**2*(a**2 + sig**2/b)
    print(f"   a={a} sig={sig} b={b}: T_slow(SGDM,eta=0.001)={ts_small:.6f}  SGD(eta_eff) form={sgd_form:.6f}  diff={abs(ts_small-sgd_form):.2e}")
