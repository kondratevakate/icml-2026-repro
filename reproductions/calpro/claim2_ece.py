"""
CalPro claim 2: the evidential head reduces calibration error vs a non-adaptive baseline.
Regression calibration via the probability integral transform (PIT): under a well-calibrated
Gaussian predictive N(mu, sigma^2), pit = Phi((y-mu)/sigma) is Uniform[0,1]. ECE = mean over
levels q of |P(pit <= q) - q|. We compare the evidential adaptive sigma(x) against a
homoscedastic baseline (same mu, constant sigma = calibration RMSE), in-distribution and shift.
"""
import numpy as np
import torch
from verify_calpro import sample_data
from verify_calpro_phase2 import train_nig, infer

def phi(z):
    return 0.5 * (1.0 + torch.erf(torch.tensor(z, dtype=torch.float64) / np.sqrt(2.0))).numpy()

def ece(y, mu, sigma, n_levels=20):
    pit = phi((y - mu) / np.maximum(sigma, 1e-8))
    qs = np.linspace(0.05, 0.95, n_levels)
    return float(np.mean([abs(np.mean(pit <= q) - q) for q in qs]))

def run():
    rng = np.random.default_rng(0)
    xtr, ytr = sample_data(3000, -2.0, 2.0, rng)
    xcal, ycal = sample_data(3000, -2.0, 2.0, rng)
    xte, yte = sample_data(5000, -2.0, 2.0, rng)
    xsh, ysh = sample_data(5000, 2.0, 3.5, rng)

    model = train_nig(xtr, ytr)
    mu_cal, sig_cal = infer(model, xcal)
    mu_te, sig_te = infer(model, xte)
    mu_sh, sig_sh = infer(model, xsh)

    sig_const = float(np.sqrt(np.mean((ycal - mu_cal) ** 2)))   # homoscedastic baseline sigma
    print(f"baseline constant sigma = {sig_const:.3f}\n")
    print(f"{'set':<14}{'ECE baseline':>14}{'ECE evidential':>16}{'reduction':>11}")
    for name, mu, sig, y in [("in-distribution", mu_te, sig_te, yte),
                             ("shifted", mu_sh, sig_sh, ysh)]:
        eb = ece(y, mu, np.full_like(sig, sig_const))
        ee = ece(y, mu, sig)
        red = (eb - ee) / eb * 100 if eb > 0 else 0.0
        print(f"{name:<14}{eb:>14.4f}{ee:>16.4f}{red:>10.1f}%")

if __name__ == "__main__":
    run()
