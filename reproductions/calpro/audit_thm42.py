"""
Audit of CalPro Theorem 4.2 / Corollary 4.3 (anchored claim A2).

Corollary 4.3 lower-bounds worst-case coverage under shift by
    1 - alpha - (KL(rho||Pi) + log(1/delta)) / (2 n_cal) - L_s * eps
We check the ONLY data-estimable pieces on the repo's own toy setting:
  - L_s : local Lipschitz constant of the nonconformity score s = |y - mu| / sigma,
          estimated by local finite differences on the calibration set (the paper's
          own stated procedure, Sec 4.4).
  - eps : Wasserstein-1 distance between calibration and shifted test distributions
          (the paper's own stated choice of ambiguity metric, Eq. 10). 1-D W1 is
          the L1 gap between sorted samples -- exact, no scipy needed.
Then we ask: is the bound non-vacuous, i.e. is L_s * eps < 1 - alpha?
"""
import numpy as np
import torch

from verify_calpro import sample_data
from verify_calpro_phase2 import train_nig, infer


def w1(a, b):
    """Exact 1-D Wasserstein-1 via sorted quantile coupling."""
    n = min(len(a), len(b))
    qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def main():
    rng = np.random.default_rng(100)
    torch.manual_seed(100)
    xtr, ytr = sample_data(3000, -2.0, 2.0, rng)
    xcal, ycal = sample_data(3000, -2.0, 2.0, rng)
    xsh, ysh = sample_data(5000, 2.0, 3.5, rng)

    model = train_nig(xtr, ytr)
    mu_c, sig_c = infer(model, xcal)
    s_cal = np.abs(ycal - mu_c) / sig_c

    # L_s by local finite differences among nearest neighbours in (x,y), on supp(D0)
    P = np.stack([xcal, ycal], axis=1)
    order = np.argsort(xcal)
    P, s = P[order], s_cal[order]
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    ds = np.abs(s[1:] - s[:-1])
    ok = d > 1e-9
    ratios = ds[ok] / d[ok]
    L_s = float(np.quantile(ratios, 0.99))   # robust high quantile, not the raw max

    # eps: W1 between calibration and shifted joint (use x and y marginals, summed)
    eps = w1(xcal, xsh) + w1(ycal, ysh)

    print(f"L_s (99th pct of local finite differences) = {L_s:.3f}")
    print(f"eps (W1 cal -> shifted, x + y marginals)   = {eps:.3f}")
    print(f"L_s * eps                                  = {L_s * eps:.3f}")
    for alpha in (0.20, 0.10, 0.05):
        slack = 1.0 - alpha
        print(f"  alpha={alpha:.2f}: nominal 1-alpha={slack:.2f}, "
              f"L_s*eps={L_s*eps:.2f} -> "
              f"{'VACUOUS (bound <= 0)' if L_s*eps >= slack else 'non-vacuous'}")
    print("\n(KL(rho||Pi)/(2 n_cal) is additionally >= 0, so this is a LOWER bound on how")
    print(" much the RHS is degraded; including KL can only make the bound weaker.)")


if __name__ == "__main__":
    main()
