"""
Adversarial re-verification of CalPro claim 1.

Two knobs the original 5-seed script fixed arbitrarily:
  (A) shift severity: test on [2, 3.5] = ZERO overlap with train/cal support [-2, 2].
      The paper's baselines degrade 15-25 pp; ours degrade ~59 pp, i.e. our shift is
      far outside the paper's own regime. We sweep milder shifts and CALIBRATE the
      shift so that the baseline lands inside the paper's stated 15-25 pp band --
      that is the only apples-to-apples setting for testing CalPro's 5 pp bound.
  (B) NIG training budget (epochs / lr / evidence-regularizer lam).

10 seeds. degradation = max(0, tau - coverage) * 100.
"""
import numpy as np
import torch

from verify_calpro import (sample_data, split_conformal_interval, empirical_coverage,
                           fit_base_predictor, predict)
from verify_calpro_phase2 import train_nig, infer

TAUS = (0.8, 0.9, 0.95)


def degradation(nominal, achieved):
    return max(0.0, nominal - achieved) * 100.0


def one_seed(seed, lo, hi, epochs, lr, lam):
    rng = np.random.default_rng(100 + seed)
    torch.manual_seed(100 + seed)
    xtr, ytr = sample_data(3000, -2.0, 2.0, rng)
    xcal, ycal = sample_data(3000, -2.0, 2.0, rng)
    xsh, ysh = sample_data(5000, lo, hi, rng)

    w = fit_base_predictor(xtr, ytr)
    sv = np.abs(ycal - predict(w, xcal))
    mu_sh_v = predict(w, xsh)

    model = train_nig(xtr, ytr, epochs=epochs, lr=lr, lam=lam)
    mu_cal, sig_cal = infer(model, xcal)
    mu_sh, sig_sh = infer(model, xsh)
    sc = np.abs(ycal - mu_cal) / sig_cal

    dv, dc = [], []
    for tau in TAUS:
        dv.append(degradation(tau, empirical_coverage(ysh, mu_sh_v, split_conformal_interval(sv, tau))))
        dc.append(degradation(tau, empirical_coverage(ysh, mu_sh, split_conformal_interval(sc, tau) * sig_sh)))
    return dv, dc


def sweep(name, configs, n_seeds=10):
    print(f"\n=== {name} ({n_seeds} seeds x {len(TAUS)} taus) ===")
    print(f"{'config':<38} {'base mean':>10} {'CalPro mean':>12} {'CalPro max':>11} {'<=5pp':>7}")
    for label, kw in configs:
        V, C = [], []
        for s in range(n_seeds):
            dv, dc = one_seed(s, **kw)
            V += dv; C += dc
        V, C = np.array(V), np.array(C)
        print(f"{label:<38} {V.mean():10.1f} {C.mean():12.1f} {C.max():11.1f} {str(C.max() <= 5.0):>7}")


if __name__ == "__main__":
    base = dict(epochs=1500, lr=5e-3, lam=1e-2)
    # (A) shift severity sweep, training held at the original setting
    sweep("A: shift severity (train/cal on [-2,2])", [
        (f"test on [{lo},{hi}]", dict(lo=lo, hi=hi, **base))
        for lo, hi in [(-2.0, 2.0), (0.0, 2.0), (1.0, 2.5), (1.0, 3.0),
                       (1.5, 3.0), (2.0, 3.0), (2.0, 3.5)]
    ])
    # (B) NIG training budget, at the ORIGINAL harsh shift
    sweep("B: NIG budget @ harsh shift [2,3.5]", [
        (f"ep={e} lr={lr} lam={lm}", dict(lo=2.0, hi=3.5, epochs=e, lr=lr, lam=lm))
        for e, lr, lm in [(1500, 5e-3, 1e-2), (6000, 5e-3, 1e-2), (6000, 1e-3, 1e-2),
                          (6000, 5e-3, 1e-1), (6000, 5e-3, 1.0), (15000, 5e-3, 1e-1)]
    ])
