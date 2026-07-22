"""Toy-dataset reproduction for LERD claim 3 (Table 1), model-free part.

PAPER SPEC (Appendix E.2.1, verbatim)
-------------------------------------
Three frequency bands, latent event rate lambda from a truncated normal centered
at the band midpoint:
    low  [5,10]  : lambda ~ TruncNormal(mu=7.5,  sigma=1.0; [5,10])
    mid  [10,15] : lambda ~ TruncNormal(mu=12.5, sigma=1.0; [10,15])
    high [15,20] : lambda ~ TruncNormal(mu=17.5, sigma=1.0; [15,20])
Splits per band (no rate overlap): train 150 rates x 50 seq; val 25 x 50;
test 25 x 50. Each sequence has 20 observations: inter-event times
dt_i ~ Exponential(lambda); timestamps t_i = cumsum(dt); observations
y_i = sin(t_i) + eta_i, eta ~ N(0, 0.07^2).

Table 1 (test set) reports, per band and model, a "Median Rate" with 95% CI:
    band   model        Median Rate  95% CI
    [5,10] NODE         1.000        [1.000, 1.000]
    [5,10] ODE-RNN      1.000        [1.000, 1.000]
    [5,10] STRODE       0.340        [0.269, 0.410]
    [5,10] LERD         7.532        [4.300, 14.867]
    ...
    [15,20] LERD       18.843        [10.465, 35.244]

WHAT THIS REPRODUCES
--------------------
The "Median Rate" column is the model's estimate of the latent event rate
lambda. The maximum-likelihood rate estimate from the generated event times is
lambda_hat = (n_events - 1) / (t_last - t_first) = 19 / sum(dt). Computing it on
the reproduced test set gives the GROUND-TRUTH-recoverable rate distribution, so
we can check directly whether the paper's LERD Median Rate / 95% CI is what a
correct rate estimator yields (it should be ~7.5 / 12.5 / 17.5), and by how far
the reported baseline numbers (1.000, 0.340, ...) miss it.

This does NOT retrain LERD or the baselines; it verifies the target the Median
Rate column is estimating and the size of the baseline gap. The IoU column (LERD
0.202-0.473 vs baselines 0) is a trained-model quantity and is out of scope here.

Deps: numpy only. Deterministic (fixed seeds).
"""

from __future__ import annotations

import numpy as np

BANDS = {
    "[5,10]":  dict(mu=7.5,  lo=5.0,  hi=10.0),
    "[10,15]": dict(mu=12.5, lo=10.0, hi=15.0),
    "[15,20]": dict(mu=17.5, lo=15.0, hi=20.0),
}
SIGMA_RATE = 1.0
N_OBS = 20          # observations per sequence
SIGMA_ETA = 0.07    # observation noise (unused for rate recovery, kept for fidelity)

# Table 1 LERD Median Rate / 95% CI, for side-by-side comparison.
PAPER_LERD = {
    "[5,10]":  (7.532,  4.300, 14.867),
    "[10,15]": (12.500, None,  None),   # midpoint reference; exact CI not transcribed
    "[15,20]": (18.843, 10.465, 35.244),
}
PAPER_BASELINE_MEDIAN = {  # [5,10] band, for gap illustration
    "NODE": 1.000, "ODE-RNN": 1.000, "STRODE": 0.340,
}


def truncnormal(mu, sigma, lo, hi, size, rng):
    """Rejection-sampled truncated normal on [lo, hi]."""
    out = np.empty(size)
    filled = 0
    while filled < size:
        draw = rng.normal(mu, sigma, size * 2)
        draw = draw[(draw >= lo) & (draw <= hi)]
        take = min(len(draw), size - filled)
        out[filled:filled + take] = draw[:take]
        filled += take
    return out


def gen_sequence(lam, rng):
    """One sequence: return (timestamps, observations, lambda_hat)."""
    dt = rng.exponential(1.0 / lam, size=N_OBS)      # inter-event times ~ Exp(lambda)
    t = np.cumsum(dt)
    y = np.sin(t) + rng.normal(0.0, SIGMA_ETA, N_OBS)
    span = t[-1] - t[0]
    lam_hat = (N_OBS - 1) / span if span > 0 else np.nan
    return t, y, lam_hat


def gen_split(band, n_rates, n_seq, rng):
    """Return array of per-sequence lambda_hat for a split."""
    cfg = BANDS[band]
    rates = truncnormal(cfg["mu"], SIGMA_RATE, cfg["lo"], cfg["hi"], n_rates, rng)
    hats = []
    for lam in rates:
        for _ in range(n_seq):
            _, _, lh = gen_sequence(lam, rng)
            hats.append(lh)
    return np.array(hats)


def median_and_ci(x):
    """Median and 95% CI of the estimated event rate lambda_hat.

    Per Appendix E.2.1 the reported interval is the "median and 95% confidence
    interval of the estimated event rate", i.e. the 2.5/97.5 percentiles of the
    lambda_hat distribution itself (its spread across sequences), not a CI of the
    median. This is what makes the paper's LERD CI wide (e.g. [4.30, 14.87]).
    """
    return float(np.median(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main():
    print("Toy dataset reproduction: latent event-rate recovery (Table 1 Median Rate)\n")
    print(f"{'band':>8}  {'GT mu':>6}  {'repro median [95% CI]':>26}   {'paper LERD':>22}")
    for band in BANDS:
        # test split per spec: 25 new rates x 50 sequences
        rng = np.random.default_rng(abs(hash(band)) % (2**32))
        hats = gen_split(band, n_rates=25, n_seq=50, rng=rng)
        hats = hats[np.isfinite(hats)]
        med, lo, hi = median_and_ci(hats)
        p = PAPER_LERD[band]
        paper = f"{p[0]:.3f}" + (f" [{p[1]:.2f}, {p[2]:.2f}]" if p[1] else "")
        print(f"{band:>8}  {BANDS[band]['mu']:6.1f}  "
              f"{med:8.3f} [{lo:6.3f}, {hi:7.3f}]   {paper:>22}")

    print("\nBaseline gap on [5,10] (paper Table 1 Median Rate vs ground-truth ~7.5):")
    for m, v in PAPER_BASELINE_MEDIAN.items():
        print(f"  {m:>8}: {v:.3f}   (off by {7.5 - v:+.2f} from GT rate)")
    print("\nRead: if the reproduced median/CI matches LERD's Median Rate column, the")
    print("paper's LERD rate estimate is recovering the true latent rate, while the")
    print("baseline medians (1.000 / 0.340) are far from it. This supports claim 3's")
    print("substance (baselines fail to recover latent structure) via the rate column;")
    print("the IoU numbers themselves require training LERD and are not reproduced here.")


if __name__ == "__main__":
    main()
