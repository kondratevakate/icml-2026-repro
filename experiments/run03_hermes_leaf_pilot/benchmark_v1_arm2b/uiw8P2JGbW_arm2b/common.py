"""Shared first-principles model of the paper's framework.

Setting (Sec. 3-5 of "Model Monotonicity in Autobidding Auctions"):
- A measure space of queries. A *model* M induces a partition (clusters) of queries.
- Predictions are *calibrated*: for cluster C, the predicted conversion rate for bidder i
  is  hatp_i(C) = E[ p_i(q) | q in C ]  (mass-weighted average of true rates).
- Model M_A *refines* M_B if every cluster of M_B is a disjoint union of clusters of M_A.
  Refinement therefore induces a mean-preserving spread of the prediction vector.
- tCPA bidder i has target t_i; uniform bidding at multiplier mu gives bid
  b_i(C) = mu * t_i * hatp_i(C).
"""
import numpy as np

RNG_DEFAULT = 20260803


def random_refined_instance(rng, n_bidders=3, n_coarse=4, sub_lo=2, sub_hi=4):
    """Return (t, mass_fine, p_fine, group) for a fine partition and its coarsening.

    mass_fine[k] : mass of fine cluster k
    p_fine[k, i] : true (calibrated) conversion rate of bidder i on fine cluster k
    group[k]     : index of the coarse cluster containing fine cluster k
    """
    t = rng.uniform(0.5, 5.0, size=n_bidders)
    masses, ps, group = [], [], []
    for c in range(n_coarse):
        s = rng.integers(sub_lo, sub_hi + 1)
        m = rng.uniform(0.2, 1.0, size=s)
        masses.append(m)
        ps.append(rng.uniform(0.0, 1.0, size=(s, n_bidders)))
        group.extend([c] * s)
    mass = np.concatenate(masses)
    mass = mass / mass.sum()
    return t, mass, np.vstack(ps), np.array(group)


def coarsen(mass, p_fine, group):
    """Calibrated coarse predictions: mass-weighted averages within each coarse cluster."""
    n_c = group.max() + 1
    mass_c = np.zeros(n_c)
    p_c = np.zeros((n_c, p_fine.shape[1]))
    for c in range(n_c):
        sel = group == c
        mc = mass[sel].sum()
        mass_c[c] = mc
        p_c[c] = (mass[sel, None] * p_fine[sel]).sum(axis=0) / mc
    return mass_c, p_c


def fp_revenue(t, mass, p, mu=None):
    """First-price revenue with uniform bidding: E[ max_i mu_i t_i hatp_i(C) ]."""
    mu = np.ones_like(t) if mu is None else mu
    bids = p * (mu * t)[None, :]
    return float((mass * bids.max(axis=1)).sum())


def fp_alloc(t, mass, p, mu=None):
    mu = np.ones_like(t) if mu is None else mu
    bids = p * (mu * t)[None, :]
    return bids.argmax(axis=1)


def welfare(v, mass, p, winners):
    """Welfare = E[ v_winner * p_winner(true rate on that cluster) ]."""
    idx = np.arange(len(winners))
    return float((mass * v[winners] * p[idx, winners]).sum())


def sp_payments(t, mass, p):
    """Second-price / VCG (single slot): winner pays the second-highest bid."""
    bids = p * t[None, :]
    srt = np.sort(bids, axis=1)
    return bids.argmax(axis=1), srt[:, -2]


def sp_revenue(t, mass, p):
    _, pay = sp_payments(t, mass, p)
    return float((mass * pay).sum())
