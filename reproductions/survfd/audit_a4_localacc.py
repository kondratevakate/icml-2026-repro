"""A4: local accuracy of the exact order-2 SurvSHAP-IQ decomposition.

ALL SPEC BELOW TRANSCRIBED FROM paper.txt (extracted from the real arXiv PDF).

Fig. 2, ten scenarios (p. 6), verbatim:
  (1)  b1 x1                + b2 x2                  + b3 x3
  (2)  b1 x1 log(t+1)       + b2 x2                  + b3 x3
  (3)  b1 x1                + b2 x2                  + b3 x3 + b13 x1 x3
  (4)  b1 x1 log(t+1)       + b2 x2                  + b3 x3 + b13 x1 x3
  (5)  b1 x1                + b2 x2                  + b3 x3 + b13 x1 x3 log(t+1)
  (6)  b1 x1^2              + b2 (2/pi)arctan(0.7x2) + b3 x3
  (7)  b1 x1^2 log(t+1)     + b2 (2/pi)arctan(0.7x2) + b3 x3
  (8)  b1 x1^2              + b2 (2/pi)arctan(0.7x2) + b3 x3 + b12 x1x2 + b13 x1 x2^3
  (9)  b1 x1^2 log(t+1)     + b2 (2/pi)arctan(0.7x2) + b3 x3 + b12 x1x2 + b13 x1 x2^3
  (10) b1 x1^2              + b2 (2/pi)arctan(0.7x2) + b3 x3 + b12 x1x2 + b13 x1 x2^3 log(t+1)
Setup (p. 6): n = 1000 observations; x1,x2,x3 ~ N(0,1); lambda = 0.03,
  b1=0.4, b2=-0.8, b3=-0.6, b12=-0.5, b13=0.2; maximum follow-up t = 70;
  "the exact order-2 SurvSHAP-IQ decomposition is computed (cf. Eq. (10)) using the
   ground-truth log-hazard, hazard, and survival functions on the full dataset."

Local accuracy, Eq. (B35)/(B36) verbatim:
  Phi(t)   = sum_{M subset P, |M| <= k} phi^(k)_M(t|x)
  sigma(t) = sqrt( E[ (F(t|x) - E[F(t|X)] - Phi(t))^2 ] / E[ F(t|X)^2 ] )
  sigma_bar = (1/|T|) sum_{t in T} sigma(t)
NOTE this is a POPULATION-NORMALISED RMSE over the dataset, not a per-individual
absolute residual.

Paper's reported targets (p. 6, Sec. 4.1) -- note these are NOT the 0.015 figure:
  ground-truth decompositions: below 0.001   for SURVIVAL
                               below 0.00001 for HAZARD and LOG-HAZARD
  (the < 0.015 figure applies only to PREDICTED survival from fitted CoxPH/GBSA
   models, which are not reproduced here -- no sklearn/scikit-survival available.)
"""
import numpy as np
from itertools import combinations

B1, B2, B3, B12, B13 = 0.4, -0.8, -0.6, -0.5, 0.2
LAM = 0.03
P = 3
N = 1000
TMAX = 70.0
TIMES = np.linspace(TMAX / 20, TMAX, 20)   # time grid inside the follow-up window
ORDER = 2                                  # paper: "exact order-2 SurvSHAP-IQ"


def scenarios():
    lg = lambda t: np.log(t + 1)
    at = lambda x2: (2 / np.pi) * np.arctan(0.7 * x2)
    return {
        1:  lambda t, x: B1 * x[0] + B2 * x[1] + B3 * x[2],
        2:  lambda t, x: B1 * x[0] * lg(t) + B2 * x[1] + B3 * x[2],
        3:  lambda t, x: B1 * x[0] + B2 * x[1] + B3 * x[2] + B13 * x[0] * x[2],
        4:  lambda t, x: B1 * x[0] * lg(t) + B2 * x[1] + B3 * x[2] + B13 * x[0] * x[2],
        5:  lambda t, x: B1 * x[0] + B2 * x[1] + B3 * x[2] + B13 * x[0] * x[2] * lg(t),
        6:  lambda t, x: B1 * x[0] ** 2 + B2 * at(x[1]) + B3 * x[2],
        7:  lambda t, x: B1 * x[0] ** 2 * lg(t) + B2 * at(x[1]) + B3 * x[2],
        8:  lambda t, x: B1 * x[0] ** 2 + B2 * at(x[1]) + B3 * x[2]
                         + B12 * x[0] * x[1] + B13 * x[0] * x[1] ** 3,
        9:  lambda t, x: B1 * x[0] ** 2 * lg(t) + B2 * at(x[1]) + B3 * x[2]
                         + B12 * x[0] * x[1] + B13 * x[0] * x[1] ** 3,
        10: lambda t, x: B1 * x[0] ** 2 + B2 * at(x[1]) + B3 * x[2]
                         + B12 * x[0] * x[1] + B13 * x[0] * x[1] ** 3 * lg(t),
    }


NGRID = 257


def make_F(G, kind):
    """Ground-truth target function F(t|x); x is a list of 3 broadcastable arrays."""
    if kind == "loghazard":
        return lambda t, x: np.log(LAM) + G(t, x)
    if kind == "hazard":
        return lambda t, x: LAM * np.exp(G(t, x))

    def S(t, x):
        # S(t|x) = exp(-int_0^t lambda exp(G(u|x)) du), trapezoid on a fine grid
        us = np.linspace(0.0, t, NGRID)
        sh = np.broadcast(*[np.asarray(c) for c in x]).shape
        us_b = us.reshape((NGRID,) + (1,) * len(sh))
        h = LAM * np.exp(G(us_b, [np.asarray(c)[None, ...] for c in x]))
        h = np.broadcast_to(h, (NGRID,) + sh)
        return np.exp(-np.trapezoid(h, us, axis=0))
    return S


SUBSETS = [S for r in range(0, P + 1) for S in combinations(range(P), r)]


def value_all(F, t, Xind, Xref):
    """v(t|M) for every subset M, for every individual.

    Returns dict M -> array (n_ind,) of E_{X_Mbar}[F(t|x_M, X_Mbar)] - E_X[F(t|X)].
    Expectation over the reference sample = the dataset itself (marginal projection).
    """
    ni, nr = Xind.shape[0], Xref.shape[0]
    base = float(np.mean(F(t, [Xref[:, i] for i in range(P)])))
    out = {}
    for M in SUBSETS:
        if len(M) == 0:
            out[M] = np.zeros(ni)
            continue
        cols = []
        for i in range(P):
            if i in M:
                cols.append(Xind[:, i][:, None])      # (ni,1) individual value
            else:
                cols.append(Xref[:, i][None, :])      # (1,nr) reference draws
        out[M] = F(t, cols).mean(axis=1) - base
    return out, base


def phi_order_k(v, K, k=ORDER):
    """n-Shapley interaction index (Eq. 10) for subset K, order k, p=3."""
    kk = len(K)
    rest = [i for i in range(P) if i not in K]
    from math import comb
    total = 0.0
    for r in range(len(rest) + 1):
        for M in combinations(rest, r):
            delta = 0.0
            for Lr in range(kk + 1):
                for L in combinations(K, Lr):
                    delta = delta + (-1) ** (kk - Lr) * v[tuple(sorted(set(M) | set(L)))]
            total = total + delta / ((P - kk + 1) * comb(P - kk, len(M)))
    return total


def sigma_curve(G, kind, X):
    F = make_F(G, kind)
    Ks = [K for K in SUBSETS if 1 <= len(K) <= ORDER]
    sig = []
    for t in TIMES:
        v, base = value_all(F, t, X, X)
        Phi = sum(phi_order_k(v, K) for K in Ks)
        Fx = F(t, [X[:, i] for i in range(P)])
        num = np.mean((Fx - base - Phi) ** 2)
        den = np.mean(Fx ** 2)
        sig.append(np.sqrt(num / den))
    return float(np.mean(sig))


def main():
    print("=== A4: average local accuracy sigma_bar (Eq. B35/B36), exact order-2 ===")
    print(f"    n={N}, p=3, |T|={len(TIMES)} timepoints on (0,{TMAX}], ground-truth targets")
    print("    paper targets: log-hazard & hazard < 1e-5, survival < 1e-3\n")
    for kind, target in [("loghazard", 1e-5), ("hazard", 1e-5), ("survival", 1e-3)]:
        print(f"--- {kind} (target < {target:g}) ---")
        print("seed | " + " ".join(f"{s:>9}" for s in range(1, 11)) + " |    max   pass")
        for seed in range(5):
            rng = np.random.default_rng(seed)
            X = rng.normal(0, 1, size=(N, P))
            row = [sigma_curve(G, kind, X) for G in scenarios().values()]
            mx = max(row)
            print(f"{seed:>4} | " + " ".join(f"{r:9.2e}" for r in row)
                  + f" | {mx:8.2e} {str(mx < target):>6}")
        print()


if __name__ == "__main__":
    main()
