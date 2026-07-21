"""A3 (Theorem 3.3 asymmetric propagation) and A4 (local accuracy < 0.015)
using the paper's OWN ten simulated scenarios (arXiv 2602.16505, Fig. 2 setup).

Paper spec transcribed verbatim:
  x1,x2,x3 ~ N(0,1);  lambda = 0.03 (constant baseline hazard h0);
  beta1=0.4, beta2=-0.8, beta3=-0.6, beta12=-0.5, beta13=0.2;
  max follow-up t = 70;  h(t|x) = h0 * exp(G(t|x));  S(t|x) = exp(-int_0^t h).
  Risk scores G(t|x), scenarios 1..10:
    1  b1 x1 + b2 x2 + b3 x3
    2  b1 x1 log(t+1) + b2 x2 + b3 x3
    3  b1 x1 + b2 x2 + b3 x3 + b13 x1 x3
    4  b1 x1 log(t+1) + b2 x2 + b3 x3 + b13 x1 x3
    5  b1 x1 + b2 x2 + b3 x3 + b13 x1 x3 log(t+1)
    6  b1 x1^2 + b2 (2/pi) arctan(0.7 x2) + b3 x3
    7  b1 x1^2 log(t+1) + b2 (2/pi) arctan(0.7 x2) + b3 x3
    8  b1 x1^2 + b2 (2/pi) arctan(0.7 x2) + b3 x3 + b12 x1 x2 + b13 x1 x3^2
    9  b1 x1^2 log(t+1) + b2 (2/pi) arctan(0.7 x2) + b3 x3 + b12 x1 x2 + b13 x1 x3^2
   10  b1 x1^2 + b2 (2/pi) arctan(0.7 x2) + b3 x3 + b12 x1 x2 + b13 x1 x3^2 log(t+1)

Theorem 3.3 (verbatim): with G = g_Z(t|x) + sum_{M in Iid} g_M(x) and Id={Z},
2<=|Z|<p the only time-dependent set, features independent, for SurvFD of log h:
  1. for any L SUBSET of Z it may occur that L in Id* while L in Iid (downward prop.)
  2. for any L SUPERSET of Z with L in Iid, it holds L in Iid* (NO upward propagation)
Scenarios 5 and 10 are exactly of this form with Z={1,3} (1-indexed), p=3.

Local accuracy (Sec. B, Eq. B35): the decomposition error, i.e. the difference between
an individual's survival prediction and the dataset average, versus the sum of all
SurvSHAP-IQ attributions. residual(t|x) = | sum_K phi_K(t|x) - (S(t|x) - E_X[S(t|X)]) |.
"""
import numpy as np
from itertools import combinations
from math import comb

B1, B2, B3, B12, B13 = 0.4, -0.8, -0.6, -0.5, 0.2
LAM = 0.03
P = 3
# time grid inside the paper's follow-up window [0, 70]
TIMES = np.array([1.0, 5.0, 20.0, 50.0, 70.0])


def scenarios():
    L = np.log
    return {
        1: lambda t, x: B1 * x[0] + B2 * x[1] + B3 * x[2],
        2: lambda t, x: B1 * x[0] * L(t + 1) + B2 * x[1] + B3 * x[2],
        3: lambda t, x: B1 * x[0] + B2 * x[1] + B3 * x[2] + B13 * x[0] * x[2],
        4: lambda t, x: B1 * x[0] * L(t + 1) + B2 * x[1] + B3 * x[2] + B13 * x[0] * x[2],
        5: lambda t, x: B1 * x[0] + B2 * x[1] + B3 * x[2] + B13 * x[0] * x[2] * L(t + 1),
        6: lambda t, x: B1 * x[0] ** 2 + B2 * (2 / np.pi) * np.arctan(0.7 * x[1]) + B3 * x[2],
        7: lambda t, x: B1 * x[0] ** 2 * L(t + 1) + B2 * (2 / np.pi) * np.arctan(0.7 * x[1]) + B3 * x[2],
        8: lambda t, x: B1 * x[0] ** 2 + B2 * (2 / np.pi) * np.arctan(0.7 * x[1]) + B3 * x[2]
                        + B12 * x[0] * x[1] + B13 * x[0] * x[2] ** 2,
        9: lambda t, x: B1 * x[0] ** 2 * L(t + 1) + B2 * (2 / np.pi) * np.arctan(0.7 * x[1]) + B3 * x[2]
                        + B12 * x[0] * x[1] + B13 * x[0] * x[2] ** 2,
        10: lambda t, x: B1 * x[0] ** 2 + B2 * (2 / np.pi) * np.arctan(0.7 * x[1]) + B3 * x[2]
                         + B12 * x[0] * x[1] + B13 * x[0] * x[2] ** 2 * L(t + 1),
    }


def logh(G, t, x):
    return np.log(LAM) + G(t, x)


NGRID = 129


def surv(G, t, x):
    # G may be time-varying; integrate cumulative hazard numerically on a fine grid.
    # Vectorised over the grid: broadcast t as a column against the sample axis.
    ts = np.linspace(0.0, t, NGRID)
    xb = [np.asarray(c)[None, :] if np.ndim(c) else np.asarray(c) for c in x]
    hs = LAM * np.exp(G(ts[:, None], xb))
    hs = np.broadcast_to(hs, (NGRID,) + np.shape(hs)[1:])
    H = np.trapezoid(hs, ts, axis=0)
    return np.exp(-H)


def _cols(X, M, x_ref):
    return [x_ref[i] if i in M else X[:, i] for i in range(P)]


def pure_effects(F, t, x_ref, X, subsets):
    f = {(): float(np.mean(F(t, [X[:, i] for i in range(P)])))}
    for M in sorted(subsets, key=len):
        Ex = float(np.mean(F(t, _cols(X, M, x_ref))))
        f[M] = Ex - sum(f[Z] for Z in f if set(Z) < set(M))
    return f


ALL_SUBSETS = [S for r in range(1, P + 1) for S in combinations(range(P), r)]


def value_fn(F_S, t, x_ref, M, X, base, cache=None):
    key = tuple(sorted(M))
    if cache is not None and key in cache:
        return cache[key]
    v = float(np.mean(F_S(t, _cols(X, M, x_ref)))) - base
    if cache is not None:
        cache[key] = v
    return v


def n_shapley(F_S, t, x_ref, K, X, base, order, cache=None):
    """n-Shapley interaction index of order `order` for subset K (Eq. 10)."""
    k = len(K)
    rest = [i for i in range(P) if i not in K]
    total = 0.0
    for r in range(len(rest) + 1):
        for M in combinations(rest, r):
            delta = 0.0
            for Lr in range(k + 1):
                for Lsub in combinations(K, Lr):
                    delta += (-1) ** (k - Lr) * value_fn(F_S, t, x_ref, set(M) | set(Lsub),
                                                        X, base, cache)
            total += delta / ((P - k + 1) * comb(P - k, len(M)))
    return total


def run_a3(seed, n_ref, k_tol=6.0):
    """Theorem 3.3 on scenarios 5 and 10 (Z = {x1,x3} = indices (0,2))."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_ref, P))
    x_ref = np.array([0.7, -0.5, 1.1])
    sc = scenarios()
    out = {}
    for s in (5, 10):
        G = sc[s]
        F = lambda t, xs, G=G: logh(G, t, xs)
        vals = {t: pure_effects(F, t, x_ref, X, ALL_SUBSETS) for t in TIMES}
        se = float(np.std(F(TIMES[0], [X[:, i] for i in range(P)])) / np.sqrt(n_ref))
        tol = max(k_tol * se, 1e-4)
        spread = {M: float(max(vals[t][M] for t in TIMES) - min(vals[t][M] for t in TIMES))
                  for M in ALL_SUBSETS}
        td = {M: spread[M] > tol for M in ALL_SUBSETS}
        Z = (0, 2)
        downward = [M for M in ALL_SUBSETS if set(M) < set(Z)]      # {x1},{x3}
        upward = [M for M in ALL_SUBSETS if set(M) > set(Z)]        # {x1,x2,x3}
        out[s] = {
            "tol": tol, "spread": spread, "td": td,
            "Z_is_TD": td[Z],
            "downward_any_TD": any(td[M] for M in downward),
            "downward_detail": {M: td[M] for M in downward},
            "upward_all_TI": all(not td[M] for M in upward),
            "upward_detail": {M: td[M] for M in upward},
        }
    return out


def run_a4(seed, n_ref, order=P):
    """Local accuracy residual on all ten scenarios."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_ref, P))
    x_ref = np.array([0.7, -0.5, 1.1])
    res = {}
    for s, G in scenarios().items():
        F_S = lambda t, xs, G=G: surv(G, t, xs)
        worst = 0.0
        for t in TIMES:
            base = float(np.mean(F_S(t, [X[:, i] for i in range(P)])))
            target = float(np.asarray(F_S(t, [np.array([v]) for v in x_ref])).ravel()[0]) - base
            ks = [K for K in ALL_SUBSETS if len(K) <= order]
            cache = {}
            tot = sum(n_shapley(F_S, t, x_ref, K, X, base, order, cache) for K in ks)
            worst = max(worst, abs(tot - target))
        res[s] = worst
    return res


def main():
    seeds = list(range(10))
    N = 20000
    print("=== A3: Theorem 3.3 asymmetric propagation (scenarios 5 & 10, Z={x1,x3}) ===")
    print(f"{'seed':>4} | {'sc5 Z=TD':>8} {'down TD':>8} {'up all TI':>9} |"
          f" {'sc10 Z=TD':>9} {'down TD':>8} {'up all TI':>9}")
    a3ok = []
    for sd in seeds:
        r = run_a3(sd, N)
        ok = all(r[s]["Z_is_TD"] and r[s]["downward_any_TD"] and r[s]["upward_all_TI"]
                 for s in (5, 10))
        a3ok.append(ok)
        print(f"{sd:>4} | {str(r[5]['Z_is_TD']):>8} {str(r[5]['downward_any_TD']):>8} "
              f"{str(r[5]['upward_all_TI']):>9} | {str(r[10]['Z_is_TD']):>9} "
              f"{str(r[10]['downward_any_TD']):>8} {str(r[10]['upward_all_TI']):>9}")
    print(f"A3 holds on {sum(a3ok)}/{len(a3ok)} seeds")
    r0 = run_a3(0, N)
    for s in (5, 10):
        print(f"  scenario {s} tol={r0[s]['tol']:.2e} spreads: "
              + " ".join(f"{M}={r0[s]['spread'][M]:.2e}" for M in ALL_SUBSETS))

    print("\n=== A4: local accuracy residual (max over t), full-order SurvSHAP-IQ ===")
    print("scenario:  " + "  ".join(f"{s:>7}" for s in range(1, 11)))
    allres = []
    for sd in seeds[:5]:
        r = run_a4(sd, N)
        allres.append(r)
        print(f"seed {sd:>2}:   " + "  ".join(f"{r[s]:7.5f}" for s in range(1, 11)))
    mx = max(max(r.values()) for r in allres)
    print(f"worst residual over all scenarios/seeds/timepoints: {mx:.6f}  "
          f"(<0.015 -> {mx < 0.015})")

    print("\n  order-2 truncated (paper's practical estimator up to pairwise):")
    for sd in seeds[:3]:
        r = run_a4(sd, N, order=2)
        print(f"  seed {sd:>2}:   " + "  ".join(f"{r[s]:7.5f}" for s in range(1, 11)))


if __name__ == "__main__":
    main()
