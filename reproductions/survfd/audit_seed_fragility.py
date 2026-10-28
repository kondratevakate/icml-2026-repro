"""Seed-fragility + MC-convergence audit for verify_survfd.py.

Re-implements the same estimators but parameterised by (seed, N_REF) so that every
source of randomness can be swept. Logic is copied verbatim from verify_survfd.py;
only the global X_MC is replaced by an explicit argument.
"""
import numpy as np
from itertools import combinations
from math import comb

TIMES = np.array([0.5, 2.0, 5.0])
H0 = 0.1


def G_scenarios():
    return {
        "A": (lambda t, x1, x2: 1.2 * x1 + 0.9 * x2, {"Iid": [(0,), (1,)], "Id": []}),
        "B": (lambda t, x1, x2: 1.2 * x1 + 0.9 * x2 * np.log1p(t), {"Iid": [(0,)], "Id": [(1,)]}),
        "C": (lambda t, x1, x2: 1.2 * x1 + 0.9 * x2 + 0.8 * x1 * x2,
              {"Iid": [(0,), (1,), (0, 1)], "Id": []}),
        "D": (lambda t, x1, x2: 1.2 * x1 + 0.9 * np.arctan(2 * x2) + 0.8 * x1 * x2,
              {"Iid": None, "Id": None}),
    }


def logh(G, t, x1, x2):
    return np.log(H0) + G(t, x1, x2)


def survival(G, t, x1, x2):
    return np.exp(-H0 * t * np.exp(G(t, x1, x2)))


def pure_effects(F, t, x_ref, subsets, X):
    f = {(): float(np.mean(F(t, X[:, 0], X[:, 1])))}
    for M in sorted(subsets, key=len):
        if M == ():
            continue
        x1v = x_ref[0] if 0 in M else X[:, 0]
        x2v = x_ref[1] if 1 in M else X[:, 1]
        Ex = float(np.mean(F(t, x1v, x2v)))
        f[M] = Ex - sum(f[Z] for Z in f if set(Z) < set(M))
    return f


def classify(F, x_ref, subsets, X, k_tol=6.0):
    vals = {t: pure_effects(F, t, x_ref, subsets, X) for t in TIMES}
    se = float(np.std(F(TIMES[0], X[:, 0], X[:, 1])) / np.sqrt(X.shape[0]))
    tol = max(k_tol * se, 1e-4)
    spreads = {}
    verdict = {}
    for M in subsets:
        s = np.array([vals[t][M] for t in TIMES])
        spreads[M] = float(s.max() - s.min())
        verdict[M] = "TD" if spreads[M] > tol else "TI"
    return verdict, tol, spreads


def value_fn(F_S, t, x_ref, M, X):
    x1v = x_ref[0] if 0 in M else X[:, 0]
    x2v = x_ref[1] if 1 in M else X[:, 1]
    return float(np.mean(F_S(t, x1v, x2v))) - float(np.mean(F_S(t, X[:, 0], X[:, 1])))


def n_shapley_interaction(F_S, t, x_ref, K, X, p=2):
    rest = [i for i in range(p) if i not in K]
    total = 0.0
    for r in range(len(rest) + 1):
        for M in combinations(rest, r):
            Mset = set(M)
            delta = 0.0
            for Lr in range(len(K) + 1):
                for L in combinations(K, Lr):
                    delta += (-1) ** (len(K) - len(L)) * value_fn(F_S, t, x_ref, Mset | set(L), X)
            total += delta / ((p - len(K) + 1) * comb(p - len(K), len(M)))
    return total


def one_run(seed, n_ref, k_tol=6.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_ref, 2))
    x_ref = np.array([0.4, -0.6])
    subsets = [(0,), (1,), (0, 1)]
    out = {"claim1": {}, "mags": {}, "tol": {}, "spreads": {}}
    for name, (G, gt) in G_scenarios().items():
        F = lambda t, a, b, G=G: logh(G, t, a, b)
        v, tol, sp = classify(F, x_ref, subsets, X, k_tol)
        out["tol"][name] = tol
        out["spreads"][name] = sp
        if gt["Iid"] is not None:
            exp = {M: ("TD" if M in gt["Id"] else "TI") for M in subsets}
            out["claim1"][name] = all(v[M] == exp[M] for M in subsets)
        F_S = lambda t, a, b, G=G: survival(G, t, a, b)
        out["mags"][name] = float(np.mean([abs(n_shapley_interaction(F_S, t, x_ref, (0, 1), X))
                                           for t in TIMES]))
    num = np.mean([out["mags"]["C"], out["mags"]["D"]])
    den = np.mean([out["mags"]["A"], out["mags"]["B"]])
    out["ratio"] = num / max(den, 1e-9)
    out["num"], out["den"] = num, den
    return out


def main():
    seeds = list(range(12))
    print("=== SEED SWEEP (N_REF=60000, tol=6*SE, as shipped) ===")
    print(f"{'seed':>4} {'A':>6} {'B':>6} {'C':>6} {'3/3':>5} {'|noInt|':>9} {'|Int|':>9} {'ratio':>7}")
    ratios = []
    for s in seeds:
        r = one_run(s, 60000)
        c = r["claim1"]
        ratios.append(r["ratio"])
        print(f"{s:>4} {str(c['A']):>6} {str(c['B']):>6} {str(c['C']):>6} "
              f"{sum(c.values()):>4}/3 {r['den']:>9.5f} {r['num']:>9.5f} {r['ratio']:>7.3f}")
    print(f"\nratio across seeds: mean={np.mean(ratios):.4f} sd={np.std(ratios):.4f} "
          f"min={min(ratios):.4f} max={max(ratios):.4f}")

    print("\n=== MC CONVERGENCE of the ratio (seed 0..4 averaged) ===")
    for n in [2000, 10000, 60000, 300000, 2000000]:
        rs = [one_run(s, n)["ratio"] for s in range(5)]
        ds = [one_run(s, n)["den"] for s in range(5)]
        print(f"N_REF={n:>8}: ratio mean={np.mean(rs):.4f} sd={np.std(rs):.5f}  "
              f"no-interaction magnitude mean={np.mean(ds):.6f}")

    print("\n=== TOLERANCE SENSITIVITY (seed 0, N=60000) ===")
    r = one_run(0, 60000)
    print("observed max-min spread of pure effects across t:")
    for name in "ABCD":
        sp = r["spreads"][name]
        print(f"  {name}: x1={sp[(0,)]:.3e} x2={sp[(1,)]:.3e} x1x2={sp[(0,1)]:.3e}"
              f"   tol={r['tol'][name]:.3e}")
    print("\nclaim1 3/3 as function of k_tol multiplier (seeds 0-11):")
    for k in [0.001, 0.01, 0.1, 1.0, 3.0, 6.0, 20.0, 100.0, 1000.0]:
        oks = [sum(one_run(s, 60000, k)["claim1"].values()) for s in range(12)]
        print(f"  k={k:>8}: per-seed {oks}")


if __name__ == "__main__":
    main()
