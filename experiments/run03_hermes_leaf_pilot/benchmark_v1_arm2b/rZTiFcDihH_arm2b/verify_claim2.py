"""Claim 2 (Proposition 4.1): ALG^theta attains competitive ratio theta_K in [sqrt2, PHI)
for 2-bounded K-OPSD with K packet types -- i.e. it beats the golden-ratio barrier
when the value set is finite.

First-principles check: exhaustive adversary. For each K we search over normalized
value sets V (|V|=K, min value 1) and over the algorithm parameter theta, computing
the exact worst-case OPT/ALG ratio by enumerating ALL 2-bounded arrival sequences of
length T. theta_K := min_theta max_V max_instance ratio.
Seed: deterministic (exhaustive, no randomness).
"""
import itertools, json, time, sys
import numpy as np
from common import PHI, worst_case_ratio

T = 3
THETAS = np.round(np.arange(1.30, 1.66, 0.02), 3)


def value_sets(K, grid):
    if K == 1:
        return [(1.0,)]
    return [(1.0,) + c for c in itertools.combinations(grid, K - 1)]


def theta_K(K, grid):
    best_theta, best_val, best_V = None, float("inf"), None
    for th in THETAS:
        mx, argV = 0.0, None
        for V in value_sets(K, grid):
            r, _ = worst_case_ratio(list(V), float(th), T)
            if r > mx:
                mx, argV = r, V
        if mx < best_val - 1e-9:
            best_val, best_theta, best_V = mx, float(th), argV
    return best_theta, best_val, best_V


if __name__ == "__main__":
    t0 = time.time()
    out = {"claim": 2, "source": "Proposition 4.1", "T_enum": T, "phi": PHI,
           "sqrt2": 2 ** 0.5, "per_K": {}}
    grids = {1: [], 2: [1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5],
             3: [1.3, 1.6, 2.0, 2.6], 4: [1.3, 1.7, 2.2]}
    for K in [1, 2, 3, 4]:
        th, val, V = theta_K(K, grids[K])
        out["per_K"][str(K)] = {"theta_star": th, "worst_case_ratio": val,
                                "worst_value_set": list(V) if V else [1.0]}
        print(K, th, round(val, 5), V, f"{time.time()-t0:.1f}s", flush=True)
    ratios = [out["per_K"][str(K)]["worst_case_ratio"] for K in [2, 3, 4]]
    out["monotone_in_K"] = all(ratios[i] <= ratios[i + 1] + 1e-9 for i in range(len(ratios) - 1))
    out["all_below_phi"] = all(r < PHI - 1e-9 for r in ratios)
    out["all_at_least_sqrt2_minus_tol"] = all(r >= 2 ** 0.5 - 1e-9 for r in ratios)
    # MUTATION: force theta = PHI (classical algorithm) and also break the finiteness
    # of the value set by adding many types -> ratio must approach PHI.
    big = [1.0, 1.2, 1.45, 1.6, 1.618, 2.0, 2.4, 2.6]
    mut, _ = worst_case_ratio(big, PHI, T)
    out["mutation_many_types_theta_phi"] = mut
    out["mutation_breaks_gap"] = mut > max(ratios) - 1e-9
    out["runtime_s"] = time.time() - t0
    json.dump(out, open("results/claim2.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
