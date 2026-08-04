"""Claim 3 (Section 4/5): the symmetric Sinkhorn algorithm empirically requires only
5-10 iterations to reduce the normalization error below 0.1%.

Metric: relative normalization error  err_k = max_i | (diag(d)W diag(d) 1)_i - m_i | / m_i
after k symmetric Sinkhorn iterations, starting from d = 1. We report the smallest k
with err_k < 1e-3 (0.1%) across many configurations: point clouds in 2D/3D, Gaussian and
exponential kernels, uniform and non-uniform mass, a wide bandwidth sweep, and
clustered (ill-conditioned) geometries.

Mutation: (M1) strongly clustered data with very small bandwidth (near-block-diagonal
kernel, poor conditioning) -> iteration count must grow well beyond 10;
(M2) target mass with extreme dynamic range (1e-3 .. 1e3) -> count must grow.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sinkhorn_lib import gaussian_kernel, exponential_kernel

SEED = 20260803


def iters_to_tol(W, m, tol=1e-3, max_iter=2000):
    d = np.ones(len(m))
    for k in range(max_iter + 1):
        err = float(np.max(np.abs(d * (W @ d) - m) / m))
        if err < tol:
            return k, err
        d = np.sqrt(d * m / (W @ d))
    return max_iter, err


rng = np.random.default_rng(SEED)
out = {"claim": 3, "seed": SEED, "source": "Section 4/5 (empirical convergence)",
       "tolerance": 1e-3, "configs": []}

for dim in (2, 3):
    for kname in ("gaussian", "exponential"):
        for eps in (0.3, 0.5, 0.8, 1.2, 2.0):
            for mass in ("uniform", "nonuniform"):
                n = 200
                X = rng.normal(size=(n, dim))
                W = gaussian_kernel(X, eps) if kname == "gaussian" else exponential_kernel(X, eps)
                m = np.ones(n) if mass == "uniform" else rng.uniform(0.5, 2.0, n)
                k, err = iters_to_tol(W, m)
                out["configs"].append(dict(dim=dim, kernel=kname, eps=eps, mass=mass,
                                           n=n, iters=k, final_err=err))

ks = [c["iters"] for c in out["configs"]]
out["n_configs"] = len(ks)
out["iters_min"] = int(min(ks)); out["iters_max"] = int(max(ks))
out["iters_median"] = float(np.median(ks)); out["iters_mean"] = float(np.mean(ks))
out["frac_within_5_to_10"] = float(np.mean([5 <= k <= 10 for k in ks]))
out["frac_at_most_10"] = float(np.mean([k <= 10 for k in ks]))

# ---- Mutation M1: clustered geometry, tiny bandwidth
Xc = np.vstack([rng.normal(loc=c, scale=0.02, size=(60, 2))
                for c in ([0, 0], [8, 0], [0, 8], [8, 8])])
Wc = gaussian_kernel(Xc, 0.05)
Wc = np.maximum(Wc, 1e-300)
k1, e1 = iters_to_tol(Wc, np.ones(len(Xc)))
out["mutation_clustered_tiny_eps"] = dict(iters=k1, final_err=e1)

# ---- Mutation M2: extreme target-mass dynamic range
X2 = rng.normal(size=(200, 2))
W2 = gaussian_kernel(X2, 0.8)
m2 = 10.0 ** rng.uniform(-3, 3, 200)
k2, e2 = iters_to_tol(W2, m2)
out["mutation_extreme_mass_range"] = dict(iters=k2, final_err=e2)

out["mutations_increase_iterations"] = bool(k1 > 10 or k2 > 10)
out["verdict"] = ("verified" if out["frac_within_5_to_10"] >= 0.8 else
                  ("falsified" if out["frac_at_most_10"] < 0.5 else "inconclusive"))
out["note"] = ("Paper's '5-10 iterations' is an empirical claim without released code or "
               "the paper's exact datasets; reproduced on synthetic point clouds spanning "
               "kernel type, dimension, bandwidth and mass profile.")

os.makedirs("results", exist_ok=True)
json.dump(out, open("results/claim3.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "configs"}, indent=1))
