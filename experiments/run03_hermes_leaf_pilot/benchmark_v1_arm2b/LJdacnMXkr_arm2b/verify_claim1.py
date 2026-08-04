"""Claim 1 (Theorem 4.1): a symmetric, entrywise-positive smoothing operator can be
rescaled by a DIAGONAL matrix (symmetric Sinkhorn) into a diffusion operator that is
self-adjoint w.r.t. the mass-weighted inner product <x,y>_m = x^T diag(m) y.

Test: random point clouds / kernels / non-uniform mass vectors. Check
  (a) existence of d>0 with diag(d)W diag(d)1 = m,
  (b) T = diag(m)^{-1} diag(d) W diag(d) satisfies <Tx,y>_m = <x,Ty>_m,
  (c) T differs from W only by diagonal rescalings (T = diag(a) W diag(b)).
Mutation: (M1) make W non-symmetric -> self-adjointness must break;
          (M2) replace Sinkhorn d by plain row normalization -> m-self-adjointness breaks
               for non-constant m/degree.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sinkhorn_lib import gaussian_kernel, exponential_kernel, sinkhorn_operator, row_normalized

SEED = 20260803
rng = np.random.default_rng(SEED)
out = {"claim": 1, "seed": SEED, "source": "Theorem 4.1", "cases": []}

for name, kern, dim, n in [("gaussian_2d", "g", 2, 120), ("exponential_3d", "e", 3, 100),
                           ("gaussian_5d", "g", 5, 80)]:
    X = rng.normal(size=(n, dim))
    eps = 0.8
    W = gaussian_kernel(X, eps) if kern == "g" else exponential_kernel(X, eps)
    m = rng.uniform(0.3, 3.0, size=n)          # non-uniform mass
    T, S, d, iters = sinkhorn_operator(W, m)

    x = rng.normal(size=n); y = rng.normal(size=n)
    lhs = (T @ x) @ (m * y)
    rhs = x @ (m * (T @ y))
    adj_err = abs(lhs - rhs) / max(abs(lhs), 1e-30)

    rowsum_err = float(np.max(np.abs(S.sum(1) - m) / m))
    mass_err = float(np.max(np.abs(T @ np.ones(n) - 1.0)))
    # T = diag(a) W diag(b): recover a,b and check
    a = d / m; b = d
    diag_resc_err = float(np.max(np.abs(T - (a[:, None] * W) * b[None, :])) / np.max(np.abs(T)))

    # mutation M1: non-symmetric W
    Wm = W * (1 + 0.5 * rng.random(W.shape))
    dm = np.ones(n)
    for _ in range(500):
        dm = np.sqrt(dm * m / (Wm @ dm))
    Sm = (dm[:, None] * Wm) * dm[None, :]
    Tm = Sm / m[:, None]
    adj_err_M1 = float(abs((Tm @ x) @ (m * y) - x @ (m * (Tm @ y))) / max(abs((Tm @ x) @ (m * y)), 1e-30))

    # mutation M2: row normalization instead of Sinkhorn
    Tr = row_normalized(W)
    adj_err_M2 = float(abs((Tr @ x) @ (m * y) - x @ (m * (Tr @ y))) / max(abs((Tr @ x) @ (m * y)), 1e-30))

    out["cases"].append(dict(name=name, n=n, dim=dim, eps=eps, sinkhorn_iters=int(iters),
                             rowsum_err=rowsum_err, mass_err=mass_err,
                             m_self_adjoint_rel_err=float(adj_err),
                             diagonal_rescaling_rel_err=diag_resc_err,
                             d_min=float(d.min()), d_max=float(d.max()),
                             mutation_nonsym_W_adj_err=adj_err_M1,
                             mutation_rownorm_adj_err=adj_err_M2))

ok = all(c["m_self_adjoint_rel_err"] < 1e-10 and c["rowsum_err"] < 1e-10
         and c["diagonal_rescaling_rel_err"] < 1e-12 for c in out["cases"])
mut = all(c["mutation_nonsym_W_adj_err"] > 1e-3 and c["mutation_rownorm_adj_err"] > 1e-3
          for c in out["cases"])
out["all_cases_self_adjoint"] = bool(ok)
out["mutations_break_property"] = bool(mut)
out["verdict"] = "verified" if (ok and mut) else "inconclusive"

os.makedirs("results", exist_ok=True)
json.dump(out, open("results/claim1.json", "w"), indent=1)
print(json.dumps(out, indent=1))
