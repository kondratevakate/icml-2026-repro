"""Claim 4 (Theorem 4.1): the Sinkhorn-normalized operator SIMULTANEOUSLY satisfies
  (i) symmetry (self-adjointness w.r.t. the mass-weighted inner product),
  (ii) mass conservation (row sums = 1, i.e. constants preserved),
  (iii) entrywise positivity,
  (iv) spectral damping (eigenvalues in [0,1]),
properties NOT jointly satisfied by row-normalization (D^-1 W) or symmetric
normalization (D^-1/2 W D^-1/2).

For each scheme we check all four properties on the same kernels; the claim requires
Sinkhorn 4/4 and each baseline < 4/4.
Mutation: (M1) indefinite kernel (Gaussian minus a rank-1 negative shift, still
symmetric & positive entrywise but not PSD) -> spectral damping must fail for Sinkhorn
too, showing the check is discriminating rather than vacuous.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sinkhorn_lib import (gaussian_kernel, exponential_kernel, sinkhorn_operator,
                          row_normalized, sym_normalized)

SEED = 20260803
rng = np.random.default_rng(SEED)
TOL = 1e-8


def check(T, m):
    n = T.shape[0]
    M = np.diag(m)
    A = M @ T
    sym = float(np.max(np.abs(A - A.T)) / np.max(np.abs(A)))
    mass = float(np.max(np.abs(T @ np.ones(n) - 1.0)))
    pos = bool(np.all(T > 0))
    ev = np.linalg.eigvals(T)
    ev = np.real_if_close(ev, tol=1e6)
    evr = np.sort(np.real(ev))
    return dict(self_adjoint_m=bool(sym < TOL), sym_rel_err=sym,
                mass_conserving=bool(mass < TOL), mass_err=mass,
                entrywise_positive=pos,
                eig_min=float(evr[0]), eig_max=float(evr[-1]),
                max_abs_imag=float(np.max(np.abs(np.imag(ev)))),
                spectral_damping=bool(evr[0] >= -TOL and evr[-1] <= 1 + TOL),
                n_satisfied=int(sum([sym < TOL, mass < TOL, pos,
                                     evr[0] >= -TOL and evr[-1] <= 1 + TOL])))


out = {"claim": 4, "seed": SEED, "source": "Theorem 4.1", "cases": []}

for name, kern, dim, eps in [("gaussian_2d", "g", 2, 0.8), ("gaussian_3d", "g", 3, 1.0),
                             ("exponential_2d", "e", 2, 0.8)]:
    n = 150
    X = rng.normal(size=(n, dim))
    W = gaussian_kernel(X, eps) if kern == "g" else exponential_kernel(X, eps)
    m = rng.uniform(0.5, 2.0, n)          # non-uniform mass measure

    T_sink, S, d, iters = sinkhorn_operator(W, m)
    T_row = row_normalized(W)
    T_sym = sym_normalized(W)

    case = dict(name=name, n=n, dim=dim, eps=eps, sinkhorn_iters=int(iters),
                sinkhorn=check(T_sink, m),
                row_normalized=check(T_row, m),
                sym_normalized=check(T_sym, m))
    # row normalization IS self-adjoint w.r.t. its own degree measure - check that too,
    # to be fair to the baseline (it still fails mass/positivity/damping jointly?)
    deg = W.sum(1)
    case["row_normalized_wrt_own_degree"] = check(T_row, deg)
    out["cases"].append(case)

sink_ok = all(c["sinkhorn"]["n_satisfied"] == 4 for c in out["cases"])
row_fail = all(c["row_normalized_wrt_own_degree"]["n_satisfied"] < 4 or
               c["row_normalized"]["n_satisfied"] < 4 for c in out["cases"])
sym_fail = all(c["sym_normalized"]["n_satisfied"] < 4 for c in out["cases"])

# ---- Mutation M1: indefinite but entrywise-positive symmetric kernel
n = 120
X = rng.normal(size=(n, 2))
# entrywise-positive, symmetric, but NOT positive semidefinite:
Wind = 1.0 + 0.9 * np.cos(np.pi * ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
Wind = (Wind + Wind.T) / 2
m = np.ones(n)
try:
    T_mut, _, _, _ = sinkhorn_operator(Wind, m)
    mut = check(T_mut, m)
    mut["kernel_min_eig"] = float(np.min(np.linalg.eigvalsh(Wind)))
except Exception as ex:
    mut = {"failed": type(ex).__name__}
out["mutation_indefinite_kernel"] = mut

out["sinkhorn_satisfies_all_four"] = bool(sink_ok)
out["row_normalization_fails"] = bool(row_fail)
out["sym_normalization_fails"] = bool(sym_fail)
out["mutation_breaks_spectral_damping"] = bool(
    isinstance(mut, dict) and mut.get("spectral_damping") is False)
out["row_normalization_satisfies_all_four_wrt_own_degree"] = bool(
    all(c["row_normalized_wrt_own_degree"]["n_satisfied"] == 4 for c in out["cases"]))
out["exclusivity_caveat"] = (
    "Sinkhorn attains all four properties for an ARBITRARY prescribed mass measure m. "
    "Row normalization D^-1 W fails them for a prescribed m, but is itself self-adjoint, "
    "mass-conserving, positive and spectrally damped with respect to its OWN degree "
    "measure - i.e. the paper's exclusivity sub-claim holds only when the mass measure is "
    "specified in advance, which is a genuine counter-example to the claim as literally "
    "stated. Symmetric normalization D^-1/2 W D^-1/2 fails mass conservation outright.")
out["verdict"] = ("inconclusive" if (sink_ok and
                                     out["row_normalization_satisfies_all_four_wrt_own_degree"])
                  else ("verified" if (sink_ok and row_fail and sym_fail) else "inconclusive"))

os.makedirs("results", exist_ok=True)
json.dump(out, open("results/claim4.json", "w"), indent=1)
print(json.dumps(out, indent=1))
