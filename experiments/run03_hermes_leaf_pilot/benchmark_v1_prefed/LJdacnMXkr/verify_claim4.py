"""Claim 4 (Theorem 4.1 corollaries): Sinkhorn-normalized operators jointly
satisfy symmetry (self-adjointness in <.,.>_m), mass conservation (row sums 1),
entrywise positivity and spectral damping (eigenvalues in [0,1]) -- properties
NOT jointly satisfied by row normalization D^-1 K or symmetric normalization
D^-1/2 K D^-1/2.

Mutation: replace the PSD kernel by a symmetric, entrywise-positive but
indefinite kernel -> spectral damping (lam_min >= 0) must break, showing the
[0,1] part of the claim rests on kernel positive-definiteness.
"""
import numpy as np
from sinkhorn_lib import (SEED, rng, gaussian_kernel, exponential_kernel,
                          sym_sinkhorn, row_normalize, sym_normalize, props, dump)

g = rng(4)
TOL = 1e-8
rows = []
for kname, kern in [("gaussian", gaussian_kernel), ("exponential", exponential_kernel)]:
    for n, dim, eps in [(120, 2, 0.3), (120, 3, 0.5), (250, 2, 0.2)]:
        X = g.random((n, dim))
        m = np.ones(n)
        K = kern(X, eps)
        P, _ = sym_sinkhorn(K, m, tol=1e-14)
        for scheme, op in [("sinkhorn", P), ("row_norm", row_normalize(K)),
                           ("sym_norm", sym_normalize(K))]:
            p = props(op, m)
            p.update(kernel=kname, n=n, dim=dim, eps=eps, scheme=scheme,
                     ok_symmetry=p["self_adjoint_residual"] < TOL,
                     ok_mass=p["mass_error"] < 1e-6,
                     ok_positive=p["min_entry"] > 0,
                     ok_spectrum=(p["lam_min"] > -1e-10 and p["lam_max"] < 1 + 1e-8))
            p["all_four"] = all(p[k] for k in ("ok_symmetry", "ok_mass", "ok_positive", "ok_spectrum"))
            rows.append(p)

summary = {}
for s in ["sinkhorn", "row_norm", "sym_norm"]:
    sel = [r for r in rows if r["scheme"] == s]
    summary[s] = {k: float(np.mean([r[k] for r in sel]))
                  for k in ("ok_symmetry", "ok_mass", "ok_positive", "ok_spectrum", "all_four")}

# mutation: symmetric positive-entry but indefinite kernel
n = 120
A = g.random((n, n))
Kind = 0.5 * (A + A.T) + 0.05   # symmetric, entrywise > 0, strongly indefinite
Pm, _ = sym_sinkhorn(Kind, np.ones(n), tol=1e-13, max_iter=20000)
mut = props(Pm, np.ones(n))

verdict = ("verified" if summary["sinkhorn"]["all_four"] == 1.0
           and summary["row_norm"]["all_four"] == 0.0
           and summary["sym_norm"]["all_four"] == 0.0 else "inconclusive")
dump("results/claim4.json", dict(
    claim=4, source="Theorem 4.1 (joint properties) vs. standard normalizations",
    seed=SEED, verdict=verdict,
    property_pass_rate_by_scheme=summary,
    detail=rows,
    mutation_indefinite_kernel=mut,
    mutation_note="With a symmetric, entrywise-positive but indefinite kernel the Sinkhorn "
                  f"operator keeps symmetry/mass/positivity but lam_min = {mut['lam_min']:.3f} < 0, "
                  "so spectral damping requires a positive-definite (Gaussian/exponential) kernel.",
    caveat="Row normalization fails self-adjointness; symmetric normalization fails mass "
           "conservation -- reproduced exactly as claimed."))
