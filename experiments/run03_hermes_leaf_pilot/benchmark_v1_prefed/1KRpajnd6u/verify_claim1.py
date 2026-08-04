"""verify_claim1.py — Proposition 1 (discrete conservation) + the ~1e-7..1e-8 magnitude.

Run: .venv/bin/python verify_claim1.py
Writes results/claim1.json
"""
import json, itertools, platform, subprocess, sys
import numpy as np
import sympy as sp
from fluxnet_core import stencil, update, l_head_flux, _softmax, _sigmoid

OUT = "results/claim1.json"
res = {"command": ".venv/bin/python verify_claim1.py", "claim": 1}

# ---------- (a) symbolic exactness on a small periodic ring (exhaustive, exact rationals)
# 1D ring of N cells, stencil radius 1, fully symbolic fluxes: sum must cancel identically.
sym = {}
for N in (3, 4, 5, 6):
    u = sp.symbols(f"u0:{N}")
    Fp = sp.symbols(f"p0:{N}")   # flux i -> i+1
    Fm = sp.symbols(f"m0:{N}")   # flux i -> i-1
    un = [u[i] - Fp[i] - Fm[i] + Fp[(i - 1) % N] + Fm[(i + 1) % N] for i in range(N)]
    diff = sp.simplify(sp.expand(sum(un) - sum(u)))
    sym[f"N={N}"] = str(diff)
res["symbolic_sum_difference"] = sym
res["symbolic_exact"] = all(v == "0" for v in sym.values())

# ---------- (b) float64 / float32 numeric conservation, exhaustive over configs x seeds
def cons_err(dtype, shape, radius, seed, steps=20, mutate=None):
    rng = np.random.default_rng(seed)
    ndim = len(shape)
    offs = stencil(radius, ndim)
    K = len(offs)
    u = rng.random(shape).astype(dtype) + dtype(0.5)
    s0 = np.sum(u.astype(np.float64))
    worst = 0.0
    for t in range(steps):
        a_logit = rng.normal(0, 2, shape).astype(dtype)
        pi_logits = rng.normal(0, 2, (K,) + shape).astype(dtype)
        F = l_head_flux(u, dtype(0.0), a_logit, pi_logits, offs).astype(dtype)
        if mutate == "asymmetric_inflow":
            # break the shared-flux property: receiver gets a *different* amount
            inflow = np.zeros_like(u)
            for k, d in enumerate(offs):
                inflow += np.roll(F[k] * dtype(1.001), shift=d, axis=tuple(range(ndim)))
            u = u - np.sum(F, axis=0) + inflow
        elif mutate == "nonperiodic":
            # zero-padded (non-periodic) shift => mass leaves the domain
            inflow = np.zeros_like(u)
            for k, d in enumerate(offs):
                sh = np.roll(F[k], shift=d, axis=tuple(range(ndim)))
                sl = [slice(None)] * ndim
                for ax, dd in enumerate(d):
                    if dd > 0:
                        sl[ax] = slice(0, dd)
                    elif dd < 0:
                        sl[ax] = slice(dd, None)
                    if dd != 0:
                        tmp = sh.copy(); tmp[tuple(sl)] = 0; sh = tmp
                        sl[ax] = slice(None)
                inflow += sh
            u = u - np.sum(F, axis=0) + inflow
        else:
            u = update(u, F, offs)
        worst = max(worst, abs(np.sum(u.astype(np.float64)) - s0) / abs(s0))
    return worst

shapes = [(64,), (128,), (32, 32), (64, 64)]
radii = [1, 2, 3]
seeds = list(range(20))
for dt_name, dt in (("float32", np.float32), ("float64", np.float64)):
    errs = [cons_err(dt, sh, r, s) for sh in shapes for r in radii for s in seeds]
    res[f"conservation_rel_err_{dt_name}"] = {
        "n_configs": len(errs), "max": max(errs), "median": float(np.median(errs)),
        "min": min(errs),
    }

# ---------- (c) mutation tests
mut = {}
for name in ("asymmetric_inflow", "nonperiodic"):
    errs = [cons_err(np.float32, sh, r, s, mutate=name) for sh in shapes for r in radii for s in range(5)]
    mut[name] = {"max": max(errs), "median": float(np.median(errs)), "min": min(errs)}
res["mutation"] = mut

base_max = res["conservation_rel_err_float32"]["max"]
res["mutation_ratio_asymmetric_over_baseline"] = mut["asymmetric_inflow"]["min"] / base_max
res["mutation_ratio_nonperiodic_over_baseline"] = mut["nonperiodic"]["min"] / base_max

res["paper_reported_range"] = {"table2": [1.19e-7, 2.38e-7], "table3_fluxnet": 3.3e-8,
                               "table4_fluxnet": 7.7e-8}
res["verdict"] = ("verified" if res["symbolic_exact"]
                  and res["conservation_rel_err_float64"]["max"] < 1e-13
                  and 1e-9 < base_max < 1e-5
                  and mut["asymmetric_inflow"]["min"] > 1e3 * base_max else "falsified")
res["env"] = {"python": sys.version.split()[0], "numpy": np.__version__, "sympy": sp.__version__}
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps(res, indent=2))
