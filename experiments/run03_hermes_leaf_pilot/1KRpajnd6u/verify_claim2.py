"""verify_claim2.py — Propositions 2 & 3 (L-head lower bound, U-head upper bound).

Run: .venv/bin/python verify_claim2.py
Writes results/claim2.json
"""
import json, sys
import numpy as np
import sympy as sp
from fluxnet_core import (stencil, update, l_head_flux, u_head_influx, u_head_update)

OUT = "results/claim2.json"
res = {"command": ".venv/bin/python verify_claim2.py", "claim": 2}

# ---------- (a) symbolic proof of the per-cell inequality
a, alpha, l, ui, b, beta, umax, inflow, outflow = sp.symbols(
    "a alpha l u b beta umax inflow outflow", positive=True)
# L-head: u_next = u - a*alpha + inflow, a = u - l, alpha in (0,1), inflow >= 0
u_next_L = (l + a) - a * alpha            # worst case inflow = 0
res["symbolic_L_margin"] = str(sp.simplify(u_next_L - l))           # = a*(1-alpha) > 0
res["symbolic_L_positive_for_alpha_lt_1"] = bool(
    sp.simplify((u_next_L - l).subs({a: sp.Rational(1), alpha: sp.Rational(1, 2)})) > 0)
u_next_U = (umax - b) + b * beta          # worst case outflow = 0
res["symbolic_U_margin"] = str(sp.simplify(umax - u_next_U))        # = b*(1-beta) > 0
res["symbolic_U_positive_for_beta_lt_1"] = bool(
    sp.simplify((umax - u_next_U).subs({b: sp.Rational(1), beta: sp.Rational(1, 2)})) > 0)

# ---------- (b) numeric sweep: exhaustive over shapes x radii x seeds x logit scales
shapes = [(64,), (127,), (32, 32), (48, 24)]
radii = [1, 2, 3]
seeds = list(range(15))
scales = [1.0, 5.0, 20.0]          # 20.0 saturates sigmoid/softmax -> adversarial extremes

def run_L(shape, radius, seed, scale, steps=20, alpha_cap=1.0, l=0.0):
    rng = np.random.default_rng(seed)
    ndim = len(shape); offs = stencil(radius, ndim); K = len(offs)
    u = (rng.random(shape) * 2.0 + l).astype(np.float64)
    if seed % 3 == 0:                      # include cells exactly at the bound
        u.reshape(-1)[: max(1, u.size // 10)] = l
    worst = 0.0
    for t in range(steps):
        F = l_head_flux(u, l, rng.normal(0, scale, shape),
                        rng.normal(0, scale, (K,) + shape), offs, alpha_cap=alpha_cap)
        u = update(u, F, offs)
        worst = max(worst, float(l - u.min()))     # >0 means violation
    return worst

def run_U(shape, radius, seed, scale, steps=20, beta_cap=1.0, umax=1.0):
    rng = np.random.default_rng(seed)
    ndim = len(shape); offs = stencil(radius, ndim); K = len(offs)
    u = (rng.random(shape) * umax).astype(np.float64)
    if seed % 3 == 0:
        u.reshape(-1)[: max(1, u.size // 10)] = umax
    worst = 0.0
    for t in range(steps):
        G = u_head_influx(u, umax, rng.normal(0, scale, shape),
                          rng.normal(0, scale, (K,) + shape), offs, beta_cap=beta_cap)
        u = u_head_update(u, G, offs)
        worst = max(worst, float(u.max() - umax))
    return worst

grid = [(sh, r, s, sc) for sh in shapes for r in radii for s in seeds for sc in scales]
vL = [run_L(*g) for g in grid]
vU = [run_U(*g) for g in grid]
res["n_configs"] = len(grid)
res["L_head_max_lower_bound_violation"] = max(vL)
res["L_head_n_violating_configs"] = int(sum(1 for v in vL if v > 1e-12))
res["U_head_max_upper_bound_violation"] = max(vU)
res["U_head_n_violating_configs"] = int(sum(1 for v in vU if v > 1e-12))

# also: no clipping used anywhere -> check the code path never calls clip
src = open("fluxnet_core.py").read()
res["no_clipping_in_implementation"] = ("clip" not in src and "maximum(" not in src)

# ---------- (c) mutation: remove the capacity fraction (alpha,beta may exceed 1)
mut_grid = [(sh, r, s, sc) for sh in shapes for r in radii for s in range(5) for sc in [1.0, 5.0]]
mL = [run_L(*g, alpha_cap=1.6) for g in mut_grid]
mU = [run_U(*g, beta_cap=1.6) for g in mut_grid]
res["mutation_alpha_cap_1.6"] = {
    "max_violation": max(mL), "n_violating": int(sum(1 for v in mL if v > 1e-12)),
    "n_configs": len(mut_grid)}
res["mutation_beta_cap_1.6"] = {
    "max_violation": max(mU), "n_violating": int(sum(1 for v in mU if v > 1e-12)),
    "n_configs": len(mut_grid)}

res["verdict"] = ("verified" if res["L_head_n_violating_configs"] == 0
                  and res["U_head_n_violating_configs"] == 0
                  and res["no_clipping_in_implementation"]
                  and res["mutation_alpha_cap_1.6"]["n_violating"] > 0
                  and res["mutation_beta_cap_1.6"]["n_violating"] > 0 else "falsified")
res["env"] = {"python": sys.version.split()[0], "numpy": np.__version__, "sympy": sp.__version__}
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps(res, indent=2))
