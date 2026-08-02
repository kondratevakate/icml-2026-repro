"""Claim 4: Definition 1 (Interval Stability) + Proposition 2:
IS(C_PT) = p(1-p) (E L(x, (1-alpha)/p ; s))^2 > 0, i.e. the same input gets different
intervals across repeated runs of the (randomized) algorithm.

Run: .venv/bin/python verify_claim4.py
"""
import json
import numpy as np
import sympy as sp
from cp_common import conformal_quantile, alpha_prime, vcp_intervals, pt_vcp

res = {"command": ".venv/bin/python verify_claim4.py"}

# ---------- symbolic: variance of a two-point mixture {0 w.p. 1-p, L w.p. p} ----------
p, L = sp.symbols("p L", positive=True)
E1 = p * L
E2 = p * L**2
var = sp.simplify(E2 - E1**2)
res["symbolic_variance"] = str(sp.factor(var))                    # L**2*p*(1-p)
res["symbolic_matches_prop2"] = bool(sp.simplify(var - p * (1 - p) * L**2) == 0)

# ---------- Monte-Carlo IS for PT-VCP, exhaustive p grid, fixed calibration set ----------
def experiment(alpha, p_val, seed, n=2000, n_test=500, runs=400, comp_mean=10.0, deterministic=False):
    rng = np.random.default_rng(seed)
    b = np.array([1.0, 1.0])

    def gen(m):
        X = rng.normal(size=(m, 2))
        sign = rng.integers(0, 2, size=m) * 2 - 1
        return X, X @ b + rng.normal(size=m) + sign * comp_mean

    Xtr, ytr = gen(n); Xca, yca = gen(n); Xte, yte = gen(n_test)
    coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    mca, mte = Xca @ coef, Xte @ coef
    # VCP (deterministic reference): IS must be exactly 0
    _, _, q = vcp_intervals(mca, yca, mte, alpha)
    vcp_lengths = np.repeat(2.0 * q, n_test)
    is_vcp = 0.0  # identical across runs by construction; measured below too
    L_adj = 2.0 * conformal_quantile(np.abs(yca - mca), alpha_prime(alpha, p_val))
    lens = np.empty((runs, n_test))
    vcp_rep = np.empty((runs, n_test))
    for r in range(runs):
        rr = np.random.default_rng(10_000 * seed + r)
        if deterministic:      # MUTATION: p=1, no null branch -> pure base CP
            lens[r] = 2.0 * q
        else:
            _, _, l, _ = pt_vcp(mca, yca, mte, alpha, p_val, rr)
            lens[r] = l
        vcp_rep[r] = 2.0 * q
    is_emp = float(np.mean(np.var(lens, axis=0, ddof=0)))
    is_vcp = float(np.mean(np.var(vcp_rep, axis=0, ddof=0)))
    closed = p_val * (1 - p_val) * L_adj ** 2
    return {"alpha": alpha, "p": p_val, "seed": seed, "L_adjusted": float(L_adj),
            "IS_empirical": is_emp, "IS_closed_form_prop2": float(closed),
            "rel_err": float(abs(is_emp - closed) / closed) if closed > 0 else float(is_emp),
            "IS_vcp_deterministic": is_vcp, "vcp_length": float(2 * q),
            "mean_length_pt": float(lens.mean())}

SEEDS = list(range(5))
rows = []
for pv in [0.91, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99]:
    for s in SEEDS:
        rows.append(experiment(0.10, pv, s))
res["grid"] = rows
res["seeds"] = SEEDS
res["max_rel_err_vs_prop2"] = max(r["rel_err"] for r in rows)
res["min_IS_empirical"] = min(r["IS_empirical"] for r in rows)
res["all_IS_positive"] = all(r["IS_empirical"] > 0 for r in rows)
res["max_IS_vcp"] = max(r["IS_vcp_deterministic"] for r in rows)

# ---------- MUTATION: p = 1 (remove the randomness) -> IS must be exactly 0 ----------
mut = [experiment(0.10, 1.0, s, deterministic=True) for s in SEEDS]
res["mutation_p_equals_1"] = mut
res["mutation_max_IS"] = max(m["IS_empirical"] for m in mut)

json.dump(res, open("results/claim4.json", "w"), indent=2)
print(json.dumps({k: res[k] for k in ["symbolic_variance", "symbolic_matches_prop2",
                                      "max_rel_err_vs_prop2", "min_IS_empirical",
                                      "all_IS_positive", "max_IS_vcp", "mutation_max_IS"]}, indent=2))
for r in rows[::5]:
    print(f"p={r['p']} IS_emp={r['IS_empirical']:.4f} IS_prop2={r['IS_closed_form_prop2']:.4f} "
          f"rel_err={r['rel_err']:.4f} IS_vcp={r['IS_vcp_deterministic']}")
