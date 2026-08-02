"""Claim 1: Algorithm 1 (Prejudicial Trick) preserves marginal coverage (Theorem 6)
while shrinking average interval length (Lemma 1 / Thm 10 / Cor 3).

Run: .venv/bin/python verify_claim1.py
"""
import json, itertools
import numpy as np
import sympy as sp
from cp_common import (conformal_quantile, alpha_prime, vcp_intervals, pt_vcp, covered)

OUT = "results/claim1.json"
res = {"command": ".venv/bin/python verify_claim1.py"}

# ---------- (a) symbolic identity behind Theorem 6 ----------
a, p = sp.symbols("alpha p", positive=True)
ap = 1 - (1 - a) / p
res["symbolic_p_times_one_minus_alpha_prime"] = str(sp.simplify(p * (1 - ap)))  # must be 1-alpha
res["symbolic_identity_holds"] = bool(sp.simplify(p * (1 - ap) - (1 - a)) == 0)

# ---------- (b) exact coverage of PT given an exactly-(1-a')-covering base ----------
# If base has conditional coverage exactly 1-alpha', PT coverage = p*(1-alpha') = 1-alpha.
grid = []
for A in [0.05, 0.10, 0.15, 0.20, 0.30]:
    for P in [0.90, 0.92, 0.95, 0.96, 0.97, 0.98, 0.99]:
        if not (1 - A < P < 1):
            continue
        apv = alpha_prime(A, P)
        grid.append({"alpha": A, "p": P, "alpha_prime": apv,
                     "pt_coverage_exact": P * (1 - apv),
                     "mutated_pt_coverage": P * (1 - A)})
res["exact_grid"] = grid
res["exact_grid_max_abs_coverage_error"] = max(abs(g["pt_coverage_exact"] - (1 - g["alpha"])) for g in grid)
res["mutated_grid_max_coverage_deficit"] = max((1 - g["alpha"]) - g["mutated_pt_coverage"] for g in grid)

# ---------- (c) finite-sample Monte Carlo, misspecified DGP (App. D.1.1 style) ----------
def run_mc(seeds, alpha, p, mu_gap=10.0, n_tr=2000, n_ca=2000, n_te=4000,
           beta=(1.0, 1.0), adjust=True, gaussian_noise=False):
    cov_v, len_v, cov_p, len_p = [], [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        b = np.array(beta, dtype=float)
        d = b.size

        def gen(n):
            X = rng.normal(size=(n, d))
            if gaussian_noise:
                eps = rng.normal(size=n)
            else:
                sign = rng.integers(0, 2, size=n) * 2 - 1
                eps = rng.normal(size=n) + sign * mu_gap
            return X, X @ b + eps

        Xtr, ytr = gen(n_tr); Xca, yca = gen(n_ca); Xte, yte = gen(n_te)
        # misspecified fit: OLS linear model with Gaussian-noise assumption
        coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        mca, mte = Xca @ coef, Xte @ coef
        lo, hi, q = vcp_intervals(mca, yca, mte, alpha)
        cov_v.append(covered(lo, hi, yte).mean()); len_v.append(2 * q)
        plo, phi, plen, _ = pt_vcp(mca, yca, mte, alpha, p, rng, adjust=adjust)
        cov_p.append(covered(plo, phi, yte).mean()); len_p.append(plen.mean())
    f = lambda v: {"mean": float(np.mean(v)), "sem": float(np.std(v, ddof=1) / np.sqrt(len(v)))}
    return {"vcp_coverage": f(cov_v), "vcp_length": f(len_v),
            "pt_coverage": f(cov_p), "pt_length": f(len_p)}

SEEDS = list(range(20))
res["mc_seeds"] = SEEDS
mc = {}
for A, P in itertools.product([0.10, 0.20], [0.96, 0.98]):
    mc[f"alpha{A}_p{P}"] = run_mc(SEEDS, A, P)
res["mc_correct"] = mc

# ---------- MUTATION 1: remove the alpha' adjustment ----------
res["mc_mutation_no_alpha_adjust"] = {
    f"alpha{A}_p{P}": run_mc(SEEDS, A, P, adjust=False)
    for A, P in itertools.product([0.10, 0.20], [0.96, 0.98])}

# ---------- MUTATION 2: well-specified Gaussian scores (paper Example 3 failure case) ----------
res["mc_mutation_gaussian_wellspecified"] = {
    f"alpha{A}_p{P}": run_mc(SEEDS, A, P, gaussian_noise=True)
    for A, P in itertools.product([0.10, 0.20], [0.96, 0.98])}

# Example 3 analytic check: Phi^-1(1-a/2) < p*Phi^-1((1+ (1-a)/p)/2) for all valid (a,p)
from scipy.stats import norm
ex3 = []
for A in [0.05, 0.1, 0.2, 0.3, 0.5]:
    for P in np.round(np.arange(0.01, 1.0, 0.01), 2):
        if not (1 - A < P < 1):
            continue
        lhs = norm.ppf(1 - A / 2); rhs = P * norm.ppf(0.5 * (1 + (1 - A) / P))
        ex3.append(lhs < rhs)
res["example3_inequality_all_hold"] = bool(np.all(ex3))
res["example3_n_points"] = len(ex3)

# verdict summary numbers
res["summary"] = {
    "all_mc_pt_coverage_ge_nominal_minus_2sem": all(
        v["pt_coverage"]["mean"] >= (1 - float(k.split("_")[0][5:])) - 2 * v["pt_coverage"]["sem"]
        for k, v in mc.items()),
    "all_mc_pt_shorter_than_vcp": all(v["pt_length"]["mean"] < v["vcp_length"]["mean"] for v in mc.values()),
}
json.dump(res, open(OUT, "w"), indent=2)
print(json.dumps({k: res[k] for k in
                  ["symbolic_identity_holds", "exact_grid_max_abs_coverage_error",
                   "mutated_grid_max_coverage_deficit", "example3_inequality_all_hold", "summary"]}, indent=2))
print("mc_correct:", json.dumps(res["mc_correct"], indent=2))
print("mutation1:", json.dumps(res["mc_mutation_no_alpha_adjust"], indent=2))
print("mutation2:", json.dumps(res["mc_mutation_gaussian_wellspecified"], indent=2))
