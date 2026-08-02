"""Claim 5: Proposition 1 -- PT is a special case of localized (normalized-score) CP when the
local scale estimator sigma_hat outputs extreme values (0+ w.p. 1-p, 1 w.p. p); and the
Interval Stability metric (Definition 1) flags such a method while giving 0 for deterministic CP.

Localized CP: S_norm(x,y) = S(x,y)/sigma_hat(x); quantile Q taken on normalized calibration
scores; interval at x' is mu_hat(x') +- Q * sigma_hat(x').

Run: .venv/bin/python verify_claim5.py
"""
import json
import numpy as np
from cp_common import conformal_quantile, alpha_prime, vcp_intervals, pt_vcp, covered

ALPHA = 0.10
res = {"command": ".venv/bin/python verify_claim5.py", "alpha": ALPHA}


def make_data(seed, n=2000, n_test=2000, comp_mean=10.0):
    rng = np.random.default_rng(seed)
    b = np.array([1.0, 1.0])

    def gen(m):
        X = rng.normal(size=(m, 2))
        s = rng.integers(0, 2, size=m) * 2 - 1
        return X, X @ b + rng.normal(size=m) + s * comp_mean

    Xtr, ytr = gen(n); Xca, yca = gen(n); Xte, yte = gen(n_test)
    coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xca @ coef, yca, Xte @ coef, yte, rng


def localized_cp(mu_ca, y_ca, mu_te, alpha, sig_ca, sig_te):
    s_norm = np.abs(y_ca - mu_ca) / sig_ca
    Q = conformal_quantile(s_norm, alpha)
    half = Q * sig_te
    return mu_te - half, mu_te + half, 2.0 * half, Q


SEEDS = list(range(10))
EPS = [1e-2, 1e-4, 1e-6, 1e-9, 1e-12]
P_GRID = [0.93, 0.95, 0.96, 0.98]

equiv, stability, mutation = [], [], []
for seed in SEEDS:
    mca, yca, mte, yte, rng = make_data(seed)
    base_q_alpha_prime = {pv: conformal_quantile(np.abs(yca - mca), alpha_prime(ALPHA, pv))
                          for pv in P_GRID}
    _, _, q_vcp = vcp_intervals(mca, yca, mte, ALPHA)
    for pv in P_GRID:
        r2 = np.random.default_rng(777 * seed + int(pv * 100))
        pick_ca = r2.uniform(size=mca.size) <= pv     # sigma_hat = 1 w.p. p, else 0+
        pick_te = r2.uniform(size=mte.size) <= pv
        for eps in EPS:
            sig_ca = np.where(pick_ca, 1.0, eps)
            sig_te = np.where(pick_te, 1.0, eps)
            lo, hi, length, Q = localized_cp(mca, yca, mte, ALPHA, sig_ca, sig_te)
            # PT reference: same null mask, half-width = base quantile at alpha'
            qa = base_q_alpha_prime[pv]
            pt_len = np.where(pick_te, 2.0 * qa, 0.0)
            equiv.append({"seed": seed, "p": pv, "eps": eps,
                          "Q_localized": float(Q), "base_q_at_alpha_prime": float(qa),
                          "max_abs_length_diff_vs_PT": float(np.max(np.abs(length - pt_len))),
                          "mean_len_localized": float(length.mean()),
                          "mean_len_PT": float(pt_len.mean()),
                          "coverage_localized": float(covered(lo, hi, yte).mean()),
                          "vcp_length": float(2 * q_vcp)})

    # ---- Interval Stability of randomized localized CP (sigma_hat re-drawn each run) ----
    runs, n_te = 300, mte.size
    for pv in P_GRID:
        lens = np.empty((runs, n_te))
        lens_fixed = np.empty((runs, n_te))
        r3 = np.random.default_rng(99 * seed + int(pv * 100))
        fixed_ca = np.ones(mca.size)          # MUTATION: deterministic sigma_hat == 1
        fixed_te = np.ones(n_te)
        for r in range(runs):
            pc = r3.uniform(size=mca.size) <= pv
            pt_ = r3.uniform(size=n_te) <= pv
            _, _, l, _ = localized_cp(mca, yca, mte, ALPHA,
                                      np.where(pc, 1.0, 1e-12), np.where(pt_, 1.0, 1e-12))
            lens[r] = l
            _, _, lf, _ = localized_cp(mca, yca, mte, ALPHA, fixed_ca, fixed_te)
            lens_fixed[r] = lf
        stability.append({"seed": seed, "p": pv,
                          "IS_localized_random_sigma": float(np.mean(np.var(lens, axis=0))),
                          "IS_localized_fixed_sigma": float(np.mean(np.var(lens_fixed, axis=0))),
                          "mean_len_random_sigma": float(lens.mean()),
                          "mean_len_fixed_sigma": float(lens_fixed.mean()),
                          "vcp_length": float(2 * q_vcp)})
        mutation.append({"seed": seed, "p": pv,
                         "fixed_sigma_equals_vcp_length": bool(
                             abs(lens_fixed.mean() - 2 * q_vcp) < 1e-9)})

res["equivalence"] = equiv
res["stability"] = stability
res["mutation_fixed_sigma"] = mutation
res["seeds"] = SEEDS
tight = [e for e in equiv if e["eps"] <= 1e-9]
res["max_length_diff_vs_PT_eps_le_1e-9"] = max(e["max_abs_length_diff_vs_PT"] for e in tight)
res["max_Q_gap_vs_base_alpha_prime_eps_le_1e-9"] = max(
    abs(e["Q_localized"] - e["base_q_at_alpha_prime"]) for e in tight)
res["all_localized_shorter_than_vcp"] = all(s["mean_len_random_sigma"] < s["vcp_length"]
                                            for s in stability)
res["min_IS_random_sigma"] = min(s["IS_localized_random_sigma"] for s in stability)
res["max_IS_fixed_sigma"] = max(s["IS_localized_fixed_sigma"] for s in stability)
res["all_mutation_fixed_sigma_equals_vcp"] = all(m["fixed_sigma_equals_vcp_length"] for m in mutation)

json.dump(res, open("results/claim5.json", "w"), indent=2)
print(json.dumps({k: res[k] for k in
                  ["max_length_diff_vs_PT_eps_le_1e-9", "max_Q_gap_vs_base_alpha_prime_eps_le_1e-9",
                   "all_localized_shorter_than_vcp", "min_IS_random_sigma", "max_IS_fixed_sigma",
                   "all_mutation_fixed_sigma_equals_vcp"]}, indent=2))
for e in equiv[:5]:
    print(e)
for s in stability[:4]:
    print(s)
