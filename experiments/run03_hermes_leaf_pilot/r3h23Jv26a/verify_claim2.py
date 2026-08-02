"""Claim 2: Table 1 row alpha=0.10, p=0.96 -- VCP length 22.894 -> PT-VCP length 22.614,
coverage preserved at ~0.90. Setting: App. D.1.1.

The paper fixes mu=20, alpha, p and "5 random seeds" but does NOT specify beta or the
train/cal/test sizes, and it is ambiguous whether mu=20 means the mixture components are
N(+-20,1) (component means +-20) or that the two components are separated by 20 (i.e. +-10).
We therefore sweep both readings and a small size/beta grid, with 20 seeds each.

Run: .venv/bin/python verify_claim2.py
"""
import json, itertools
import numpy as np
from cp_common import vcp_intervals, pt_vcp, covered

ALPHA, P = 0.10, 0.96
PAPER = {"vcp_len": 22.894, "vcp_len_sem": 0.138, "pt_len": 22.614, "pt_len_sem": 0.254,
         "vcp_cov": 0.906, "pt_cov": 0.909}
SEEDS = list(range(20))


def run(seeds, alpha, p, comp_mean, n, beta):
    b = np.array(beta, float)
    cv, lv, cp, lp = [], [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)

        def gen(m):
            X = rng.normal(size=(m, b.size))
            sign = rng.integers(0, 2, size=m) * 2 - 1
            return X, X @ b + rng.normal(size=m) + sign * comp_mean

        Xtr, ytr = gen(n); Xca, yca = gen(n); Xte, yte = gen(2 * n)
        coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)   # misspecified linear/Gaussian fit
        mca, mte = Xca @ coef, Xte @ coef
        lo, hi, q = vcp_intervals(mca, yca, mte, alpha)
        cv.append(covered(lo, hi, yte).mean()); lv.append(2 * q)
        plo, phi, plen, _ = pt_vcp(mca, yca, mte, alpha, p, rng)
        cp.append(covered(plo, phi, yte).mean()); lp.append(plen.mean())
    m = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
    return {"vcp_cov": m(cv), "vcp_len": m(lv), "pt_cov": m(cp), "pt_len": m(lp)}


res = {"command": ".venv/bin/python verify_claim2.py", "paper_table1_alpha0.10_p0.96": PAPER,
       "seeds": SEEDS, "sweep": []}
for comp_mean, n, beta in itertools.product([10.0, 20.0], [500, 1000, 2000, 5000],
                                            [(1.0, 1.0), (1.0, 2.0)]):
    r = run(SEEDS, ALPHA, P, comp_mean, n, beta)
    r.update({"component_mean": comp_mean, "n_per_fold": n, "beta": list(beta),
              "pt_shorter": r["pt_len"][0] < r["vcp_len"][0],
              "abs_err_vcp_len": abs(r["vcp_len"][0] - PAPER["vcp_len"]),
              "abs_err_pt_len": abs(r["pt_len"][0] - PAPER["pt_len"])})
    res["sweep"].append(r)

best = min(res["sweep"], key=lambda r: r["abs_err_vcp_len"] + r["abs_err_pt_len"])
res["best_matching_config"] = best
res["all_configs_pt_shorter"] = all(r["pt_shorter"] for r in res["sweep"])
res["n_configs"] = len(res["sweep"])
# directional replication with the best config, per-seed detail
res["fraction_of_configs_within_1pct_of_paper_vcp_len"] = sum(
    r["abs_err_vcp_len"] / PAPER["vcp_len"] < 0.01 for r in res["sweep"]) / len(res["sweep"])

json.dump(res, open("results/claim2.json", "w"), indent=2)
print(json.dumps({k: res[k] for k in ["best_matching_config", "all_configs_pt_shorter",
                                      "n_configs", "fraction_of_configs_within_1pct_of_paper_vcp_len"]}, indent=2))
for r in res["sweep"]:
    print(f"mean={r['component_mean']:>5} n={r['n_per_fold']:>5} beta={r['beta']} "
          f"VCP len={r['vcp_len'][0]:.3f}+-{r['vcp_len'][1]:.3f} cov={r['vcp_cov'][0]:.3f} | "
          f"PT len={r['pt_len'][0]:.3f}+-{r['pt_len'][1]:.3f} cov={r['pt_cov'][0]:.3f} | shorter={r['pt_shorter']}")
