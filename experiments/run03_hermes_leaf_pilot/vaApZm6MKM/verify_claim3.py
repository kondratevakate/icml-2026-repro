"""verify_claim3.py — Claim 3 (Figure 4, Section 4.2 of v1 / Figure 1, Sec 4.2 of v2).

Claim: in the synthetic experiment, group-based metrics like CovGap remain unreliable and
unaligned with their theoretical values even at 5,000 test points, whereas ERT metrics such as
L1-ERT converge with far fewer samples.

Setup reproduced from the paper: X ~ U([-1,1]^8), Y ~ N(0, sigma(X1)), sigma(x)=0.5+|x|+x^2.
  * "not conditional": standard split CP with score S(X,Y)=|Y| calibrated on 3,000 iid samples
    => C(x) = [-q, q]; p(x) = 2*Phi(q/sigma(x1)) - 1.
  * "conditional": oracle sets = true conditional alpha/2 and 1-alpha/2 quantiles => p(x) = 1-alpha.
alpha = 0.1. ERT via Algorithm 1 with 5-fold CV and a LightGBM classifier. CovGap with k-means
groups (10 groups; 5 and 20 also recorded). Theoretical values computed analytically from p(x)
with 300,000 Monte-Carlo samples of X (as in the paper's caption).

MUTATION: run Algorithm 1 with the PartitionWise (k-means) classifier instead of LightGBM.
Prediction: the ERT then degrades to CovGap-like behaviour (fails to separate the two scenarios),
showing the classifier, not the ERT bookkeeping, is the mechanism.

Run: .venv/bin/python verify_claim3.py
"""
import json
import numpy as np
from scipy.stats import norm
from ertlib import algorithm1, covgap

ALPHA = 0.1
T = 1 - ALPHA
D = 8
SIZES = [250, 500, 1000, 2000, 5000]
SEEDS = list(range(10))


def sigma(x1):
    return 0.5 + np.abs(x1) + x1 ** 2


def sample_X(n, rng):
    return rng.uniform(-1, 1, size=(n, D))


def calibrate_q(rng, n_cal=3000):
    X = sample_X(n_cal, rng)
    Y = rng.normal(0.0, sigma(X[:, 0]))
    s = np.abs(Y)
    k = int(np.ceil((n_cal + 1) * T))
    return float(np.sort(s)[k - 1])


def p_standard(x1, q):
    return 2 * norm.cdf(q / sigma(x1)) - 1


def theoretical(q, rng, n=300_000):
    X = sample_X(n, rng)
    p = p_standard(X[:, 0], q)
    return {"L1": float(np.mean(np.abs(p - T))), "L2": float(np.mean((T - p) ** 2)),
            "CovGap_target_same_as_L1": float(np.mean(np.abs(p - T)))}


rows = []
theo = []
for s in SEEDS:
    rng = np.random.default_rng(12345 + s)
    q = calibrate_q(rng)
    th = theoretical(q, np.random.default_rng(999_000 + s))
    theo.append(th)
    for n in SIZES:
        X = sample_X(n, rng)
        Y = rng.normal(0.0, sigma(X[:, 0]))
        Z_nc = (np.abs(Y) <= q).astype(float)                      # standard CP sets
        lo = norm.ppf(ALPHA / 2) * sigma(X[:, 0])
        hi = norm.ppf(1 - ALPHA / 2) * sigma(X[:, 0])
        Z_c = ((Y >= lo) & (Y <= hi)).astype(float)                # oracle conditional sets
        print("seed",s,"n",n,flush=True)
        for scen, Z in [("not_conditional", Z_nc), ("conditional", Z_c)]:
            ert = algorithm1(X, Z, T, losses=("L1", "L2"), k=5, clf_kind="gb", seed=s)
            ert_pw = algorithm1(X, Z, T, losses=("L1", "L2"), k=5, clf_kind="partitionwise",
                                seed=s, n_groups=10)
            rows.append({
                "seed": s, "n": n, "scenario": scen, "q": q,
                "L1_ERT": ert["L1"], "L2_ERT": ert["L2"],
                "L1_ERT_partitionwise_mutation": ert_pw["L1"],
                "CovGap_k10": covgap(X, Z, T, 10, seed=s),
                "CovGap_k5": covgap(X, Z, T, 5, seed=s),
                "CovGap_k20": covgap(X, Z, T, 20, seed=s),
                "marginal_coverage": float(Z.mean()),
                "theoretical_L1_not_conditional": th["L1"],
            })

theo_L1 = float(np.mean([t["L1"] for t in theo]))
theo_L2 = float(np.mean([t["L2"] for t in theo]))


def agg(n, scen, key):
    v = [r[key] for r in rows if r["n"] == n and r["scenario"] == scen]
    return float(np.mean(v)), float(np.std(v))


summary = {}
for n in SIZES:
    e = {}
    for key in ["L1_ERT", "L2_ERT", "CovGap_k10", "CovGap_k5", "CovGap_k20",
                "L1_ERT_partitionwise_mutation", "marginal_coverage"]:
        m1, s1 = agg(n, "not_conditional", key)
        m2, s2 = agg(n, "conditional", key)
        e[key] = {"not_conditional_mean": m1, "not_conditional_std": s1,
                  "conditional_mean": m2, "conditional_std": s2,
                  "separation": m1 - m2}
    summary[f"n={n}"] = e

n5 = summary["n=5000"]
checks = {
    "theoretical_L1_not_conditional": theo_L1,
    "theoretical_L1_conditional": 0.0,
    # (a) CovGap is unaligned with its theoretical target (it estimates the same E|p-(1-a)|)
    "covgap_k10_at_5000_not_conditional": n5["CovGap_k10"]["not_conditional_mean"],
    "covgap_relative_error_vs_theory_at_5000": abs(
        n5["CovGap_k10"]["not_conditional_mean"] - theo_L1) / theo_L1,
    "covgap_unaligned_at_5000_(rel_err>0.5)": abs(
        n5["CovGap_k10"]["not_conditional_mean"] - theo_L1) / theo_L1 > 0.5,
    # (b) CovGap gives nearly identical diagnostics in the two very different scenarios
    "covgap_separation_at_5000": n5["CovGap_k10"]["separation"],
    "covgap_nearly_identical_at_5000_(|sep|<0.02)": abs(n5["CovGap_k10"]["separation"]) < 0.02,
    # (c) L1-ERT separates the scenarios and tracks the theoretical value already at small n
    "L1_ERT_separation_at_5000": n5["L1_ERT"]["separation"],
    "L1_ERT_at_5000_not_conditional": n5["L1_ERT"]["not_conditional_mean"],
    "L1_ERT_at_5000_conditional": n5["L1_ERT"]["conditional_mean"],
    "L1_ERT_separates_at_5000": n5["L1_ERT"]["separation"] > 5 * abs(n5["CovGap_k10"]["separation"]),
    "L1_ERT_separation_at_500": summary["n=500"]["L1_ERT"]["separation"],
    "L1_ERT_already_separates_at_500": (
        summary["n=500"]["L1_ERT"]["separation"] > 0.02
        and summary["n=500"]["L1_ERT"]["separation"] >
        3 * abs(summary["n=500"]["CovGap_k10"]["separation"])),
    # MUTATION: PartitionWise classifier inside the ERT
    "mutation_partitionwise_L1_ERT_separation_at_5000":
        n5["L1_ERT_partitionwise_mutation"]["separation"],
    "mutation_degrades_separation": (
        n5["L1_ERT_partitionwise_mutation"]["separation"] < 0.5 * n5["L1_ERT"]["separation"]),
}

out = {"alpha": ALPHA, "d": D, "sizes": SIZES, "seeds": SEEDS,
       "theoretical_L1_mean_over_seeds": theo_L1, "theoretical_L2_mean_over_seeds": theo_L2,
       "summary": summary, "checks": checks,
       "all_checks_pass": bool(checks["covgap_unaligned_at_5000_(rel_err>0.5)"]
                               and checks["covgap_nearly_identical_at_5000_(|sep|<0.02)"]
                               and checks["L1_ERT_separates_at_5000"]
                               and checks["L1_ERT_already_separates_at_500"]
                               and checks["mutation_degrades_separation"]),
       "rows": rows,
       "command": ".venv/bin/python verify_claim3.py"}
with open("results/claim3.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
