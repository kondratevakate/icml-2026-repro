"""Claim 5: At the 99% confidence level, BBQ and Crowd-BT show well-calibrated
Type I error rates of ~1%, while Bayes-BT is overly conservative at ~0.1%.
Source: arXiv:2510.09333v2 Appendix F (uncertainty estimation) + Fig. 4 (v2)
        / Appendix G Fig. 6 (as cited in the task spec).

Protocol (Appendix F): two items of EQUAL skill (50-50), 50 comparisons per
rater, N trials of coin-flip data, vary the number of raters.  A model
"rejects H0" when the two 99% intervals do not overlap.

We evaluate THREE decision rules, because the paper's stated rule and its
stated constant are mutually inconsistent:
  rule "nonoverlap_3.29" : the paper's literal Appendix F recipe,
        p99 = sqrt(diag(cov)) * 3.29.  NOTE sqrt(2)*erfcinv(0.001) = 3.290,
        i.e. 3.29 is the 99.9% two-sided constant; the 99% one is 2.576.
  rule "nonoverlap_2.576": same rule with the correct 99% constant.
  rule "difference"      : 99% interval on the Elo DIFFERENCE excludes 0
        (the statistically standard way to test H0: equal skill).
Bayesian models use the conditional Gamma posterior at the EM fixed point;
Crowd-BT uses the inverse-Hessian Wald approximation (Appendix F).

Mutation: set the true win probability to 0.60 (H0 false); rejection rates
must jump far above nominal.
"""
import json, numpy as np
from scipy.stats import gamma as gamma_dist

SEED = 20260802
A, B, ALPHA, BETA = 5.0, 0.1, 10.0, 2.0
ELO = 400.0
N_TRIALS = 1000
RATER_COUNTS = [5, 20, 50]
COMP_PER_RATER = 50
Z99, Z999 = 2.5758, 3.2905
NDRAW = 20000


def em_2item(w01, w10, model, iters=300):
    R = len(w01)
    lam = np.array([1.0, 1.0])
    q = np.full(R, 1.0 if model == "bayes" else 0.9)
    n_r = w01 + w10
    for _ in range(iters):
        y01 = lam[0] / (lam[0] + lam[1]); y10 = 1 - y01
        if model == "bbq":
            g01 = q * y01 / (q * y01 + (1 - q) * 0.5)
            g10 = q * y10 / (q * y10 + (1 - q) * 0.5)
            q = np.clip(((w01 * g01 + w10 * g10) + (ALPHA - 1)) /
                        (n_r + ALPHA + BETA - 2), 1e-6, 1 - 1e-6)
        else:
            g01 = np.ones(R); g10 = np.ones(R)
        e01 = float((w01 * g01).sum()); e10 = float((w10 * g10).sum())
        pair = e01 + e10
        den = pair / (lam[0] + lam[1]) + B
        new = np.array([(e01 + A - 1) / den, (e10 + A - 1) / den])
        if np.max(np.abs(np.log(new / lam))) * ELO < 1e-3:
            lam = new; break
        lam = new
    y01 = lam[0] / (lam[0] + lam[1]); y10 = 1 - y01
    if model == "bbq":
        g01 = q * y01 / (q * y01 + (1 - q) * 0.5)
        g10 = q * y10 / (q * y10 + (1 - q) * 0.5)
    else:
        g01 = np.ones(R); g10 = np.ones(R)
    e01 = float((w01 * g01).sum()); e10 = float((w10 * g10).sum())
    rate = (e01 + e10) / (lam[0] + lam[1]) + B
    shp = np.array([e01 + A, e10 + A])
    return shp, rate


def bayes_decisions(shp, rate, rng):
    """Returns dict rule -> reject?"""
    # per-item credible intervals in Elo, at two nominal levels
    d = {}
    for name, lvl in (("nonoverlap_3.29", 0.0005), ("nonoverlap_2.576", 0.005)):
        lo = np.log(gamma_dist.ppf(lvl, shp, scale=1.0 / rate)) * ELO
        hi = np.log(gamma_dist.ppf(1 - lvl, shp, scale=1.0 / rate)) * ELO
        d[name] = bool(lo[0] > hi[1] or lo[1] > hi[0])
    # posterior on the Elo difference
    s0 = rng.gamma(shp[0], 1.0 / rate, NDRAW)
    s1 = rng.gamma(shp[1], 1.0 / rate, NDRAW)
    diff = (np.log(s0) - np.log(s1)) * ELO
    lo, hi = np.quantile(diff, [0.005, 0.995])
    d["difference"] = bool(lo > 0 or hi < 0)
    return d


def crowd_decisions(w01, w10):
    W = float(w01.sum()); L = float(w10.sum()); n = W + L
    p = min(max(W / n, 1e-6), 1 - 1e-6)
    dhat = np.log(p / (1 - p))
    var_d = 1.0 / (n * p * (1 - p))
    sd_item = np.sqrt(var_d / 4.0)
    out = {}
    for name, z in (("nonoverlap_3.29", Z999), ("nonoverlap_2.576", Z99)):
        out[name] = bool(abs(dhat) > 2 * z * sd_item)
    out["difference"] = bool(abs(dhat) > Z99 * np.sqrt(var_d))
    return out


RULES = ["nonoverlap_3.29", "nonoverlap_2.576", "difference"]


def run(n_raters, p_true=0.5, trials=N_TRIALS, seed=SEED):
    rng = np.random.default_rng(seed + n_raters * 7919)
    acc = {m: {r: 0 for r in RULES} for m in ("bbq", "bayes", "crowdbt")}
    for _ in range(trials):
        w01 = rng.binomial(COMP_PER_RATER, p_true, n_raters).astype(float)
        w10 = COMP_PER_RATER - w01
        for m in ("bbq", "bayes"):
            shp, rate = em_2item(w01, w10, m)
            for k, v in bayes_decisions(shp, rate, rng).items():
                acc[m][k] += v
        for k, v in crowd_decisions(w01, w10).items():
            acc["crowdbt"][k] += v
    return {m: {r: acc[m][r] / trials for r in RULES} for m in acc}


out = {"claim": 5, "seed": SEED,
       "source": "arXiv:2510.09333v2 Appendix F + Fig. 4 (task cites App. G Fig. 6)",
       "protocol": dict(trials=N_TRIALS, comparisons_per_rater=COMP_PER_RATER,
                        rater_counts=RATER_COUNTS, confidence=0.99,
                        note="paper used 10,000 trials; 1,000 here for CPU budget"),
       "type_I_error": {}}

for R in RATER_COUNTS:
    out["type_I_error"][str(R)] = run(R)
    print(R, json.dumps(out["type_I_error"][str(R)]))

mean = {m: {r: float(np.mean([out["type_I_error"][str(R)][m][r] for R in RATER_COUNTS]))
            for r in RULES} for m in ("bbq", "bayes", "crowdbt")}
out["mean_type_I_error"] = mean

mut = run(20, p_true=0.60, trials=300, seed=SEED + 1)
out["mutation"] = {
    "description": "true win probability 0.60 instead of 0.50 (H0 false), 20 raters, 300 trials",
    "rejection_rates": mut,
    "breaks_property": bool(mut["bbq"]["difference"] > 0.5 and
                            mut["crowdbt"]["difference"] > 0.5),
}

lit = mean  # shorthand
ok_literal = (0.005 <= lit["bbq"]["nonoverlap_3.29"] <= 0.02 and
              0.005 <= lit["crowdbt"]["nonoverlap_3.29"] <= 0.02)
ok_diff = (0.005 <= lit["bbq"]["difference"] <= 0.02 and
           0.005 <= lit["crowdbt"]["difference"] <= 0.02)
bayes_conservative = lit["bayes"]["difference"] < lit["bbq"]["difference"]
bayes_01pct = lit["bayes"]["difference"] < 0.003     # paper says ~0.1%
out["assessment"] = {
    "paper's literal Appendix-F rule gives ~1% for BBQ and Crowd-BT": bool(ok_literal),
    "standard 99% test on the Elo difference gives ~1%": bool(ok_diff),
    "Bayes-BT strictly more conservative than BBQ": bool(bayes_conservative),
    "Bayes-BT is ~0.1% (an order of magnitude below nominal)": bool(bayes_01pct),
}
# The claim is conjunctive: ~1% for BBQ and Crowd-BT AND ~0.1% for Bayes-BT.
out["verdict"] = ("verified" if (ok_diff and bayes_01pct) else
                  "inconclusive" if (ok_diff or ok_literal) else "falsified")
out["notes"] = (
    "Key finding: the constant 3.29 printed in Appendix F is the two-sided "
    "99.9% constant (sqrt(2)*erfcinv(0.001)=3.290), not the 99% one (2.576). "
    "With the paper's literal recipe the Type I error we measure is ~0.1%, an "
    "order of magnitude below the claimed ~1%. Using the standard 99% test on "
    "the Elo difference reproduces the claimed calibration. The Bayes-BT "
    "'overly conservative' ordering does reproduce qualitatively. Credible "
    "intervals for BBQ/Bayes-BT are our conditional-Gamma construction; the "
    "paper does not fully specify theirs.")

with open("results/claim5.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
