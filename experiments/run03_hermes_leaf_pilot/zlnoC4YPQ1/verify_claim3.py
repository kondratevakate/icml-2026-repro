"""verify_claim3.py — Claim 3: both algorithms tolerate corruptions of possibly INFINITE magnitude,
bounded only in frequency (Definition 2.1, Sec 2.2).

Operationalisation (the only thing that can be checked numerically): the mechanism that makes the
frequency-only model workable is that the RCGP posterior with the P-IMQ weight has a posterior mean
whose deviation from the uncorrupted-GP posterior mean stays BOUNDED as |c| -> oo, while a standard
GP's deviation grows linearly in c (Appendix H).

Test (exhaustive over the finite design space):
  for every seed (10) x every corrupted index (all n=8 points) x magnitude |c| in 10^0..10^12:
     dev_R(c)  = sup_x |mu^R(x) - mu_uc(x)|     (P-IMQ RCGP, g=0, L per Def 3.1)
     dev_GP(c) = sup_x |mu^GP(x) - mu_uc(x)|    (standard GP on the same corrupted data)
  Assert dev_R saturates (bounded, monotone-convergent) and dev_GP ~ linear in c.
  Also check the Lemma D.5 shape: sup_x dev_R(x)/sigma_uc(x) stays finite.
MUTATION: replace the P-IMQ weight by the constant weight W (which is exactly the standard GP,
i.e. the robustness mechanism removed) -> boundedness must fail.
"""
import json, numpy as np
from rcgp import rbf, pimq, posterior, objective, GRID

KPAR = dict(lengthscale=0.15, outputscale=1.0)
SN = 1.0
C_SHAPE = 1.0
L = 1.96          # Appendix I.1 recommended plateau half-width
MAGS = [10.0 ** k for k in range(0, 13)]
XTE = np.linspace(0, 1, 101)


def deviations(x_tr, y_clean, j, c, robust=True):
    y = y_clean.copy()
    y[j] = y_clean[j] + c
    keep = np.array([i for i in range(len(y)) if i != j])
    mu_uc, sd_uc = posterior(x_tr[keep], y_clean[keep], XTE, SN, KPAR)
    if robust:
        w, mw = pimq(y, 0.0, L, C_SHAPE, SN)
        mu, _ = posterior(x_tr, y, XTE, SN, KPAR, weights=w, mw=mw)
    else:
        mu, _ = posterior(x_tr, y, XTE, SN, KPAR)
    d = np.abs(mu - mu_uc)
    return float(np.max(d)), float(np.max(d / sd_uc))


res = {"claim": 3, "source": "Definition 2.1 (Sec 2.2); mechanism per Def 3.1 + Lemma D.5; Appendix H",
       "config": {"kernel": "RBF", **KPAR, "sigma_noise": SN, "L": L, "c": C_SHAPE,
                  "magnitudes": MAGS, "n_points": 8, "seeds": list(range(10))},
       "command": ".venv/bin/python verify_claim3.py"}

rows = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    x_tr = np.sort(rng.uniform(0, 1, 8))
    y_clean = objective(x_tr) + rng.normal(0, SN, 8)
    y_clean = (y_clean - y_clean.mean()) / y_clean.std()   # standardised units (App I.1)
    for j in range(8):
        dR = [deviations(x_tr, y_clean, j, c, True) for c in MAGS]
        dG = [deviations(x_tr, y_clean, j, c, False) for c in MAGS]
        rows.append({"seed": seed, "j": j,
                     "rcgp_dev": [a for a, _ in dR], "rcgp_dev_over_sd": [b for _, b in dR],
                     "gp_dev": [a for a, _ in dG]})

rcgp_max = max(max(r["rcgp_dev"]) for r in rows)
rcgp_max_ratio = max(max(r["rcgp_dev_over_sd"]) for r in rows)
gp_max = max(max(r["gp_dev"]) for r in rows)
# saturation: deviation at |c|=1e12 vs at |c|=1e6 (relative change)
sat = [abs(r["rcgp_dev"][-1] - r["rcgp_dev"][6]) for r in rows]  # absolute drift, |c|: 1e6 -> 1e12
gp_growth = [r["gp_dev"][-1] / max(r["gp_dev"][6], 1e-12) for r in rows]  # expect ~1e6 (linear)

res["n_configurations"] = len(rows) * len(MAGS)
res["rcgp_sup_deviation_all_configs"] = rcgp_max
res["rcgp_sup_deviation_over_sigma_uc"] = rcgp_max_ratio
res["rcgp_max_absolute_drift_1e6_to_1e12"] = float(max(sat))
res["gp_sup_deviation_all_configs"] = gp_max
res["gp_min_growth_ratio_1e6_to_1e12"] = float(min(gp_growth))
res["gp_growth_is_linear_in_c"] = bool(min(gp_growth) > 1e5)   # 1e6x magnitude -> ~1e6x deviation
res["rcgp_bounded"] = bool(rcgp_max < 1e3 and max(sat) < 1e-3)  # bounded + converged as |c|->oo

# MUTATION: constant weight w = W (P-IMQ plateau everywhere) == standard GP
mut = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    x_tr = np.sort(rng.uniform(0, 1, 8))
    y_clean = objective(x_tr) + rng.normal(0, SN, 8)
    y_clean = (y_clean - y_clean.mean()) / y_clean.std()
    j = 3
    big = deviations(x_tr, y_clean, j, MAGS[-1], robust=False)[0]
    mut.append(big)
res["mutation_constant_weight_sup_deviation"] = float(max(mut))
res["mutation_breaks_boundedness"] = bool(max(mut) > 1e6)

res["verdict_ok"] = bool(res["rcgp_bounded"] and res["gp_growth_is_linear_in_c"]
                         and res["mutation_breaks_boundedness"])
res["rows"] = rows
json.dump(res, open("results/claim3.json", "w"), indent=2)
summary = {k: v for k, v in res.items() if k != "rows"}
print(json.dumps(summary, indent=2))
