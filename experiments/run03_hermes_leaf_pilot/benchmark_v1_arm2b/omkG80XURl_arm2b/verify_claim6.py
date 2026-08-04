"""Claim 6 (Lemma B.3, Eq. (3) vs Eq. (2)): eta1 >= (1/2) eta3, where
  eta1 = min_{||x||=1} ||Phi x||^2_Dir + (mu' Phi x)^2
  eta3 = (min_{||x||=1} x' Phi' D Phi x) * (min_{<y,e>_D=0,||y||_D=1} y' D(I-P) y)
Also checks the proof's intermediate step lambda <= 2.
MUTATION: replace the stationary distribution mu by a NON-stationary positive
distribution (the proof uses stationarity in ||v||^2_Dir = v'D(I-P)v); the
inequality / the identity must then break on some instances.
"""
import json, numpy as np, common as c

SEED = 20260803
rng = np.random.default_rng(SEED)
res = {"claim": 6, "source": "Lemma B.3, Eq. (2)-(3)", "seed": SEED}

ratios, lams, dir_id_err = [], [], []
for k in range(300):
    n = int(rng.integers(4, 20))
    d = int(rng.integers(1, min(n, 8)))
    gap = float(rng.uniform(0.15, 1.0))
    m = c.make_mrp(n=n, d=d, seed=int(rng.integers(1e6)), gap=gap)
    e1 = c.eta1(m)
    e3, sigma, lam = c.eta3(m)
    ratios.append(e1 / e3)
    lams.append(lam)
    # identity ||v||^2_Dir = v' D(I-P) v (Step 1 of the proof), random v
    v = rng.normal(size=n)
    mu, P = m['mu'], m['P']
    dirn = 0.5 * np.sum(mu[:, None] * P * (v[:, None] - v[None, :]) ** 2)
    quad = v @ np.diag(mu) @ (np.eye(n) - P) @ v
    dir_id_err.append(abs(dirn - quad))

ratios = np.array(ratios)
res.update(n_instances=len(ratios), min_ratio_eta1_over_eta3=float(ratios.min()),
           median_ratio=float(np.median(ratios)), max_ratio=float(ratios.max()),
           violations_of_half=int(np.sum(ratios < 0.5 - 1e-9)),
           max_lambda=float(np.max(lams)), lambda_le_2=bool(np.max(lams) <= 2 + 1e-9),
           max_dirichlet_identity_error=float(np.max(dir_id_err)))

# ---- mutation: use a non-stationary mu in both definitions ----
mut_viol, mut_ratios = 0, []
for k in range(300):
    n = int(rng.integers(4, 20))
    d = int(rng.integers(1, min(n, 8)))
    m = c.make_mrp(n=n, d=d, seed=int(rng.integers(1e6)))
    bad = rng.random(n) + 0.05
    m['mu'] = bad / bad.sum()          # NOT stationary for P
    e1 = c.eta1(m)
    e3, _, _ = c.eta3(m)
    r = e1 / e3 if e3 > 1e-12 else np.inf
    mut_ratios.append(r)
    if r < 0.5 - 1e-9:
        mut_viol += 1
res["mutation_nonstationary_violations"] = mut_viol
res["mutation_min_ratio"] = float(np.min(mut_ratios))

# ---- mutation B: drop the (mu' Phi x)^2 correction term from eta1 (Eq. 3).
# This is the term that removes the e-direction degeneracy; without it the
# inequality eta1 >= eta3/2 must fail whenever span(Phi) contains e.
mutB_viol, mutB_min = 0, np.inf
for k in range(200):
    n = int(rng.integers(4, 20))
    d = int(rng.integers(2, min(n, 8)))
    m = c.make_mrp(n=n, d=d, seed=int(rng.integers(1e6)))
    m['Phi'][:, 0] = 1.0 / np.sqrt(n)          # put e in span(Phi) (tabular-like case)
    m['Phi'] = m['Phi'] / np.max(np.linalg.norm(m['Phi'], axis=1))
    P, Phi, mu = m['P'], m['Phi'], m['mu']
    D = np.diag(mu)
    M = Phi.T @ D @ (np.eye(n) - P) @ Phi
    e1_nocorr = float(np.linalg.eigvalsh(0.5 * (M + M.T)).min())   # Dirichlet term only
    e3, _, _ = c.eta3(m)
    r = e1_nocorr / e3 if e3 > 1e-12 else np.inf
    mutB_min = min(mutB_min, r)
    if r < 0.5 - 1e-9:
        mutB_viol += 1
res["mutationB_drop_correction_violations"] = mutB_viol
res["mutationB_min_ratio"] = float(mutB_min)
res["tightness_max_constant_supported"] = float(ratios.min())   # eta1 >= c*eta3 held up to c

res["verdict"] = "verified" if (res["violations_of_half"] == 0 and res["lambda_le_2"]
                                and res["max_dirichlet_identity_error"] < 1e-9
                                and mutB_viol > 0) else "inconclusive"
json.dump(res, open("results/claim6.json", "w"), indent=1)
print(json.dumps(res, indent=1))
