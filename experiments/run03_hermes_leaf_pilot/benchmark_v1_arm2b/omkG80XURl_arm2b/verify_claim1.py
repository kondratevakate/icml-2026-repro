"""Claim 1 (Theorem 4.1): double-chain algorithm, i.i.d. sampling.
(a) converges to a UNIQUE, SAMPLE-INDEPENDENT fixed point of the projected
    Bellman equation (Eq. 13/14);
(b) sample complexity O~(eps^-1 eta^-2)  ->  measured exponent of eta ~= -2.
Mutation: reuse the same sample for the second chain (double-sampling bias)
-> the fixed point must move away from theta*.
"""
import json, numpy as np, common as c

SEED = 20260803
res = {"claim": 1, "source": "Theorem 4.1, Eq. (13)-(16)", "seed": SEED,
       "setup": "n=8 states, d=3 features, normalized ||phi(s)||<=1"}

# ---------- (a) unique, sample-independent fixed point ----------
m = c.make_mrp(n=8, d=3, seed=1)
ts = c.theta_star(m)
res["eta1"] = c.eta1(m)
res["theta_star"] = ts.tolist()
res["theta_star_eq14_maxdiff"] = float(np.max(np.abs(ts - c.theta_star_eq14(m))))
rng = np.random.default_rng(SEED)
tails = []
for k in range(8):                       # different trajectories AND initializations
    th0 = rng.normal(size=m['d']) * 3.0
    _, _, tail = c.run_double(m, 100000, 1e-3, seed=100 + k, theta0=th0)
    tails.append(tail)
tails = np.array(tails)
res["fixedpoint_spread"] = float(np.max(np.std(tails, axis=0)))
res["fixedpoint_max_dist_to_theta_star"] = float(np.max(np.linalg.norm(tails - ts, axis=1)))

mut = np.array([c.run_double(m, 100000, 1e-3, seed=100 + k, coupled=True)[2] for k in range(4)])
res["mutation_coupled_dist_to_theta_star"] = float(np.mean(np.linalg.norm(mut - ts, axis=1)))

# ---------- (b) sample complexity scaling in eta ----------
def family(scale):
    mm = c.make_mrp(n=8, d=3, seed=1)
    mm['Phi'] = mm['Phi'] * np.array([1.0, 1.0, scale])
    mm['Phi'] = mm['Phi'] / np.max(np.linalg.norm(mm['Phi'], axis=1))
    return mm

EPS, NSEED, TMAX = 0.02, 10, 300000
rows = []
for scale in [1.0, 0.7, 0.5, 0.35, 0.25]:
    mm = family(scale)
    eta = c.eta1(mm)
    d0 = float(np.sum(c.theta_star(mm) ** 2))     # theta0 = 0
    alpha = min(eta / 18, 0.5 * EPS * d0 * eta)   # alpha = Theta(eps*eta)
    errs = np.mean([c.run_double(mm, TMAX, alpha, seed=1000 + s)[1] for s in range(NSEED)], axis=0)
    T = c.hitting_time(errs, EPS * d0)
    rows.append(dict(scale=scale, eta=eta, alpha=alpha, d0=d0, T_to_eps=T,
                     final_err=float(errs[-1]), target=EPS * d0))
    print(rows[-1], flush=True)

ok = [r for r in rows if r["T_to_eps"]]
slope, _, r2 = c.loglog_slope([r["eta"] for r in ok], [r["T_to_eps"] for r in ok])
res.update(eta_sweep=rows, eta_exponent_measured=slope, eta_exponent_r2=r2,
           eta_exponent_predicted=-2.0)

res["verdict"] = "verified" if (
    res["fixedpoint_spread"] < 0.05
    and res["fixedpoint_max_dist_to_theta_star"] < 0.1
    and res["mutation_coupled_dist_to_theta_star"] > 0.1
    and len(ok) >= 4 and abs(slope + 2.0) < 0.5) else "inconclusive"
json.dump(res, open("results/claim1.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "eta_sweep"}, indent=1))
