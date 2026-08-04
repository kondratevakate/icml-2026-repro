"""Claim 2 (Theorem 4.2): double-chain, MARKOVIAN sampling (two independent chains),
constant stepsize -> same O~(eps^-1 eta^-2) sample complexity.
Mutation: reuse the SAME chain for both roles (coupled) -> fixed point is biased.
"""
import json, numpy as np, common as c

SEED = 20260803
res = {"claim": 2, "source": "Theorem 4.2", "seed": SEED,
       "setup": "n=8 states, d=3 features, Markov sampling, gap=0.6 (non-trivial mixing)"}

def family(scale):
    mm = c.make_mrp(n=8, d=3, seed=1, gap=0.6)
    mm['Phi'] = mm['Phi'] * np.array([1.0, 1.0, scale])
    mm['Phi'] = mm['Phi'] / np.max(np.linalg.norm(mm['Phi'], axis=1))
    return mm

EPS, NSEED, TMAX = 0.02, 8, 300000
rows = []
for scale in [1.0, 0.7, 0.5, 0.35]:
    mm = family(scale)
    eta = c.eta1(mm)
    d0 = float(np.sum(c.theta_star(mm) ** 2))
    alpha = min(eta / 18, 0.5 * EPS * d0 * eta)
    errs = np.mean([c.run_double(mm, TMAX, alpha, seed=2000 + s, markov=True)[1]
                    for s in range(NSEED)], axis=0)
    T = c.hitting_time(errs, EPS * d0)
    rows.append(dict(scale=scale, eta=eta, alpha=alpha, d0=d0, T_to_eps=T,
                     final_err=float(errs[-1]), target=EPS * d0))
    print(rows[-1], flush=True)

ok = [r for r in rows if r["T_to_eps"]]
slope, _, r2 = c.loglog_slope([r["eta"] for r in ok], [r["T_to_eps"] for r in ok])
res.update(eta_sweep=rows, eta_exponent_measured=slope, eta_exponent_r2=r2,
           eta_exponent_predicted=-2.0)

# convergence to the same theta* under Markov noise + mutation
mm = family(1.0)
ts = c.theta_star(mm)
tails = np.array([c.run_double(mm, 150000, 1e-3, seed=300 + k, markov=True)[2] for k in range(5)])
res["markov_fixedpoint_max_dist_to_theta_star"] = float(np.max(np.linalg.norm(tails - ts, axis=1)))
res["markov_fixedpoint_spread"] = float(np.max(np.std(tails, axis=0)))
mut = np.array([c.run_double(mm, 150000, 1e-3, seed=300 + k, markov=True, coupled=True)[2]
                for k in range(3)])
res["mutation_coupled_dist_to_theta_star"] = float(np.mean(np.linalg.norm(mut - ts, axis=1)))

res["verdict"] = "verified" if (
    len(ok) >= 3 and abs(slope + 2.0) < 0.6
    and res["markov_fixedpoint_max_dist_to_theta_star"] < 0.1
    and res["mutation_coupled_dist_to_theta_star"] > 0.1) else "inconclusive"
json.dump(res, open("results/claim2.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "eta_sweep"}, indent=1))
