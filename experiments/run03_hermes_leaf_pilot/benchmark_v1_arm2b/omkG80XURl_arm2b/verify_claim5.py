"""Claim 5 (Theorem 4.4): the SINGLE-chain variant (Eq. 17-18) attains only QUARTIC
condition-number dependence O~(1/(eta^4 T)), vs the double-chain quadratic rate.

Method (mirrors verify_claim1/2): for each instance choose the stepsize that the
theory prescribes to make the steady-state floor ~eps.
  double chain: floor ~ alpha/eta   -> alpha ∝ eps*eta   -> T ∝ 1/(eps*eta^2)
  single chain: floor ~ alpha/eta^3 -> alpha ∝ eps*eta^3 -> T ∝ 1/(eps*eta^4)
Measure hitting time T(eps) vs eta on a log-log fit for BOTH algorithms on the
SAME instances; the exponent gap must be ~2.
"""
import json, numpy as np, common as c

SEED = 20260803
res = {"claim": 5, "source": "Theorem 4.4 (single chain) vs Theorems 4.1-4.2", "seed": SEED}

def family(scale):
    mm = c.make_mrp(n=8, d=3, seed=1, gap=0.6)
    mm['Phi'] = mm['Phi'] * np.array([1.0, 1.0, scale])
    mm['Phi'] = mm['Phi'] / np.max(np.linalg.norm(mm['Phi'], axis=1))
    return mm

EPS, NSEED, TMAX = 0.05, 6, 400000
ETA_REF = 0.11
rows = []
for scale in [1.0, 0.8, 0.65, 0.5]:
    mm = family(scale)
    eta = c.eta1(mm)
    d0 = float(np.sum(c.theta_star(mm) ** 2))
    a_single = min(eta / 18, 0.5 * EPS * d0 * eta * (eta / ETA_REF) ** 2)
    a_double = min(eta / 18, 0.5 * EPS * d0 * eta)
    es = np.mean([c.run_single(mm, TMAX, a_single, seed=5000 + s, markov=True)[1]
                  for s in range(NSEED)], axis=0)
    ed = np.mean([c.run_double(mm, TMAX // 2, a_double, seed=5000 + s, markov=True)[1]
                  for s in range(NSEED)], axis=0)
    tgt = EPS * d0
    rows.append(dict(scale=scale, eta=eta, d0=d0, alpha_single=a_single, alpha_double=a_double,
                     T_single=c.hitting_time(es, tgt), T_double=c.hitting_time(ed, tgt),
                     final_err_single=float(es[-1]), final_err_double=float(ed[-1]), target=tgt))
    print(rows[-1], flush=True)

oks = [r for r in rows if r["T_single"]]
okd = [r for r in rows if r["T_double"]]
ss, _, r2s = c.loglog_slope([r["eta"] for r in oks], [r["T_single"] for r in oks])
sd, _, r2d = c.loglog_slope([r["eta"] for r in okd], [r["T_double"] for r in okd])
res.update(sweep=rows, single_eta_exponent=ss, single_r2=r2s,
           double_eta_exponent=sd, double_r2=r2d,
           exponent_gap=ss - sd, predicted_single=-4.0, predicted_double=-2.0)

res["verdict"] = "verified" if (len(oks) >= 3 and len(okd) >= 3
                                and abs(ss + 4.0) < 1.0 and abs(sd + 2.0) < 0.8
                                and (sd - ss) > 1.0) else "inconclusive"
json.dump(res, open("results/claim5.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "sweep"}, indent=1))
