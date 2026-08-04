"""Claim 3 (Theorem 4.3): double-chain, Markov sampling, DECAYING stepsize
alpha_t = a/(t+c0) with a = Theta(1/eta): converges, and the bound carries
NO EXPLICIT DIMENSION-DEPENDENT TERMS (d enters only through ||theta*||,||theta0||).

Executable test: sweep d with normalized features (max_s ||phi(s)||<=1, the paper's
assumption). Compare the dimension-normalized error E||theta_T-theta*||^2 / ||theta*||^2
at fixed eta-scaled horizon; regress log(normalized error) on log d -> slope ~ 0.
MUTATION: drop the normalization (entries of phi O(1), so ||phi(s)||=O(sqrt d)),
which the paper says re-introduces an O(d) factor -> slope must become clearly positive.
"""
import json, numpy as np, common as c

SEED = 20260803
res = {"claim": 3, "source": "Theorem 4.3 (xi=1) + Table-1 note (4) on dimension scaling",
       "seed": SEED}

def sweep(normalize):
    out = []
    for d in [2, 4, 8, 16, 32]:
        mm = c.make_mrp(n=64, d=d, seed=7, gap=0.8, normalize=normalize)
        eta = c.eta1(mm)
        ts = c.theta_star(mm)
        nrm = float(np.sum(ts ** 2))
        T = 60000
        a = 4.0 / eta                                     # a = Theta(1/eta)
        c0 = max(1000.0, 20 * a)                          # c0 large enough for stability
        seq = a / (np.arange(T) + c0)
        errs = np.mean([c.run_double(mm, T, seq, seed=4000 + s, markov=True)[1]
                        for s in range(5)], axis=0)
        out.append(dict(d=d, eta=eta, theta_star_sq=nrm, a=a, c0=c0,
                        alpha_0=float(seq[0]), alpha_T=float(seq[-1]),
                        err_T=float(errs[-1]), err_T_normalized=float(errs[-1] / nrm),
                        decay_ratio=float(errs[-1] / errs[0])))
        print(('norm' if normalize else 'MUT '), out[-1], flush=True)
    return out

main = sweep(True)
slope, _, r2 = c.loglog_slope([r["d"] for r in main], [r["err_T_normalized"] for r in main])
res["sweep_normalized_features"] = main
res["d_exponent_measured"] = slope
res["d_exponent_r2"] = r2
res["all_converged"] = bool(all(r["decay_ratio"] < 0.1 for r in main))

mut = sweep(False)
mslope, _, mr2 = c.loglog_slope([r["d"] for r in mut], [r["err_T_normalized"] for r in mut])
res["mutation_unnormalized_features"] = mut
res["mutation_d_exponent"] = mslope

res["verdict"] = "verified" if (res["all_converged"] and abs(slope) < 0.35
                                and mslope > slope + 0.3) else "inconclusive"
json.dump(res, open("results/claim3.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.startswith(("sweep", "mutation_unn"))}, indent=1))
