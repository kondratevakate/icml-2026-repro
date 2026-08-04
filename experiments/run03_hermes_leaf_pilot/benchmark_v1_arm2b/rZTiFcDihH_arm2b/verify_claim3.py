"""Claim 3 (Theorem 4.2 + Theorem 4.3): ALG^{theta,U} attains theta_K-regret
O~(sqrt(KT)) (upper bound) and this is nearly tight against Omega(sqrt(T)).

Part A (upper bound): EXP3 over the theta-grid ("U" = unknown value set), alpha =
1/theta_K with theta_K taken from the exhaustive claim-2 computation. Fit
log R = c + a log T + b log K.

Part B (lower bound): two-environment indistinguishability construction. Two
stochastic 2-bounded environments differing only by a value gap Delta; the two
environments have different optimal theta. For each T we sweep Delta and record
max over the two environments of the learner's regret; the max over Delta is a
lower bound on the minimax regret of this learner. We fit its T-exponent, which
should be ~0.5 (Omega(sqrt T)).
Seeds: 20260803 + offsets.
"""
import json, os, time
import numpy as np
from learning import exp3_alpha_regret, fit_scaling, sample_instance, run_alg_theta, opt_dp

SEED = 20260803
Ts = [500, 1000, 2000, 4000, 8000]
Ks = [2, 4, 8, 16]
ARMS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def two_env_regret(T, delta, seed, block=25):
    """Env b in {0,1}: expiring packets are worth (1 +/- delta) relative to future ones.
    Learner: EXP3 over {theta_low=1.0, theta_high=2.0}; only one is right per env."""
    res = []
    for b in (0, 1):
        rng = np.random.default_rng(seed + 7919 * b)
        arms = [1.0, 2.0]
        w = np.ones(2)
        nb = max(1, T // block)
        eta = np.sqrt(np.log(2) / (nb * 2))
        alg = bench = 0.0
        for _ in range(nb):
            p = w / w.sum()
            i = int(rng.choice(2, p=p))
            vals = np.array([1.0, 1.0 + (delta if b else -delta)])
            arr = sample_instance(rng, block, 2, vals, 2)
            a = run_alg_theta(arr, block, arms[i])
            best = max(run_alg_theta(arr, block, th) for th in arms)
            alg += a
            bench += best
            w[i] *= np.exp(eta * (a / max(1.0, opt_dp(arr, block, 2))) / p[i])
            w /= w.max()
        res.append(bench - alg)
    return max(res)


if __name__ == "__main__":
    t0 = time.time()
    theta_K = 1.6
    if os.path.exists("results/claim2.json"):
        c2 = json.load(open("results/claim2.json"))
        theta_K = max(v["worst_case_ratio"] for v in c2["per_K"].values())
    out = {"claim": 3, "source": "Theorem 4.2 (upper), Theorem 4.3 (lower)",
           "seed": SEED, "theta_K_used": theta_K, "alpha": 1 / theta_K}

    grid = {}
    for K in Ks:
        for T in Ts:
            rs = [exp3_alpha_regret(T, K, SEED + 1000 * K + i, 1 / theta_K, ARMS, 2)[0]
                  for i in range(5)]
            grid[f"K{K}_T{T}"] = float(np.mean(rs))
        print("upper K", K, f"{time.time()-t0:.1f}s", flush=True)
    slack = [exp3_alpha_regret(2000, K, SEED + K, 1 / theta_K, ARMS, 2)[3] for K in Ks]
    fit = fit_scaling(grid, Ks, Ts)
    out["part_A_upper"] = {"regret_grid": grid, "fit": fit,
                           "exp_T_near_half": bool(abs(fit["exp_T"] - 0.5) < 0.15),
                           "exp_K_near_half": bool(abs(fit["exp_K"] - 0.5) < 0.25),
                           "alpha_slack_T2000": [float(x) for x in slack],
                           "alpha_benchmark_met": bool(all(x >= 0 for x in slack))}

    lb = {}
    for T in Ts:
        best = 0.0
        for delta in [0.02, 0.05, 0.1, 0.2, 0.4]:
            r = np.mean([two_env_regret(T, delta, SEED + 31 * i) for i in range(3)])
            best = max(best, float(r))
        lb[f"T{T}"] = best
        print("lower T", T, best, f"{time.time()-t0:.1f}s", flush=True)
    xs = np.log(Ts); ys = np.log([max(lb[f"T{T}"], 1e-6) for T in Ts])
    slope = float(np.polyfit(xs, ys, 1)[0])
    out["part_B_lower"] = {"minimax_regret_proxy": lb, "exp_T": slope,
                           "at_least_sqrt": bool(slope > 0.35),
                           "c_over_sqrtT": {k: lb[k] / np.sqrt(int(k[1:])) for k in lb}}
    # MUTATION: remove the indistinguishability (delta = 1.0, huge gap) -> regret must
    # become O(1)-ish / sub-sqrt because the learner identifies the env immediately.
    mut = {f"T{T}": float(np.mean([two_env_regret(T, 1.0, SEED + i) for i in range(3)]))
           for T in Ts}
    ys2 = np.log([max(mut[f"T{T}"], 1e-6) for T in Ts])
    out["mutation_large_gap"] = {"regret": mut, "exp_T": float(np.polyfit(xs, ys2, 1)[0])}
    out["runtime_s"] = time.time() - t0
    json.dump(out, open("results/claim3.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("part_A_upper", "part_B_lower",
                                          "mutation_large_gap")}, indent=2)[:2500])
