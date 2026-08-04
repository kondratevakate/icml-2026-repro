"""Claim 5 (Theorem 5.2): randomized ALG^{Rs} for s-bounded (unbounded slackness)
instances achieves an e/(e-1)-regret bound of O~(sqrt(KT)).

Part A (competitive constant): exhaustive worst-case E[OPT]/E[ALG] of the randomized
mixing algorithm over ALL s-bounded instances of length T=3 for s=3 (and s=2 as a
control), optimized over the mixing probability. Claim implies the worst-case ratio
should be at most e/(e-1) = 1.5820.

Part B (regret scaling): stochastic s-bounded environment (s=3, 4), alpha = (e-1)/e,
EXP3 over the theta grid; fit log R = c + a log T + b log K.
Seeds: 20260803 + offsets.
"""
import json, math, time
import numpy as np
from randomized import worst_ratio_randomized
from learning import exp3_alpha_regret, fit_scaling

SEED = 20260803
E_CONST = math.e / (math.e - 1)
Ts = [500, 1000, 2000, 4000, 8000]
Ks = [2, 4, 8, 16]
ARMS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

if __name__ == "__main__":
    t0 = time.time()
    out = {"claim": 5, "source": "Theorem 5.2", "seed": SEED,
           "target_constant_e_over_e_minus_1": E_CONST, "part_A": {}}
    for s in [2, 3]:
        vals = [1.0, 1.6]
        best, curve = (None, 1e9), {}
        for p in np.round(np.arange(0.0, 1.01, 0.125), 3):
            r, _ = worst_ratio_randomized(vals, float(p), 3, max_arrivals=2, sbound=s)
            curve[str(p)] = r
            if r < best[1]:
                best = (float(p), r)
        out["part_A"][f"s{s}"] = {"values": vals, "best_p": best[0],
                                  "worst_case_ratio": best[1], "curve": curve}
        print("A s", s, best, f"{time.time()-t0:.1f}s", flush=True)
    r3 = out["part_A"]["s3"]["worst_case_ratio"]
    out["A_s3_le_e_over_e_minus_1"] = bool(r3 <= E_CONST + 1e-6)
    out["A_s3_ratio"] = r3
    out["A_mutation_p_extreme"] = {k: out["part_A"][k]["curve"]["1.0"] for k in out["part_A"]}
    out["A_mutation_worse"] = bool(out["part_A"]["s3"]["curve"]["1.0"] > r3 + 1e-9)

    out["part_B"] = {}
    for s in [3, 4]:
        grid = {}
        for K in Ks:
            for T in Ts:
                rs = [exp3_alpha_regret(T, K, SEED + 1000 * K + i, 1 / E_CONST, ARMS, s)[0]
                      for i in range(5)]
                grid[f"K{K}_T{T}"] = float(np.mean(rs))
            print("B s", s, "K", K, f"{time.time()-t0:.1f}s", flush=True)
        slack = [exp3_alpha_regret(2000, K, SEED + K, 1 / E_CONST, ARMS, s)[3] for K in Ks]
        fit = fit_scaling(grid, Ks, Ts)
        out["part_B"][f"s{s}"] = {"regret_grid": grid, "fit": fit,
                                  "exp_T_near_half": bool(abs(fit["exp_T"] - 0.5) < 0.15),
                                  "exp_K_near_half": bool(abs(fit["exp_K"] - 0.5) < 0.25),
                                  "alpha_slack_T2000": [float(x) for x in slack],
                                  "alpha_benchmark_met": bool(all(x >= 0 for x in slack))}
    # MUTATION: single bad fixed arm -> linear regret
    mut = {}
    for T in Ts:
        mut[f"T{T}"] = float(np.mean([exp3_alpha_regret(T, 8, SEED + i, 1 / E_CONST,
                                                        [1.0], 3)[0] for i in range(5)]))
    xs = np.log(Ts); ys = np.log([max(mut[f"T{T}"], 1e-6) for T in Ts])
    out["mutation_fixed_bad_threshold"] = {"regret": mut,
                                           "exp_T": float(np.polyfit(xs, ys, 1)[0])}
    out["runtime_s"] = time.time() - t0
    json.dump(out, open("results/claim5.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("A_s3_ratio", "part_B",
                                          "mutation_fixed_bad_threshold")}, indent=2)[:2000])
