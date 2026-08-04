"""Claim 1 (Proposition 3.1): EDF_Phi^L on 2- and 3-bounded K-OPSD has
Phi-regret O~(sqrt(KT)).

Phi-regret := (1/Phi) * OPT_T - ALG_T  (alpha-regret with alpha = 1/Phi).
Learner: EXP3 over a grid of EDF threshold parameters (the learning layer L),
base algorithm EDF with Phi threshold. We measure the empirical Phi-regret for
T in {500..8000}, K in {2,4,8,16}, s in {2,3}, and fit log R = c + a log T + b log K.
Claim predicts a ~ 0.5 and b ~ 0.5 (up to log factors, so a slight excess is allowed).
Seeds: 20260803 + K*1000 + trial.
"""
import json, time
import numpy as np
from common import PHI
from learning import exp3_alpha_regret, fit_scaling

SEED = 20260803
ARMS = [1.0, 1.2, 1.4, PHI, 1.8, 2.0]
Ts = [500, 1000, 2000, 4000, 8000]
Ks = [2, 4, 8, 16]

if __name__ == "__main__":
    t0 = time.time()
    out = {"claim": 1, "source": "Proposition 3.1", "alpha": 1 / PHI, "phi": PHI,
           "seed": SEED, "arms": ARMS, "Ts": Ts, "Ks": Ks, "per_s": {}}
    for s in [2, 3]:
        grid = {}
        for K in Ks:
            for T in Ts:
                rs = [exp3_alpha_regret(T, K, SEED + 1000 * K + i, 1 / PHI, ARMS, s)[0]
                      for i in range(5)]
                grid[f"K{K}_T{T}"] = float(np.mean(rs))
            print("s", s, "K", K, f"{time.time()-t0:.1f}s", flush=True)
        slack = [exp3_alpha_regret(2000, K, SEED + K, 1 / PHI, ARMS, s)[3] for K in Ks]
        fit = fit_scaling(grid, Ks, Ts)
        out["per_s"][str(s)] = {"regret_grid": grid, "fit": fit,
                                "exp_T_near_half": bool(abs(fit["exp_T"] - 0.5) < 0.15),
                                "exp_K_near_half": bool(abs(fit["exp_K"] - 0.5) < 0.25),
                                "all_regret_positive": bool(all(v > 0 for v in grid.values())),
                                "alpha_slack_T2000": [float(x) for x in slack],
                                "alpha_benchmark_met": bool(all(x >= 0 for x in slack))}
    # MUTATION: replace the learning layer by a fixed adversarially-bad threshold
    # (theta = 1.0, i.e. never prioritise the expiring packet). The alpha-regret must
    # grow linearly in T instead of sqrt(T).
    mut = {}
    for T in Ts:
        rs = [exp3_alpha_regret(T, 8, SEED + i, 1 / PHI, [1.0], 2)[0] for i in range(5)]
        mut[f"T{T}"] = float(np.mean(rs))
    xs = np.log(Ts); ys = np.log([max(mut[f"T{T}"], 1e-6) for T in Ts])
    slope = float(np.polyfit(xs, ys, 1)[0])
    out["mutation_fixed_bad_threshold"] = {"regret": mut, "exp_T": slope,
                                           "breaks_sqrt_rate": bool(slope > 0.75)}
    out["runtime_s"] = time.time() - t0
    json.dump(out, open("results/claim1.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("per_s", "mutation_fixed_bad_threshold")},
                     indent=2)[:2000])
