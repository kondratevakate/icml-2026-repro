"""verify_claim1.py — Theorem 3.5 (CAffNet, OpenReview 20hdQQQrA4).

CLAIM: if the unconstrained network class universally approximates F, then CAffNet
universally approximates the constrained class {f_t in F | A(x) f_t <= b(x)}, with
    ||P*(x) - f_t(x)||_p < (3 + 3*sqrt(n_out)) * K
where K is the unconstrained approximation error.

OPERATIONALISATION: the theorem is quantified over the *layer*, not over a trained
network, so no training is needed (skill §2l). We take a feasible target f_t, form an
"unconstrained approximator" f_theta = f_t + eta with ||eta||_inf <= K, push it through
the CAffine layer, and measure
    ratio = ||P(f_theta) - f_t||_2 / K   vs   bound = 3 + 3*sqrt(n_out).
The claim is that ratio < bound for every instance, and (non-vacuity) that the layer's
output is feasible.

BRANCH COVERAGE (P25): half of the instances place f_t exactly on an active constraint,
so the K-perturbation is infeasible roughly half the time and the projection branch
(Case 2) is actually exercised. n_case2 is reported.

MUTATION: replace the CAffine enumeration with the HardNet-Aff single pseudo-inverse
correction on the identical instances. PREDICTION (stated before running): the baseline
produces infeasible outputs at a non-zero rate on rank-deficient constraint systems,
whereas CAffNet does not; its error ratio may also exceed the Theorem-3.5 bound.

BUDGET: fixed modest setting — 200 trials x 3 seeds x 4 cells (no sweep).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from caff_core import (caffine, feasible, feasible_point, hardnet_aff,
                       max_violation, random_constraints)

TRIALS = 200
SEEDS = [0, 1, 2]
CELLS = [(2, 3), (3, 4), (4, 4), (5, 3)]      # (m constraints, n_out)
KS = [0.05, 0.2, 1.0]

CMD = "python verify_claim1.py"


def main():
    t0 = time.time()
    rows = []
    n_case2 = 0
    worst_ratio = 0.0
    n_bound_violations = 0
    n_infeasible_caff = 0
    n_infeasible_hardnet = 0
    hardnet_maxviol = 0.0
    n_hardnet_ratio_over_bound = 0
    total = 0

    for (m, n) in CELLS:
        bound = 3.0 + 3.0 * np.sqrt(n)
        cell_worst = 0.0
        cell_case2 = 0
        for seed in SEEDS:
            rng = np.random.default_rng([seed, m, n])
            for t in range(TRIALS):
                rank_def = (t % 2 == 0)
                A, b = random_constraints(rng, m, n, rank_deficient=rank_def)
                f_t = feasible_point(A, b, rng, on_boundary=(t % 2 == 0))
                if not feasible(A, b, f_t):
                    continue
                K = KS[t % len(KS)]
                eta = rng.uniform(-K, K, size=n)
                f_theta = f_t + eta
                y, info = caffine(A, b, f_theta, w=None, return_info=True)
                total += 1
                if not feasible(A, b, f_theta):
                    n_case2 += 1
                    cell_case2 += 1
                if not feasible(A, b, y):
                    n_infeasible_caff += 1
                ratio = float(np.linalg.norm(y - f_t) / K)
                worst_ratio = max(worst_ratio, ratio)
                cell_worst = max(cell_worst, ratio)
                if ratio >= bound:
                    n_bound_violations += 1
                # mutation arm on the identical instance
                yh = hardnet_aff(A, b, f_theta)
                if not feasible(A, b, yh):
                    n_infeasible_hardnet += 1
                    hardnet_maxviol = max(hardnet_maxviol, max_violation(A, b, yh))
                if float(np.linalg.norm(yh - f_t) / K) >= bound:
                    n_hardnet_ratio_over_bound += 1
        rows.append({"m": m, "n_out": n, "bound": float(bound),
                     "max_ratio": cell_worst, "n_case2": cell_case2})

    res = {
        "claim": 1,
        "source": "Theorem 3.5",
        "command": CMD,
        "config": {"trials_per_seed_per_cell": TRIALS, "seeds": SEEDS,
                   "cells_m_nout": CELLS, "K_values": KS,
                   "note": "fixed modest setting; no sweep (budget rule)"},
        "n_instances": total,
        "n_case2_projected": n_case2,
        "max_ratio_over_all": worst_ratio,
        "n_bound_violations": n_bound_violations,
        "n_infeasible_caffnet": n_infeasible_caff,
        "per_cell": rows,
        "mutation_hardnet_aff": {
            "prediction": "single pseudo-inverse correction leaves some instances infeasible",
            "n_infeasible": n_infeasible_hardnet,
            "max_violation": hardnet_maxviol,
            "n_error_ratio_over_bound": n_hardnet_ratio_over_bound,
        },
        "verdict": None,
        "wall_seconds": None,
    }
    res["verdict"] = ("verified" if (n_bound_violations == 0 and n_infeasible_caff == 0
                                     and n_case2 > 100) else "falsified")
    res["wall_seconds"] = round(time.time() - t0, 2)
    Path("results").mkdir(exist_ok=True)
    Path("results/claim1.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_cell"}, indent=2))


if __name__ == "__main__":
    main()
