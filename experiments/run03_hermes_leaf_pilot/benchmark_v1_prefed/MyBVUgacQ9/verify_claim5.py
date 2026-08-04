"""verify_claim5.py -- CLAIM 5 (Section 5, Figures 2 and 4).

Two halves:

  (i) TESTABLE: "toy-problem experiments show q >= 10 is sufficient to transition from
      sublinear to linear convergence" (Figure 2).  The toy problem is fully specified
      in Section 5:
          min (mu_x/2)(x1^2+x2^2+x3^2+x4^2) + (mu_z/2) z^2
          s.t. x1 x2 - x3 x4 + q z + 1 = 0
      but mu_x, mu_z and rho are NOT reported.  We therefore sweep q over a grid and,
      for each q, report the contraction factor for a grid of (mu_x, mu_z, rho) and
      several initialisations, i.e. we give the claim the best possible chance:
      if ANY reasonable hyperparameter choice shows a sublinear->linear transition at
      q ~ 10 we will see it, and if the ordering is systematically reversed we can say so.

  (ii) NOT TESTABLE from the released material: Figure 4's comparison against PADMM
      (Yashtini 2021), IPDS-ADMM (Yuan 2025) and IADMM (Tang & Toh 2024).  There is no
      code release, no problem instance, no dimension, no hyperparameters (proximal /
      dual step sizes), no seeds and no initialisation given for those baselines.  Any
      numbers we produced would be OUR baselines, not theirs, so we record the reason
      and refuse rather than fabricate a comparison.

Run:  .venv/bin/python verify_claim5.py
"""
import json
import time

import numpy as np

import admm_lib as A

OUT = "results/claim5.json"
CMD = ".venv/bin/python verify_claim5.py"


def factor_for(q, mu_x, mu_z, rho, seeds=range(6), iters=1200):
    prob = A.toy_problem(q, mu_x=mu_x, mu_z=mu_z)
    facs, its = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        x0 = rng.normal(size=4) * 2
        z0 = rng.normal(size=1)
        x, z, w, h = prob.run(x0, z0, np.zeros(1), rho, iters=iters)
        r = h["res"]
        if not np.all(np.isfinite(np.concatenate([x, z, w]))):
            facs.append(float("inf")); its.append(iters); continue
        f, npts, r2 = A.contraction_factor(r, floor=1e-14)
        if not np.isfinite(f):
            idx = np.where(r > 1e-14)[0]
            f = float(np.exp((np.log(r[idx[-1]]) - np.log(r[idx[0]])) /
                             max(1, idx[-1] - idx[0]))) if len(idx) >= 3 else 0.0
        facs.append(float(f))
        hit = np.where(r < 1e-10)[0]
        its.append(int(hit[0]) if len(hit) else iters)
    return {"worst_factor": float(np.max(facs)), "median_factor": float(np.median(facs)),
            "worst_iters_to_1e-10": int(np.max(its)), "iters_cap": iters}


def main():
    t0 = time.time()
    res = {"claim": 5, "source": "Section 5, Figures 2 and 4", "command": CMD}

    qs = [0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 50.0]
    hyper = [(1.0, 1.0, 1.0), (1.0, 1.0, 4.0), (1.0, 1.0, 0.25),
             (1.0, 5.0, 4.0), (5.0, 1.0, 4.0), (1.0, 0.2, 1.0)]
    table = []
    for q in qs:
        cells = []
        for (mx, mz, rho) in hyper:
            c = factor_for(q, mx, mz, rho)
            c.update({"mu_x": mx, "mu_z": mz, "rho": rho})
            cells.append(c)
        best = min(cells, key=lambda c: c["worst_factor"])
        table.append({"q": q, "cells": cells,
                      "best_worst_factor": best["worst_factor"],
                      "best_hyper": {k: best[k] for k in ("mu_x", "mu_z", "rho")},
                      "best_worst_iters_to_1e-10": min(c["worst_iters_to_1e-10"]
                                                       for c in cells)})
    res["q_sweep"] = table

    below = [r for r in table if r["q"] < 10]
    at_or_above = [r for r in table if r["q"] >= 10]
    res["transition_analysis"] = {
        "best_factor_below_q10": {str(r["q"]): r["best_worst_factor"] for r in below},
        "best_factor_at_or_above_q10": {str(r["q"]): r["best_worst_factor"]
                                        for r in at_or_above},
        "all_q_ge_10_linear_factor_lt_0p999": all(r["best_worst_factor"] < 0.999
                                                  for r in at_or_above),
        "some_q_lt_10_already_linear": any(r["best_worst_factor"] < 0.999 for r in below),
        "n_q_lt_10_already_linear": sum(r["best_worst_factor"] < 0.999 for r in below),
        "median_factor_below_q10": float(np.median([r["best_worst_factor"] for r in below])),
        "median_factor_at_or_above_q10": float(np.median([r["best_worst_factor"]
                                                          for r in at_or_above])),
    }
    res["baselines_PADMM_IPDSADMM_IADMM"] = {
        "verdict": "inconclusive",
        "reason": ("Figure 4 compares against PADMM (Yashtini 2021), IPDS-ADMM (Yuan 2025) "
                   "and IADMM (Tang & Toh 2024). The paper releases no code, and reports "
                   "neither the problem instances/dimensions used in the three panels, nor "
                   "the baselines' hyperparameters (proximal and dual step sizes), nor the "
                   "initialisations/seeds. Re-implementing the baselines from their original "
                   "papers would benchmark OUR choices, not the authors'; no numbers are "
                   "produced here on purpose.")}

    res["elapsed_sec"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)

    print("toy problem q-sweep (best over 6 hyperparameter settings, worst over 6 seeds):")
    for r in table:
        print("   q=%5.1f  best worst-case contraction factor %.4f  (mu_x=%g mu_z=%g rho=%g)"
              "  iters to 1e-10: %d"
              % (r["q"], r["best_worst_factor"], r["best_hyper"]["mu_x"],
                 r["best_hyper"]["mu_z"], r["best_hyper"]["rho"],
                 r["best_worst_iters_to_1e-10"]))
    print(json.dumps(res["transition_analysis"], indent=1))
    print("baselines:", res["baselines_PADMM_IPDSADMM_IADMM"]["verdict"])
    print("wrote", OUT, "in %.1fs" % res["elapsed_sec"])


if __name__ == "__main__":
    main()
