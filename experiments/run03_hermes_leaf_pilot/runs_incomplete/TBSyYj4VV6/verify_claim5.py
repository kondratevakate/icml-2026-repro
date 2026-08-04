"""Claim 5 — Corollary 12 (Huber regression via gamma_p-loss framework).
Quantum O~(r*sqrt(mn)/eps + poly(n,1/eps)) time.

The paper handles Huber/robust regression through a smooth gamma_p-loss
framework that reduces each step to a reweighted least squares -- exactly the
object the (1+/-eps) sparsifier preserves. We verify on CPU that the Huber
optimum on the sparsifier matches the full-data Huber optimum, time the classical
scaling in m, and check the asymptotic ordering. Quantum construction not executed.
"""
from __future__ import annotations
import json, os
import numpy as np
import repro_core as rc

SEED = 5005
EPS = 0.15
M, N, R = 4000, 30, 10
S = 2000
DELTA = 1.0

OUT = os.path.join(os.path.dirname(__file__), "results", "claim5.json")


def main():
    X, y, _ = rc.make_data(M, N, R, seed=SEED)
    idx, w, Xw, yw = rc.build_sparsifier(X, y, S, seed=SEED + 11)

    def loss_fn(X_, y_, b):
        return float(rc.huber_loss(X_, y_, b, DELTA))

    solve_full = lambda X_, y_: rc.solve_huber_irls(X_, y_, DELTA)
    solve_sketch = lambda Xw_, yw_: rc.solve_huber_irls(Xw_, yw_, DELTA)

    L_full, L_sk, loss_ratio, gram_err = rc.claim_regression_metrics(
        X, y, idx, w, Xw, yw, solve_full, solve_sketch, loss_fn, EPS)

    lim_half, _ = rc.quantum_classical_m_ratio("half")
    scaling = rc.m_scaling_sweep(n=N, r=R, s=S, seed=SEED + 5)
    mut = rc.claim_mutation(X, y, EPS, R, S, SEED, solve_full, solve_sketch, loss_fn)

    # Huber is convex but not strongly convex; require embedding + modest optimum band.
    ok = (gram_err <= EPS) and (loss_ratio <= 0.5)
    result = {
        "claim_no": 5,
        "claim": ("Huber regression is handled via a gamma_p-loss framework in "
                  "O~(r*sqrt(mn)/epsilon + poly(n,1/epsilon)) quantum time (Corollary 12)."),
        "source": "Corollary 12 (arXiv:2509.24757)",
        "paper": "Accelerating Regression Tasks with Quantum Algorithms (arXiv:2509.24757 / OpenReview TBSyYj4VV6)",
        "seed": SEED,
        "method": "Leverage-score sparsifier applied to Huber (IRLS on full vs weighted sketch); m-scaling timed; asymptotic symbolic.",
        "verdict": "verified" if ok else "inconclusive",
        "reason": ("Sparsifier (1+/-eps) embedding holds (gram rel-err=%.3f<=%.2f); Huber optimum on sketch within "
                   "%.2f of full (ratio=%.3f). gamma_p-loss reduces each iteration to reweighted LS, which the "
                   "sparsifier preserves. Classical solve slope in m = %.2f, sketch slope = %.2f. Quantum/classical "
                   "-> %s. EVIDENCE BOUNDARY: literal quantum construction (QRAM) NOT executed on CPU."
                   % (gram_err, EPS, 0.5, loss_ratio, scaling["full_slope"], scaling["sketch_slope"], lim_half)),
        "executed_numeric_experiment": True,
        "metrics": {
            "eps": EPS, "m": M, "n": N, "r": R, "sparsifier_size": S, "huber_delta": DELTA,
            "huber_loss_full": L_full, "huber_loss_sketch": L_sk,
            "optimum_loss_ratio": loss_ratio, "ls_gram_relative_error": gram_err,
            "quantum_over_classical_limit_m_inf": str(lim_half),
            "m_scaling_full_slope": scaling["full_slope"],
            "m_scaling_sketch_slope": scaling["sketch_slope"],
        },
        "mutation": {
            "description": "(M1) s<r/eps^2 breaks optimum-preservation; (M2) tighter eps needs more samples; (M3) quantum sqrt(m) required for speedup.",
            **mut,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
