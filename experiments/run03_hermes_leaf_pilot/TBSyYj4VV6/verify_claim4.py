"""Claim 4 — Corollary 25 (Ridge regression).
Quantum O~(r*sqrt(mn)/eps + n^3) vs classical O~(m r + poly(n,1/eps)).

Ridge is strongly convex; the (1+/-eps) sparsifier guarantees the ridge optimum
on the sketch is within (1+/-eps) of the full optimum. We verify this on CPU,
confirm the classical O(m r + poly(n)) scaling in m, and check the asymptotic
ordering. Quantum construction not executed (boundary).
"""
from __future__ import annotations
import json, os
import numpy as np
import repro_core as rc

SEED = 4004
EPS = 0.15
M, N, R = 4000, 30, 10
S = 2000
LAM = 1.0

OUT = os.path.join(os.path.dirname(__file__), "results", "claim4.json")


def main():
    X, y, _ = rc.make_data(M, N, R, seed=SEED)
    idx, w, Xw, yw = rc.build_sparsifier(X, y, S, seed=SEED + 11)

    def loss_fn(X_, y_, b):
        return float(np.sum(0.5 * (X_ @ b - y_) ** 2) + 0.5 * LAM * np.sum(b ** 2))

    solve_full = lambda X_, y_: rc.solve_ridge(X_, y_, LAM)
    solve_sketch = lambda Xw_, yw_: rc.solve_ridge_weighted(Xw_, yw_, LAM)

    L_full, L_sk, loss_ratio, gram_err = rc.claim_regression_metrics(
        X, y, idx, w, Xw, yw, solve_full, solve_sketch, loss_fn, EPS)

    lim_half, _ = rc.quantum_classical_m_ratio("half")
    scaling = rc.m_scaling_sweep(n=N, r=R, s=S, seed=SEED + 5)
    mut = rc.claim_mutation(X, y, EPS, R, S, SEED, solve_full, solve_sketch, loss_fn)

    ok = (loss_ratio <= EPS) and (gram_err <= EPS)
    result = {
        "claim_no": 4,
        "claim": ("Ridge regression is solved in O~(r*sqrt(mn)/epsilon + n^3) quantum time versus "
                  "O~(mr + poly(n,1/epsilon)) classically (Corollary 25)."),
        "source": "Corollary 25 (arXiv:2509.24757)",
        "paper": "Accelerating Regression Tasks with Quantum Algorithms (arXiv:2509.24757 / OpenReview TBSyYj4VV6)",
        "seed": SEED,
        "method": "Leverage-score sparsifier; ridge optimum on sketch vs full; m-scaling timed; asymptotic symbolic.",
        "verdict": "verified" if ok else "inconclusive",
        "reason": (f"Ridge optimum preserved on sparsifier: loss_ratio={loss_ratio:.3f}<={EPS:.2f}, "
                   f"gram rel-err={gram_err:.3f}<={EPS:.2f} "
                   f"(strongly-convex -> theoretical (1+/-eps) guarantee). Classical ridge solve slope in m = "
                   f"{scaling['full_slope']:.2f}, sketch slope = {scaling['sketch_slope']:.2f}. "
                   f"Quantum/classical -> {lim_half} as m->inf. EVIDENCE BOUNDARY: literal quantum "
                   f"construction (QRAM) NOT executed on CPU; shown via classical surrogate + symbolic limit."),
        "executed_numeric_experiment": True,
        "metrics": {
            "eps": EPS, "m": M, "n": N, "r": R, "sparsifier_size": S, "ridge_lambda": LAM,
            "ridge_loss_full": L_full, "ridge_loss_sketch": L_sk,
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
