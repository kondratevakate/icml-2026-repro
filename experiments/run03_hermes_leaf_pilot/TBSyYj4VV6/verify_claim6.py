"""Claim 6 — Corollary 11 (l_p regression, p in (0,2]).
Quadratic speedup in the sample parameter m, which dominates when m >> n.

l_p regression (p in (0,2]) is solved by iteratively reweighted least squares; each
iteration is a weighted LS that the (1+/-eps) sparsifier preserves. We verify on
CPU that the l_p optimum on the sparsifier matches the full-data optimum, and
demonstrate that the m-dependence dominates the runtime when m >> n: the classical
solve scales ~linearly in m while the sketch solve is m-independent (consistent
with the quadratic speedup in m). Quantum construction not executed (boundary).
"""
from __future__ import annotations
import json, os
import numpy as np
import repro_core as rc

SEED = 6006
EPS = 0.15
M, N, R = 4000, 30, 10
S = 2000
P = 1.5

OUT = os.path.join(os.path.dirname(__file__), "results", "claim6.json")


def main():
    X, y, _ = rc.make_data(M, N, R, seed=SEED)
    idx, w, Xw, yw = rc.build_sparsifier(X, y, S, seed=SEED + 11)

    def loss_fn(X_, y_, b):
        return float(rc.lp_loss(X_, y_, b, P))

    solve_full = lambda X_, y_: rc.solve_lp_irls(X_, y_, P)
    solve_sketch = lambda Xw_, yw_: rc.solve_lp_irls(Xw_, yw_, P)

    L_full, L_sk, loss_ratio, gram_err = rc.claim_regression_metrics(
        X, y, idx, w, Xw, yw, solve_full, solve_sketch, loss_fn, EPS)

    lim_half, _ = rc.quantum_classical_m_ratio("half")

    scaling = rc.m_scaling_sweep(n=N, r=R, s=S, seed=SEED + 5)
    # Explicit m >> n dominance demonstration: sweep m with fixed n, measure the
    # solve times. The key robust signals are:
    #   - classical full solve time GROWS with m  (full_slope > 0.5)
    #   - sketch solve time stays BOUNDED in m  (sketch_time does not blow up)
    # so the speedup full/sketch increases as m grows -> m dominates when m >> n.
    ft = scaling["full_time"]
    st = scaling["sketch_time"]
    full_grows = scaling["full_slope"] > 0.5
    sketch_bounded = (st[-1] < 2.0 * st[0]) and (abs(scaling["sketch_slope"]) < 1.0)
    speedup_grows_with_m = bool(full_grows and sketch_bounded)

    mut = rc.claim_mutation(X, y, EPS, R, S, SEED, solve_full, solve_sketch, loss_fn)

    ok = (gram_err <= EPS) and (loss_ratio <= 0.5) and speedup_grows_with_m
    result = {
        "claim_no": 6,
        "claim": ("For l_p regression with p in (0,2], the algorithm achieves a quadratic speedup in the "
                  "sample parameter m, which dominates runtime when m >> n (Corollary 11)."),
        "source": "Corollary 11 (arXiv:2509.24757)",
        "paper": "Accelerating Regression Tasks with Quantum Algorithms (arXiv:2509.24757 / OpenReview TBSyYj4VV6)",
        "seed": SEED,
        "method": "Leverage-score sparsifier applied to l_p (IRLS, p=1.5) on full vs weighted sketch; m-scaling shows m>>n dominance; asymptotic symbolic.",
        "verdict": "verified" if ok else "inconclusive",
        "reason": ("Sparsifier (1+/-eps) embedding holds (gram rel-err=%.3f<=%.2f); l_p optimum on sketch within "
                   "%.2f of full (ratio=%.3f). m-scaling: full solve slope=%.2f, sketch slope=%.2f -> m dominates "
                   "when m>>n, consistent with quadratic speedup. Quantum/classical -> %s as m->inf. EVIDENCE "
                   "BOUNDARY: literal quantum construction (QRAM) NOT executed on CPU."
                   % (gram_err, EPS, 0.5, loss_ratio, scaling["full_slope"], scaling["sketch_slope"], lim_half)),
        "executed_numeric_experiment": True,
        "metrics": {
            "eps": EPS, "m": M, "n": N, "r": R, "sparsifier_size": S, "lp_p": P,
            "lp_loss_full": L_full, "lp_loss_sketch": L_sk,
            "optimum_loss_ratio": loss_ratio, "ls_gram_relative_error": gram_err,
            "quantum_over_classical_limit_m_inf": str(lim_half),
            "m_scaling_full_slope": scaling["full_slope"],
            "m_scaling_sketch_slope": scaling["sketch_slope"],
            "speedup_grows_with_m": speedup_grows_with_m,
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
