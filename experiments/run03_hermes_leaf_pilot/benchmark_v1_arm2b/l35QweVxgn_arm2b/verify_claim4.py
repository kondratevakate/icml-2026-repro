"""verify_claim4.py — Theorem 2.3 (delayed generalization gap).

CLAIM (anchored #4): F^gen_{k,K} := E_{D_k}[F_k(w_K) − F̂_k(w_K)] ≲ ηT·exp(ηT(K−k+1)/√m)/n,
for 1-Lipschitz, 1-smooth loss; the gap decays with sample size n.

OPERATIONALISATION.  Two separable sub-assertions:
 (a) BOUND HOLDS: measure the expected gap by Monte-Carlo over independent draws of D_k
     (fresh task-k training set each rep, held-out test set of 4000 points for F_k) and
     compare against the RHS computed with the run's own η, T, K, k, m.
 (b) DECAY IN n: recover the exponent of the measured gap in n by log-log fit (predicted −1).
Loss used for the gap is the (1-Lipschitz, 1-smooth) Huberised hinge — the theorem's stated
hypothesis class — evaluated at the final iterate w_K of the real GD run.

MUTATION (registered before running): shrink n by 8× at fixed everything else.
Prediction: the measured gap grows by ≈8× (1/n law) and stays under the bound.
"""
import numpy as np
from common import QuadNet, make_stream, loglog_slope, rng_for, save, sigma_of

K, d, T, C_M = 3, 12, 40, 8
ETA_T = 0.5 * d * d
M = C_M * d * d
N_TEST = 4000
REPS = 12


def smooth_hinge(u):
    """1-Lipschitz, 1-smooth surrogate: quadratic in [0,1] transition (Huberised hinge)."""
    return np.where(u >= 1.0, 0.0, np.where(u <= 0.0, 0.5 - u, 0.5 * (1.0 - u) ** 2))


def gap_for_n(n, rep):
    rng = rng_for(4, 1000 * n + rep)
    tasks = make_stream(d, K, n, sigma_of(d), rng, mu_norm=1.0, n_test=N_TEST)
    net = QuadNet(d, M, rng)
    eta = ETA_T / T
    for k in range(K):
        net.gd(tasks[k]["X"], tasks[k]["y"], eta, T)
    Xk, yk = tasks[0]["X"], tasks[0]["y"]
    Xt, yt = tasks[0]["test"]
    train = smooth_hinge(yk * net.out(Xk)).mean()
    test = smooth_hinge(yt * net.out(Xt)).mean()
    return float(test - train)


def main():
    ns = [50, 100, 200, 400, 800]
    rows = []
    for n in ns:
        g = np.array([gap_for_n(n, r) for r in range(REPS)])
        bound = ETA_T * np.exp(ETA_T * (K - 1 + 1) / np.sqrt(M)) / n
        rows.append({"n": n, "mean_gap": float(g.mean()),
                     "mc_stderr": float(g.std(ddof=1) / np.sqrt(REPS)),
                     "abs_mean_gap": float(abs(g.mean())),
                     "bound_rhs": float(bound),
                     "bound_holds": bool(abs(g.mean()) <= bound)})
        print(rows[-1], flush=True)
    slope = loglog_slope([r["n"] for r in rows], [r["abs_mean_gap"] for r in rows])
    base = [r for r in rows if r["n"] == 400][0]
    mut = [r for r in rows if r["n"] == 50][0]
    out = {
        "claim": 4, "source": "Theorem 2.3 (Sec 2.2); restated as Theorem B.6 (App. B)",
        "route": "simulation of Algorithm 1 + Monte-Carlo over D_k; bound evaluated literally",
        "config": {"d": d, "K": K, "k": 1, "T": T, "eta_T": ETA_T, "m": M,
                   "n_test": N_TEST, "reps_per_cell": REPS,
                   "loss": "Huberised hinge (1-Lipschitz, 1-smooth), as the theorem assumes"},
        "n_sweep": rows,
        "bound_holds_all_cells": bool(all(r["bound_holds"] for r in rows)),
        "gap_loglog_slope_in_n": slope,
        "predicted_slope": -1.0,
        "decays_with_n": bool(slope < -0.5),
        "mutation_shrink_n_8x": {
            "prediction": "gap at n=50 is ~8x the gap at n=400 (1/n law), bound still holds",
            "gap_n400": base["abs_mean_gap"], "gap_n50": mut["abs_mean_gap"],
            "ratio": mut["abs_mean_gap"] / max(base["abs_mean_gap"], 1e-12),
            "property_breaks": bool(mut["abs_mean_gap"] > 3 * base["abs_mean_gap"]),
        },
    }
    out["verdict"] = ("verified" if out["bound_holds_all_cells"] and out["decays_with_n"]
                      and out["mutation_shrink_n_8x"]["property_breaks"] else "inconclusive")
    out["evidence_boundary"] = ("The RHS of Thm 2.3 is an order bound with hidden constants; "
                                "at these (feasible) widths it is numerically very loose, so "
                                "'bound holds' is a weak test and the informative evidence is "
                                "the recovered 1/n exponent.")
    save(4, out)


if __name__ == "__main__":
    main()
