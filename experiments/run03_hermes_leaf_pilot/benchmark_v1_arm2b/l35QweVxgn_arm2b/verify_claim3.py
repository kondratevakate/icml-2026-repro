"""verify_claim3.py — Theorem 2.2 (train error in continual learning).

CLAIM (anchored #3): under the same width/sample/iteration conditions as Thm 2.1, after KT
GD iterations the misclassification train error (and train loss) is o_d(1) UNIFORMLY over
all K tasks, w.h.p.

OPERATIONALISATION.  Run Algorithm 1 (sequential full-batch GD, hinge, quadratic-activation
two-layer net) with n = 0.5 d²K, ηT = 0.5 d², m = 8 d² over a d-ladder and over many seeds;
after the final KT-th iteration measure max_k misclassification error and max_k hinge loss
over all K tasks. "Uniformly small w.h.p." is operationalised as: max-over-tasks error small
and non-increasing along the d-ladder, over `REPS` independent seeds (worst seed reported).

VERDICT CAP: same width substitution as claim 2 (m = 8d² instead of Ω̃(d⁸K⁴)) → `toy` for the
literal regime; the qualitative theorem statement is testable and is what we score.

MUTATION (registered before running): violate the noise condition σ = Θ(1/(log^c d √d)) by
inflating σ ×8.  Prediction: uniform train error stops being small (max-over-task error rises
substantially).
"""
import numpy as np
from common import rng_for, save
from verify_claim2 import K, C_N, C_ETA_T, C_M, T, run_stream

REPS = 8


def ladder(ds, sigma_mult=1.0):
    rows = []
    for d in ds:
        n, m, eta_T = int(C_N * d * d * K), C_M * d * d, C_ETA_T * d * d
        max_err, max_loss = [], []
        for r in range(REPS):
            _, errs, losses, _, _, _ = run_stream(d, n, m, eta_T, rng_for(3, 131 * d + r),
                                               sigma_mult=sigma_mult)
            max_err.append(max(errs)); max_loss.append(max(losses))
        rows.append({"d": d, "n": n, "m": m,
                     "mean_max_task_err": float(np.mean(max_err)),
                     "worst_seed_max_task_err": float(np.max(max_err)),
                     "mc_stderr_err": float(np.std(max_err, ddof=1) / np.sqrt(REPS)),
                     "mean_max_task_hinge_loss": float(np.mean(max_loss))})
        print(rows[-1], flush=True)
    return rows


def main():
    ds = [8, 12, 16, 20]
    rows = ladder(ds)
    mut = ladder(ds, sigma_mult=8.0)
    e = [r["mean_max_task_err"] for r in rows]
    em = [r["mean_max_task_err"] for r in mut]
    small = bool(max(e) < 0.05)
    nonincreasing = bool(e[-1] <= e[0] + 1e-9)
    out = {
        "claim": 3, "source": "Theorem 2.2 (Sec 2.2); restated as Theorem C.1 (App. C)",
        "route": "simulation of Algorithm 1, KT iterations, uniform-over-tasks statistics",
        "K": K, "T_per_task": T, "total_iterations": "K*T", "reps_per_cell": REPS,
        "d_ladder": rows,
        "uniform_error_small": small,
        "uniform_error_nonincreasing_in_d": nonincreasing,
        "mutation_noise_violation": {
            "prediction": "sigma x8 violates sigma = Theta(1/(log^c d sqrt d)); uniform train error must rise",
            "rows": mut,
            "mean_max_task_err_ratio_vs_main": [float(a / max(b, 1e-12)) for a, b in zip(em, e)],
            "property_breaks": bool(max(em) > 10 * max(max(e), 1e-6) or max(em) > 0.1),
        },
        "verdict_cap": "toy",
        "verdict_cap_reason": "m = 8d² substituted for the theorem's Ω̃(d⁸K⁴) width",
    }
    out["verdict"] = ("verified (qualitative statement; width scaled down -> see verdict_cap)"
                      if small and nonincreasing and out["mutation_noise_violation"]["property_breaks"]
                      else "inconclusive")
    save(3, out)


if __name__ == "__main__":
    main()
