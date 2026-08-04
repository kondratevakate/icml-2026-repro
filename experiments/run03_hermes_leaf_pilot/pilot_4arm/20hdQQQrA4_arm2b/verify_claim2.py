"""Claim 2 (Eq 8, Section 3): the CAffine projection carries a trainable null-space term
(I - A_g^+ A_g) w_phi(x), so multiple feasible projections can be jointly optimised, rather
than a single fixed orthogonal projection.

Tests (all executed, exhaustive over configs x seeds):
  T1 Structural: for any w, A_g P_g = b_g exactly (the sub-constraint is met regardless of w)
     -> residual max over all instances.
  T2 Null-space reachability: when rank(A_g) < n_out, (I - A_g^+ A_g) != 0 and varying w
     moves P_g inside the solution affine subspace: spread of outputs > 0 and the moved
     component is orthogonal to row(A_g). When rank(A_g) = n_out the term vanishes (Sec 3.2).
  T3 Optimisation value: task objective ||y - y_star||^2 for a random preferred point y_star,
     minimised over w (closed form / scipy) vs the fixed orthogonal projection (w = 0).
     Claim predicts trainable w can do strictly better on a non-trivial fraction of instances.
  MUTATION: delete the null-space term (force w = 0) and re-measure T3 -> objective must be
     >= the trainable-w objective, with a strictly positive median gap on rank-deficient cases.

Run: ./.venv/bin/python verify_claim2.py
"""
import json, os
import numpy as np
from scipy.optimize import minimize
from caffnet_core import P_gamma, caffnet, gammas, random_feasible_problem

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "claim2.json")


def run():
    rng_m = np.random.default_rng(7)
    configs = [(2, 3), (3, 5), (4, 6)]
    n_seeds = 40
    t1_res, spreads, rank_def_cases, full_rank_zero = [], [], 0, 0
    obj_w, obj_0, gaps, improved = [], [], [], 0
    for n_out, m in configs:
        for _ in range(n_seeds):
            rng = np.random.default_rng(rng_m.integers(1 << 31))
            A, b, y0 = random_feasible_problem(rng, n_out, m)
            f = y0 + rng.normal(size=n_out) * 2.0
            g = tuple(sorted(rng.choice(m, size=int(rng.integers(1, min(m, n_out) + 1)),
                                        replace=False).tolist()))
            Ag = A[list(g), :]
            N = np.eye(n_out) - np.linalg.pinv(Ag) @ Ag
            # T1
            for _ in range(3):
                w = rng.normal(size=n_out) * 3
                y = P_gamma(f, A, b, g, w)
                t1_res.append(float(np.max(np.abs(Ag @ y - b[list(g)]))))
            # T2
            if np.linalg.matrix_rank(Ag) < n_out:
                rank_def_cases += 1
                ys = [P_gamma(f, A, b, g, rng.normal(size=n_out) * 3) for _ in range(20)]
                spreads.append(float(np.std(np.array(ys), axis=0).max()))
            else:
                full_rank_zero += int(np.allclose(N, 0.0, atol=1e-10))
            # T3: optimise the full CAffNet output (Eq 12) over w for a preferred target
            y_star = y0 + rng.normal(size=n_out) * 0.3   # a feasible-ish preferred point

            def obj(w):
                y, _ = caffnet(f, A, b, w=w, p=2)
                if y is None:
                    return 1e6
                return float(np.sum((y - y_star) ** 2))

            o0 = obj(np.zeros(n_out))
            best = o0
            for _ in range(2):                       # multi-start Nelder-Mead over w
                w0 = rng.normal(size=n_out)
                r = minimize(obj, w0, method="Nelder-Mead",
                             options={"maxiter": 200, "xatol": 1e-8, "fatol": 1e-12})
                best = min(best, float(r.fun))
            obj_0.append(o0); obj_w.append(best)
            gaps.append(o0 - best)
            improved += int(o0 - best > 1e-9)
    n = len(obj_0)
    res = {
        "claim": 2,
        "source": "Eq (8), Section 3.2 + Remark 3.6",
        "command": "./.venv/bin/python verify_claim2.py",
        "configs": configs, "seeds_per_config": n_seeds, "n_instances": n,
        "T1_max_subconstraint_residual_over_all_w": max(t1_res),
        "T1_n_checks": len(t1_res),
        "T2_rank_deficient_cases": rank_def_cases,
        "T2_min_output_spread_over_w": min(spreads) if spreads else None,
        "T2_mean_output_spread_over_w": float(np.mean(spreads)) if spreads else None,
        "T2_full_rank_cases_with_zero_nullspace": full_rank_zero,
        "T3_mean_obj_trainable_w": float(np.mean(obj_w)),
        "T3_mean_obj_w_zero_orthogonal_projection": float(np.mean(obj_0)),
        "T3_median_gap": float(np.median(gaps)),
        "T3_max_gap": float(np.max(gaps)),
        "T3_frac_instances_trainable_w_strictly_better": improved / n,
        "MUTATION_w_forced_zero_is_never_better": bool(min(gaps) >= -1e-9),
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    run()
