"""verify_claim2.py — the CAffine projection's trainable null-space component.

Claim (Section 3.2, Eq. (8)):
  P_gamma(x) = f_theta(x) - Ag^dag(x)(Ag(x) f_theta(x) - bg(x)) + (I - Ag^dag(x) Ag(x)) w_phi(x)
  "allowing joint optimization across multiple feasible projections rather than a
   fixed orthogonal projection".

Tested properties (exhaustive over a grid of shapes/ranks and 40 seeds each):
  T1  Feasibility of the sub-constraint is preserved for ANY w_phi:
        ||A_g P_g - b_g||_inf ~ 0 whenever A_g y = b_g is consistent.
  T2  Reachability: the map w_phi -> P_g is ONTO the affine solution set
        {y : A_g y = b_g}.  Concretely, for any y* in that set, w_phi = y* yields
        P_g = y* (max error over instances reported).
  T3  w_phi = 0 reproduces exactly the orthogonal / minimum-norm-correction
        projection onto {y : A_g y = b_g} (agrees with scipy lstsq solution).
  T4  Joint optimisation: on a task whose optimum lies on a face but NOT at the
        orthogonal projection point, optimising w_phi attains a strictly lower loss
        than the fixed orthogonal projection (w_phi = 0).
        Loss: L(w) = ||P*(f_theta, w) - y_target||_2^2, y_target a feasible point.

MUTATION (T4-mut): remove the null-space term ((I - Ag^dag Ag) w := 0). Then the
reachable set collapses to a single point and the attainable loss must stay at the
orthogonal-projection value, i.e. strictly worse. If removing the term did NOT hurt,
the null-space component would be doing nothing and the claim would be unsupported.

Run: .venv/bin/python verify_claim2.py
"""
import itertools
import json
import time

import numpy as np
from scipy.optimize import minimize

from caffnet_core import P_gamma, caffnet_output, random_feasible_system

SEEDS = list(range(40))
SHAPES = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 2), (3, 3)]  # (k rows, n_out)


def main():
    t0 = time.time()
    res = {"claim": 2,
           "claim_text": "Eq. (8) includes a trainable null-space component "
                         "(I - Ag^dag Ag) w_phi(x), enabling joint optimisation over "
                         "feasible projections instead of a fixed orthogonal projection",
           "source": "Section 3.2, Eq. (8); text after Theorem 3.4; Remark 3.6",
           "command": ".venv/bin/python verify_claim2.py"}

    t1_max, t2_max, t3_max, n_inst = 0.0, 0.0, 0.0, 0
    nullspace_dims = []
    for (k, n), seed in itertools.product(SHAPES, SEEDS):
        rng = np.random.default_rng([12345, k, n, seed])
        Ag = rng.normal(size=(k, n))
        if seed % 3 == 1 and k >= 2:      # force a linearly dependent row
            Ag[1] = 2.5 * Ag[0]
        y_sol = rng.normal(size=n)
        bg = Ag @ y_sol                    # consistent by construction
        f_th = rng.normal(size=n)
        Agd = np.linalg.pinv(Ag)
        N = np.eye(n) - Agd @ Ag
        nullspace_dims.append(int(round(np.trace(N))))

        for _ in range(5):                 # arbitrary w_phi
            w = rng.normal(size=n) * 10.0
            Pg = P_gamma(f_th, w, Ag, bg, tuple(range(k)))
            t1_max = max(t1_max, float(np.max(np.abs(Ag @ Pg - bg))))
        # T2: reachability. P_g(w) = N f_theta + Ag^dag bg + N w, so any point
        # y* of the affine solution set {y : Ag y = bg} is reached with
        # w_phi = y* - f_theta. Verified for a random y* in that set.
        z = rng.normal(size=n)
        y_star = y_sol + N @ z
        Pg = P_gamma(f_th, y_star - f_th, Ag, bg, tuple(range(k)))
        t2_max = max(t2_max, float(np.max(np.abs(Pg - y_star))))
        # T3: w=0 equals min-norm-correction (orthogonal) projection
        Pg0 = P_gamma(f_th, np.zeros(n), Ag, bg, tuple(range(k)))
        delta = np.linalg.lstsq(Ag, bg - Ag @ f_th, rcond=None)[0]
        t3_max = max(t3_max, float(np.max(np.abs(Pg0 - (f_th + delta)))))
        n_inst += 1

    res["T1_max_abs_residual_Ag_Pg_minus_bg"] = t1_max
    res["T2_max_abs_reachability_error"] = t2_max
    res["T3_max_abs_dev_from_orthogonal_projection_at_w0"] = t3_max
    res["n_instances"] = n_inst
    res["mean_nullspace_dim"] = float(np.mean(nullspace_dims))

    # ---- T4: joint optimisation beats the fixed orthogonal projection ----
    t4 = []
    for seed in SEEDS:
        rng = np.random.default_rng([777, seed])
        n, m = 3, 5
        A, b, y0 = random_feasible_system(rng, m, n, dep_rows=1)
        f_th = y0 + rng.normal(size=n) * 3.0          # typically infeasible
        if np.all(A @ f_th <= b):
            continue
        # target: a random feasible point (interior-ish), the "task objective"
        y_tgt = y0 + 0.1 * rng.normal(size=n)
        if not np.all(A @ y_tgt <= b):
            y_tgt = y0

        def loss(w, use_ns=True):
            y, _ = caffnet_output(f_th, np.asarray(w), A, b, p=2.0,
                                  use_nullspace=use_ns)
            if y is None:
                return 1e6
            return float(np.sum((y - y_tgt) ** 2))

        l0 = loss(np.zeros(n))                        # fixed orthogonal projection
        best = l0
        for r in range(4):                            # multi-start (deterministic)
            w0 = np.zeros(n) if r == 0 else rng.normal(size=n) * 2.0
            out = minimize(loss, w0, method="Nelder-Mead",
                           options={"maxiter": 800, "xatol": 1e-8, "fatol": 1e-12})
            best = min(best, float(out.fun))
        lm = loss(np.zeros(n), use_ns=False)          # MUTATION: no null-space term
        t4.append(dict(seed=seed, loss_w0_orthogonal=l0, loss_optimised_w=best,
                       loss_mutation_no_nullspace=lm,
                       improved=bool(best < l0 - 1e-9)))
    res["T4_records"] = t4
    res["T4_n"] = len(t4)
    res["T4_n_improved"] = sum(r["improved"] for r in t4)
    res["T4_frac_improved"] = res["T4_n_improved"] / max(len(t4), 1)
    res["T4_median_rel_loss_reduction"] = float(np.median(
        [(r["loss_w0_orthogonal"] - r["loss_optimised_w"]) /
         max(r["loss_w0_orthogonal"], 1e-12) for r in t4]))
    res["MUT_no_nullspace_equals_w0"] = bool(all(
        abs(r["loss_mutation_no_nullspace"] - r["loss_w0_orthogonal"]) < 1e-9
        for r in t4))
    res["MUT_n_worse_than_optimised"] = sum(
        r["loss_mutation_no_nullspace"] > r["loss_optimised_w"] + 1e-9 for r in t4)
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open("results/claim2.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "T4_records"}, indent=2))


if __name__ == "__main__":
    main()
