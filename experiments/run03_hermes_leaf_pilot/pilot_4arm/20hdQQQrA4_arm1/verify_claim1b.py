"""verify_claim1b.py — targeted addendum to claim 1.

Two fixes to verify_claim1.py:
 (i) the (1+3*sqrt(n_out))K sub-bound of App. C is only asserted for the gamma whose
     hyperplane the SEGMENT [f_theta, f_t] crosses (the intersection point f_gamma), not for
     every feasible gamma. Here we locate that gamma explicitly and test the sub-bound there.
 (ii) a mutation that actually targets the mechanism of Eq. (12): replace argmin by argmax
     (pick the FARTHEST feasible candidate). The final bound is then predicted to break.
"""
import json, time
import numpy as np
from caffnet_core import P_gamma, caffnet, gamma_set, sample_in_ball, seed_of

K = 1e-2
OUT = "results/claim1b.json"


def crossing_gamma(f_th, f_t, A, b):
    """First hyperplane crossed going from f_t (feasible) to f_theta (infeasible)."""
    best_d, best = np.inf, None
    for j in range(A.shape[0]):
        num = b[j] - A[j] @ f_t
        den = A[j] @ (f_th - f_t)
        if abs(den) < 1e-12:
            continue
        d = num / den
        if 0 <= d <= 1 and d < best_d:
            best_d, best = d, j
    return best, best_d


def farthest_select(f_th, A, b, w, p):
    cands = [P_gamma(f_th, A, b, g, w) for g in gamma_set(A.shape[0], A.shape[1])]
    feas = [y for y in cands if np.all(A @ y - b <= 1e-9)]
    if not feas:
        return f_th
    return max(feas, key=lambda y: np.linalg.norm(y - f_th, p))


def boundary_instance(rng, n_out, m, rank):
    """Like random_feasible_instance but with f_t ON the boundary (zero slack on a random
    subset of rows), so that a K-perturbation of f_t is genuinely infeasible (Case 2)."""
    U = rng.normal(size=(m, rank)); V = rng.normal(size=(rank, n_out))
    A = U @ V
    nrm = np.linalg.norm(A, axis=1, keepdims=True); nrm[nrm < 1e-12] = 1.0
    A = A / nrm
    y0 = rng.normal(size=n_out)
    slack = rng.uniform(0.0, 0.5, size=m)
    active = rng.random(m) < 0.5
    active[rng.integers(m)] = True
    slack[active] = 0.0                      # y0 lies on these hyperplanes
    return A, A @ y0 + slack, y0


def run(n_seeds=200):
    case2 = 0
    main_tot = main_viol = 0
    main_worst = 0.0
    sub_tested = sub_viol = 0
    mut_tot = mut_viol = 0
    mut_worst = 0.0
    for n_out in (1, 2, 3, 4, 5):
        for m in (2, 3, 5, 7):
            for p in (1, 2, 3):
                for seed in range(n_seeds):
                    rng = np.random.default_rng(seed_of(n_out, m, p, seed, 7))
                    A, b, y0 = boundary_instance(
                        rng, n_out, m, rank=max(1, min(m, n_out) - 1))
                    f_t = y0
                    f_th = f_t + sample_in_ball(rng, n_out, K, p)
                    w = sample_in_ball(rng, n_out, 2 * K, p)
                    if np.all(A @ f_th - b <= 0):
                        continue                      # Case 1 of the proof
                    case2 += 1
                    bound = (3 + 3 * np.sqrt(n_out)) * K
                    y_star = caffnet(f_th, A, b, w, p=p)
                    e = np.linalg.norm(y_star - f_t, p)
                    main_tot += 1
                    main_worst = max(main_worst, e / bound)
                    if e >= bound:
                        main_viol += 1
                    j, _ = crossing_gamma(f_th, f_t, A, b)
                    if j is not None:
                        Pg = P_gamma(f_th, A, b, (j,), w)
                        sub_tested += 1
                        if np.linalg.norm(f_t - Pg, p) >= (1 + 3 * np.sqrt(n_out)) * K:
                            sub_viol += 1
                    y = farthest_select(f_th, A, b, w, p)   # MUTATION of Eq. (12)
                    err = np.linalg.norm(y - f_t, p)
                    mut_tot += 1
                    mut_worst = max(mut_worst, err / bound)
                    if err >= bound:
                        mut_viol += 1
    return dict(case2_instances=case2,
                main_instances=main_tot, main_bound_violations=main_viol,
                main_worst_ratio=float(main_worst),
                subbound_intersection_gamma_tested=sub_tested,
                subbound_intersection_gamma_violations=sub_viol,
                mutation_argmax_instances=mut_tot,
                mutation_argmax_bound_violations=mut_viol,
                mutation_argmax_worst_ratio=float(mut_worst))


if __name__ == "__main__":
    t0 = time.time()
    res = dict(claim="1b", K=K, **run(), command="python verify_claim1b.py",
               seconds=round(time.time() - t0, 2))
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
