"""verify_claim1.py — Theorem 3.5 approximation bound ||P* - f_t||_p < (3 + 3*sqrt(n_out))*K.

Exhaustive over (n_out, m, p, rank/redundancy regime) x many seeds. For each instance we
construct exactly the hypotheses of the proof (App. C):
  f_t feasible, ||f_theta - f_t||_p < K, ||w_phi||_p < 2K (Eq. 28),
then measure the realised error of the CAffNet output and the two intermediate bounds
(Eq. 31 matrix-norm bounds; the (1+3 sqrt n) bound on ||f_t - P_gamma||).

MUTATION: break Eq. 28 by letting ||w_phi||_p = 50K. The bound is then predicted to fail.
"""
import json, sys, time
import numpy as np
from caffnet_core import (P_gamma, caffnet, gamma_set, random_feasible_instance,
                          sample_in_ball, seed_of)

K = 1e-2
OUT = "results/claim1.json"


def run(w_scale, n_seeds=40):
    rows, worst = [], 0.0
    n_viol = n_tot = 0
    mat_norm_viol = 0
    sub_bound_viol = 0
    for n_out in (1, 2, 3, 4, 5):
        for m in (1, 2, 3, 5, 7):
            for p in (1, 2, 3):
                for redundant in (False, True):
                    for seed in range(n_seeds):
                        rng = np.random.default_rng(
                            seed_of(n_out, m, p, redundant, seed))
                        rank = max(1, min(m, n_out) - (1 if redundant else 0))
                        A, b, y0 = random_feasible_instance(
                            rng, n_out, m, rank=rank, redundant=redundant)
                        f_t = y0                                   # feasible target
                        f_th = f_t + sample_in_ball(rng, n_out, K, p)
                        w = sample_in_ball(rng, n_out, w_scale * K, p)
                        Ps = caffnet(f_th, A, b, w, p=p)
                        err = np.linalg.norm(Ps - f_t, p)
                        bound = (3 + 3 * np.sqrt(n_out)) * K
                        n_tot += 1
                        if err >= bound:
                            n_viol += 1
                        worst = max(worst, err / bound)
                        # Eq. 31 matrix norm bounds + (1+3 sqrt n) sub-bound
                        for gam in gamma_set(m, n_out):
                            Ag = A[list(gam), :]
                            Ap = np.linalg.pinv(Ag)
                            M1, M2 = Ap @ Ag, np.eye(n_out) - Ap @ Ag
                            for M in (M1, M2):
                                nrm = np.max([np.linalg.norm(M @ v, p) /
                                              np.linalg.norm(v, p)
                                              for v in np.eye(n_out)] +
                                             [np.linalg.norm(M @ v, p) /
                                              np.linalg.norm(v, p)
                                              for v in rng.normal(size=(20, n_out))])
                                if nrm > np.sqrt(n_out) + 1e-9:
                                    mat_norm_viol += 1
                            Pg = P_gamma(f_th, A, b, gam, w)
                            if np.all(A @ Pg - b <= 1e-9):
                                if np.linalg.norm(f_t - Pg, p) >= (1 + 3 * np.sqrt(n_out)) * K:
                                    sub_bound_viol += 1
    return dict(n_instances=n_tot, n_bound_violations=n_viol,
                worst_ratio_err_over_bound=float(worst),
                eq31_matrix_norm_violations=int(mat_norm_viol),
                sub_bound_violations_1p3sqrtn=int(sub_bound_viol))


if __name__ == "__main__":
    t0 = time.time()
    main = run(w_scale=2.0)                 # hypotheses of the theorem hold
    mut = run(w_scale=50.0, n_seeds=20)     # MUTATION: Eq. 28 broken
    res = dict(claim=1, K=K,
               bound_formula="(3+3*sqrt(n_out))*K",
               main=main, mutation_w_scale_50K=mut,
               command="python verify_claim1.py",
               numpy=np.__version__, seconds=round(time.time() - t0, 2))
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
