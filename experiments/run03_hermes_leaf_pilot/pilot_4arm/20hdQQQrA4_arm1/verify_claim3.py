"""verify_claim3.py — no full-row-rank requirement; arbitrary cardinality via
sub-constraints of size at most min(m, n_out).

 A. Feasibility (Thm 3.4 / Lemma 3.3) on instances where A(x) is deliberately
    rank-deficient, redundant, linearly dependent, and m >> n_out.
 B. HardNet-style baseline (single pseudo-inverse correction on the violated rows,
    which requires full row rank) on the SAME instances -> violations appear.
    This is the mutation: replace the decomposition mechanism with the naive one.
 C. Cardinality: verify the certificate always lies at k <= min(m, n_out) and that
    truncating Gamma (k = 1 only, or k = min(m,n_out) only) can LOSE feasibility.
 D. The 2-D example of Fig. 1 / Sec. 1: [0,1][x,y]^T <= 0 (rank-deficient, non-unique).
"""
import json, time
import numpy as np
from caffnet_core import caffnet, gamma_set, hardnet_like, random_feasible_instance, seed_of

OUT = "results/claim3.json"


def sweep():
    tot = 0
    caff_viol = 0
    caff_max = 0.0
    hard_viol = 0
    hard_max = 0.0
    k1_viol = 0
    kmax_only_viol = 0
    max_k_used = 0
    for n_out in (1, 2, 3, 4):
        for m in (1, 2, 3, 5, 8, 12):
            for regime in ("rank_deficient", "redundant", "full"):
                for seed in range(30):
                    rng = np.random.default_rng(seed_of(n_out, m, regime, seed))
                    if regime == "rank_deficient":
                        rank = max(1, min(m, n_out) - 1)
                        red = False
                    elif regime == "redundant":
                        rank = min(m, n_out); red = True
                    else:
                        rank = min(m, n_out); red = False
                    A, b, y0 = random_feasible_instance(rng, n_out, m, rank=rank, redundant=red)
                    f = y0 + rng.normal(size=n_out) * 3.0     # usually infeasible
                    w = rng.normal(size=n_out)
                    tot += 1
                    y, gam = caffnet(f, A, b, w, p=2, return_gamma=True)
                    v = float(np.max(A @ y - b))
                    caff_max = max(caff_max, v)
                    if v > 1e-9:
                        caff_viol += 1
                    if gam is not None:
                        max_k_used = max(max_k_used, len(gam))
                    yh = hardnet_like(f, A, b)
                    vh = float(np.max(A @ yh - b))
                    hard_max = max(hard_max, vh)
                    if vh > 1e-9:
                        hard_viol += 1
                    # truncated Gamma variants (mutation of the decomposition)
                    y1 = caffnet(f, A, b, w, p=2, kmax=1)
                    if np.max(A @ y1 - b) > 1e-9:
                        k1_viol += 1
                    kk = min(m, n_out)
                    ymax = f
                    best_d = np.inf
                    from caffnet_core import P_gamma
                    import itertools
                    if np.all(A @ f - b <= 1e-9):
                        ymax = f
                    else:
                        found = False
                        for gm in itertools.combinations(range(m), kk):
                            yy = P_gamma(f, A, b, gm, w)
                            if np.all(A @ yy - b <= 1e-9):
                                d = np.linalg.norm(yy - f)
                                if d < best_d:
                                    ymax, best_d, found = yy, d, True
                        if not found:
                            kmax_only_viol += 1
                            continue
                    if np.max(A @ ymax - b) > 1e-9:
                        kmax_only_viol += 1
    return dict(n_instances=tot,
                caffnet_violating_instances=caff_viol, caffnet_max_violation=caff_max,
                hardnet_like_violating_instances=hard_viol, hardnet_like_max_violation=hard_max,
                truncated_k_eq_1_violating=k1_viol,
                truncated_k_eq_min_only_violating=kmax_only_viol,
                max_cardinality_of_selected_gamma=max_k_used)


def fig1_example():
    """Sec. 1 example: {(x,y) : [0,1][x,y]^T <= 0}; the whole line y=0 (and below) is feasible;
    A is rank deficient in the column sense (n_out=2, rank 1) -> non-unique projection."""
    A = np.array([[0.0, 1.0], [0.0, 2.0]])   # linearly dependent rows
    b = np.array([0.0, 0.0])
    f = np.array([1.5, 2.0])                 # infeasible
    outs = {}
    for name, w in (("w_zero", np.zeros(2)), ("w_left", np.array([-5.0, 0.0])),
                    ("w_right", np.array([5.0, 0.0]))):
        y = caffnet(f, A, b, w, p=2)
        outs[name] = dict(y=y.tolist(), max_violation=float(np.max(A @ y - b)))
    yh = hardnet_like(f, A, b)
    outs["hardnet_like"] = dict(y=yh.tolist(), max_violation=float(np.max(A @ yh - b)))
    return outs


if __name__ == "__main__":
    t0 = time.time()
    res = dict(claim=3, sweep=sweep(), fig1_example=fig1_example(),
               command="python verify_claim3.py", numpy=np.__version__,
               seconds=round(time.time() - t0, 2))
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
