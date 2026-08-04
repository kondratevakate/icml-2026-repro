"""verify_claim2.py -- Theorem 2 (Section 3.1), X-conditional optimality.

Claim under test (paper, Thm 2, eqs. (4)-(6)):
For fixed x, the primal program
    min_{I_x: Y->[0,1]}  int I_x(y) dmu(y)
    s.t.  int I_x(y) f_k(y|x) dmu(y) >= 1-alpha  for all k in [K]
has an optimal solution of the superlevel-set form
    C*(x) = {y : h_{lambda*}(x,y) > 1} u S(x),  S(x) subset of {h_{lambda*}=1},
    h_lambda(x,y) = sum_k lambda_k(x) f_k(y|x),
where lambda* is optimal for the dual (5); and complementary slackness holds:
  (i)  lambda_k* > 0  =>  coverage_k(C*) = 1-alpha exactly,
  (ii) lambda_k* = 0  =>  coverage_k(C*) >= 1-alpha,
  (iii) there is at least one k* with lambda_{k*}* > 0  (=> exact 1-alpha for >=1 source).

Method: mu = counting measure on a finite grid Y of size m, so (4) is a finite LP.
We solve it with scipy.optimize.linprog (HiGHS) and read off the exact dual
variables. Exhaustive enumeration over the full configuration grid
K in {2,3,4,5} x m in {3,...,10} x alpha in {0.05,0.1,0.2} x 30 random instances each
(the claim space is a continuum, so we enumerate the discrete design axes fully
and use 30 seeds per cell -- never a single seed).

Mutation tests:
  M1  lambda -> uniform lambda with the same L1 norm as lambda*: the superlevel set
      {h_lambda > 1} must stop being both feasible-and-optimal.
  M2  invert the superlevel condition ({h_{lambda*} < 1}): must break feasibility.
  M3  drop one active constraint from complementary slackness bookkeeping: predicted
      objective must strictly decrease (infeasible for the dropped source).

Writes results/claim2.json.
"""
import json, os, time, itertools
import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-7


def solve_primal(F, alpha):
    """F: (K,m) conditional pmf values (mu = counting measure). Returns LP solution."""
    K, m = F.shape
    res = linprog(c=np.ones(m), A_ub=-F, b_ub=-np.ones(K) * (1 - alpha),
                  bounds=[(0.0, 1.0)] * m, method="highs")
    assert res.status == 0, res.message
    lam = np.asarray(res.ineqlin.marginals) * -1.0     # dual >= 0 for our sign convention
    return res.x, res.fun, lam


def dual_objective(lam, F, alpha):
    h = lam @ F
    return (1 - alpha) * lam.sum() - np.maximum(h - 1.0, 0.0).sum()


def superlevel_solution(lam, F, alpha):
    """Build C* = {h>1} u S with S chosen inside the tie set to just meet coverage."""
    K, m = F.shape
    h = lam @ F
    I = (h > 1.0 + TOL).astype(float)
    ties = np.where(np.abs(h - 1.0) <= 1e-9)[0]
    need = (1 - alpha) - F @ I
    if np.any(need > 1e-12) and ties.size:
        # choose S(x) inside the tie set {h_{lambda*} = 1}; any choice meeting the
        # constraints is optimal by Thm 2, we pick one by a small LP over the ties.
        sub = linprog(c=np.ones(ties.size), A_ub=-F[:, ties], b_ub=-np.maximum(need, 0.0),
                      bounds=[(0.0, 1.0)] * ties.size, method="highs")
        if sub.status == 0:
            I[ties] = sub.x
    return I, h


def one_instance(K, m, alpha, seed):
    rng = np.random.default_rng(seed)
    F = rng.gamma(0.8, 1.0, size=(K, m)) + 1e-3
    F /= F.sum(axis=1, keepdims=True)          # pmfs w.r.t. counting measure
    Iopt, val, lam = solve_primal(F, alpha)
    I_sl, h = superlevel_solution(lam, F, alpha)
    cov_sl = F @ I_sl
    size_sl = float(I_sl.sum())
    strong_duality_gap = abs(val - dual_objective(lam, F, alpha))
    active = lam > TOL
    cov_opt = F @ Iopt
    cs_i = bool(np.all(np.abs(cov_opt[active] - (1 - alpha)) < 1e-6)) if active.any() else False
    cs_ii = bool(np.all(cov_opt[~active] >= (1 - alpha) - 1e-6)) if (~active).any() else True
    cs_iii = bool(active.any())

    # ---- mutations -------------------------------------------------------
    lam_u = np.full(K, lam.sum() / K)
    I_m1 = (lam_u @ F > 1.0 + TOL).astype(float)
    m1_feasible = bool(np.all(F @ I_m1 >= (1 - alpha) - 1e-9))
    m1_size = float(I_m1.sum())

    I_m2 = (h < 1.0 - TOL).astype(float)        # inverted condition
    m2_feasible = bool(np.all(F @ I_m2 >= (1 - alpha) - 1e-9))
    m2_size = float(I_m2.sum())

    m3 = None
    if active.any():
        k_drop = int(np.argmax(lam))
        keep = [k for k in range(K) if k != k_drop]
        _, val_drop, _ = solve_primal(F[keep], alpha)
        m3 = dict(k_dropped=k_drop, size_without=float(val_drop), size_with=float(val))

    return dict(K=K, m=m, alpha=alpha, seed=seed,
                lp_optimal_size=float(val), superlevel_size=size_sl,
                size_gap=float(size_sl - val),
                superlevel_feasible=bool(np.all(cov_sl >= (1 - alpha) - 1e-6)),
                strong_duality_gap=float(strong_duality_gap),
                lambda_star=lam.tolist(),
                n_active=int(active.sum()),
                cs_i=cs_i, cs_ii=cs_ii, cs_iii=cs_iii,
                exact_coverage_sources=int(np.sum(np.abs(cov_opt - (1 - alpha)) < 1e-6)),
                mut_uniform_lambda=dict(feasible=m1_feasible, size=m1_size),
                mut_inverted_superlevel=dict(feasible=m2_feasible, size=m2_size),
                mut_drop_active_constraint=m3)


def main():
    t0 = time.time()
    rows = []
    for K, m, alpha in itertools.product([2, 3, 4, 5], range(3, 11), [0.05, 0.1, 0.2]):
        for s in range(30):
            rows.append(one_instance(K, m, alpha, seed=hash((K, m, int(alpha * 100), s)) % 2**31))
    n = len(rows)
    summary = dict(
        claim="Theorem 2 (Section 3.1): superlevel-set optimality of h_{lambda*} + complementary slackness",
        command="python verify_claim2.py",
        n_instances=n,
        grid="K in {2,3,4,5} x m in {3..10} x alpha in {0.05,0.1,0.2} x 30 seeds (exhaustive over design axes)",
        max_size_gap=float(max(r["size_gap"] for r in rows)),
        n_superlevel_matches_lp=int(sum(abs(r["size_gap"]) < 1e-6 for r in rows)),
        n_superlevel_feasible=int(sum(r["superlevel_feasible"] for r in rows)),
        max_strong_duality_gap=float(max(r["strong_duality_gap"] for r in rows)),
        n_cs_i=int(sum(r["cs_i"] for r in rows)),
        n_cs_ii=int(sum(r["cs_ii"] for r in rows)),
        n_cs_iii=int(sum(r["cs_iii"] for r in rows)),
        n_at_least_one_exact_coverage=int(sum(r["exact_coverage_sources"] >= 1 for r in rows)),
        mut_uniform_lambda_infeasible_or_larger=int(sum(
            (not r["mut_uniform_lambda"]["feasible"]) or
            (r["mut_uniform_lambda"]["size"] > r["lp_optimal_size"] + 1e-6) for r in rows)),
        mut_uniform_lambda_infeasible=int(sum(not r["mut_uniform_lambda"]["feasible"] for r in rows)),
        mut_inverted_superlevel_infeasible=int(sum(not r["mut_inverted_superlevel"]["feasible"] for r in rows)),
        mut_drop_active_strictly_smaller=int(sum(
            r["mut_drop_active_constraint"] is not None and
            r["mut_drop_active_constraint"]["size_without"] < r["mut_drop_active_constraint"]["size_with"] - 1e-9
            for r in rows)),
        runtime_sec=round(time.time() - t0, 1),
        rows=rows,
    )
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "claim2.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
