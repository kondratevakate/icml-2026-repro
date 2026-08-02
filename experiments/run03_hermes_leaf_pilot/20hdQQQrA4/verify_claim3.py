"""verify_claim3.py — no full-row-rank requirement + decomposition cardinality.

Claim (Section 3, Lemma 3.3 / Theorem 3.4, Table 1):
  Unlike HardNet, CAffNet guarantees A(x) y <= b(x) WITHOUT requiring A(x) to have
  full row rank, and handles arbitrary constraint cardinality m by decomposing into
  sub-constraint combinations of at most min(m, n_out) constraints.

Tests:
  E1 EXHAUSTIVE family: all 2-D systems built from a fixed finite dictionary of
     normal directions with duplicated / negated / scaled (hence linearly dependent)
     rows, all subsets of size m = 2..5 -> for every instance with a non-empty
     feasible set, and for a deterministic grid of f_theta and w_phi, the CAffNet
     candidate set must contain a feasible point (Lemma 3.3) and the output must
     satisfy all m constraints (Theorem 3.4). Failures counted.
  E2 RANDOM sweep with m up to 8 > n_out and forced dependent rows, 200 seeds x
     shapes, same assertions. Also records max violation ReLU(A y - b).
  E3 BASELINE: the HardNet-Aff single pseudo-inverse correction on the SAME
     instances -> fraction of infeasible outputs (must be > 0, otherwise CAffNet's
     advantage would not be demonstrated on these instances).
  E4 CARDINALITY: number of combinations |Gamma| = sum_{k<=min(m,n_out)} C(m,k)
     matches Eq. (2) and is <= 2^m - 1.

MUTATION (M-trunc): truncate Gamma to k <= min(m,n_out) - 1. If min(m,n_out) really
is the required cardinality, feasibility failures must appear. Run on the same
instances; number of failures reported.

Run: .venv/bin/python verify_claim3.py
"""
import itertools
import json
import math
import time

import numpy as np

from caffnet_core import caffnet_output, gamma_set, hardnet_aff, random_feasible_system

TOL = 1e-9


def check(A, b, f_th, w, kmax=None):
    y, info = caffnet_output(f_th, w, A, b, p=2.0, kmax=kmax)
    if y is None:
        return False, np.inf
    v = float(np.max(np.maximum(A @ y - b, 0.0)))
    return v <= TOL, v


def exhaustive_2d():
    """E1: exhaustive over a finite dictionary of directions in R^2."""
    ang = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4]
    dirs = [np.array([np.cos(a), np.sin(a)]) for a in ang]
    # add scaled duplicates -> guarantees linearly dependent rows in many subsets
    dirs = dirs + [2.0 * dirs[0], -1.0 * dirs[1], 3.0 * dirs[2]]
    idx = range(len(dirs))
    f_grid = [np.array([u, v]) for u in (-3.0, -0.7, 0.0, 0.7, 3.0)
              for v in (-3.0, -0.7, 0.0, 0.7, 3.0)]
    w_grid = [np.zeros(2), np.array([1.0, -2.0]), np.array([-5.0, 4.0])]
    n_inst = n_case = n_fail = n_infeasible_set = 0
    n_rank_def = 0
    maxviol = 0.0
    hn_fail = hn_total = 0
    mut_fail = mut_total = 0
    for m in (2, 3, 4, 5):
        for comb in itertools.combinations(idx, m):
            A = np.stack([dirs[i] for i in comb])
            b = np.ones(m) * 1.0            # contains 0 => feasible set non-empty
            n_inst += 1
            if np.linalg.matrix_rank(A) < min(m, 2):
                n_rank_def += 1
            for f_th, w in itertools.product(f_grid, w_grid):
                ok, v = check(A, b, f_th, w)
                n_case += 1
                maxviol = max(maxviol, 0.0 if ok else v)
                if not ok:
                    n_fail += 1
                yh = hardnet_aff(f_th, A, b)
                hn_total += 1
                if np.max(np.maximum(A @ yh - b, 0.0)) > 1e-7:
                    hn_fail += 1
                if min(m, 2) - 1 >= 1:       # mutation: truncated Gamma
                    okm, _ = check(A, b, f_th, w, kmax=min(m, 2) - 1)
                    mut_total += 1
                    if not okm:
                        mut_fail += 1
    return dict(n_systems=n_inst, n_rank_deficient_systems=n_rank_def,
                n_cases=n_case, n_caffnet_failures=n_fail,
                max_violation=maxviol,
                hardnet_infeasible=hn_fail, hardnet_cases=hn_total,
                hardnet_infeasible_frac=hn_fail / max(hn_total, 1),
                mutation_truncated_gamma_failures=mut_fail,
                mutation_cases=mut_total,
                mutation_failure_frac=mut_fail / max(mut_total, 1))


def random_sweep():
    """E2/E3: random systems, m possibly >> n_out, forced dependent rows."""
    n_fail = n_case = 0
    hn_fail = 0
    mut_fail = mut_total = 0
    maxviol = 0.0
    rank_def = 0
    for n_out in (1, 2, 3, 4):
        for m in (n_out, n_out + 1, n_out + 3, min(8, n_out + 5)):
            for dep in (0, 1, 2):
                if dep >= m:
                    continue
                for seed in range(200):
                    rng = np.random.default_rng([99, n_out, m, dep, seed])
                    A, b, y0 = random_feasible_system(rng, m, n_out, dep_rows=dep)
                    if np.linalg.matrix_rank(A) < min(m, n_out):
                        rank_def += 1
                    f_th = y0 + rng.normal(size=n_out) * 4.0
                    w = rng.normal(size=n_out) * 2.0
                    ok, v = check(A, b, f_th, w)
                    n_case += 1
                    if not ok:
                        n_fail += 1
                        maxviol = max(maxviol, v)
                    yh = hardnet_aff(f_th, A, b)
                    if np.max(np.maximum(A @ yh - b, 0.0)) > 1e-7:
                        hn_fail += 1
                    kk = min(m, n_out)
                    if kk - 1 >= 1:
                        okm, _ = check(A, b, f_th, w, kmax=kk - 1)
                        mut_total += 1
                        if not okm:
                            mut_fail += 1
    return dict(n_cases=n_case, n_caffnet_failures=n_fail,
                max_violation=maxviol, n_rank_deficient=rank_def,
                hardnet_infeasible=hn_fail,
                hardnet_infeasible_frac=hn_fail / max(n_case, 1),
                mutation_truncated_gamma_failures=mut_fail,
                mutation_cases=mut_total,
                mutation_failure_frac=mut_fail / max(mut_total, 1))


def cardinality_check():
    rows = []
    ok = True
    for m in range(1, 11):
        for n_out in range(1, 6):
            g = gamma_set(m, n_out)
            formula = sum(math.comb(m, k) for k in range(1, min(m, n_out) + 1))
            kmax = max(len(x) for x in g)
            good = (len(g) == formula and kmax <= min(m, n_out)
                    and len(g) <= 2 ** m - 1)
            ok &= good
            rows.append(dict(m=m, n_out=n_out, size=len(g), formula=formula,
                             max_k=kmax, bound_2m_minus_1=2 ** m - 1, ok=good))
    return dict(all_ok=bool(ok), rows=rows)


if __name__ == "__main__":
    t0 = time.time()
    res = {"claim": 3,
           "claim_text": "CAffNet guarantees A(x)y<=b(x) without full row rank of A(x) "
                         "and handles arbitrary m via sub-constraints of cardinality "
                         "at most min(m, n_out)",
           "source": "Section 3 / 3.1 (Eq. 2), Lemma 3.3 (App. A), Theorem 3.4 (App. B), Table 1",
           "command": ".venv/bin/python verify_claim3.py"}
    res["E1_exhaustive_2d"] = exhaustive_2d()
    res["E2E3_random_sweep"] = random_sweep()
    res["E4_cardinality"] = cardinality_check()
    res["elapsed_s"] = time.time() - t0
    json.dump(res, open("results/claim3.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "E4_cardinality"}, indent=2))
    print("E4 all_ok:", res["E4_cardinality"]["all_ok"])
