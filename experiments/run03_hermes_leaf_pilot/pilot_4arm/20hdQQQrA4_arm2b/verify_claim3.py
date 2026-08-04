"""Claim 3 (Section 3, Lemma 3.3 / Theorem 3.4): CAffNet guarantees A(x)y <= b(x) WITHOUT
requiring A(x) to have full row rank, for arbitrary constraint cardinality m, via
decomposition into sub-constraints of at most min(m, n_out) constraints.

Tests (executed):
  T1 Feasibility sweep: random feasible polyhedra with m >> n_out (so A has NOT full row
     rank whenever m > n_out) and additionally explicitly rank-deficient / redundant /
     duplicated-row A. For every instance: does CAffNet (Eq 12) return a point with
     A y <= b? Count violations. Also record max violation.
  T2 HardNet baseline on the same instances: the full-row-rank formula (uses (A A^T)^-1);
     count how often it is undefined (rank deficient) or leaves violations.
  T3 Cardinality: assert max |gamma| over Gamma == min(m, n_out) and |Gamma| ==
     sum_k C(m,k) <= 2^m - 1 (Eq 2), exhaustively for m=1..10, n_out=1..5.
  MUTATION: truncate Gamma to k <= min(m,n_out)-1 (i.e. violate the stated cardinality)
     and re-run T1 -> feasibility must break on some instances, proving the
     min(m, n_out) cardinality is necessary, not decorative.

Run: ./.venv/bin/python verify_claim3.py
"""
import json, math, os
from itertools import combinations
import numpy as np
from caffnet_core import caffnet, hardnet, gammas, random_feasible_problem

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "claim3.json")
TOL = 1e-9


def make_instance(rng, n_out, m, kind):
    A, b, y0 = random_feasible_problem(rng, n_out, m)
    if kind == "duplicate_rows":            # redundant -> row rank < m
        A[1::2] = A[0::2][: len(A[1::2])]
        b = A @ y0 + np.abs(rng.normal(size=m)) * 0.5 + 1e-3
    elif kind == "rank_deficient":          # rows span a low-dim subspace
        r = max(1, n_out - 1)
        B = rng.normal(size=(r, n_out))
        A = rng.normal(size=(m, r)) @ B
        b = A @ y0 + np.abs(rng.normal(size=m)) * 0.5 + 1e-3
    # tighten: put y0 on several boundaries so f_theta usually violates
    act = rng.choice(m, size=min(m, max(1, n_out)), replace=False)
    b[act] = A[act] @ y0
    return A, b, y0


def run():
    rng_m = np.random.default_rng(11)
    configs = [(1, 4), (2, 5), (2, 8), (3, 7), (3, 10), (4, 9), (5, 12)]
    kinds = ["generic", "duplicate_rows", "rank_deficient"]
    n_seeds = 100
    rows = []
    tot = caf_fail = caf_viol = 0
    hard_undef = hard_viol = 0
    mut_fail = mut_tot = 0
    max_caf_viol = 0.0
    for n_out, m in configs:
        for kind in kinds:
            cf = cv = hu = hv = mf = 0
            for _ in range(n_seeds):
                rng = np.random.default_rng(rng_m.integers(1 << 31))
                A, b, y0 = make_instance(rng, n_out, m, kind)
                f = y0 + rng.normal(size=n_out) * 2.0
                w = rng.normal(size=n_out) * 0.5
                tot += 1
                y, info = caffnet(f, A, b, w=w, p=2)
                if y is None:
                    cf += 1
                else:
                    v = float(np.max(A @ y - b))
                    max_caf_viol = max(max_caf_viol, v)
                    if v > 1e-8:
                        cv += 1
                yh, st = hardnet(f, A, b)
                if yh is None:
                    hu += 1
                elif np.max(A @ yh - b) > 1e-8:
                    hv += 1
                # mutation: cardinality capped one below min(m, n_out)
                kcap = min(m, n_out) - 1
                mut_tot += 1
                if kcap >= 1:
                    ym, _ = caffnet(f, A, b, w=w, p=2, kmax=kcap)
                    if ym is None or np.max(A @ ym - b) > 1e-8:
                        mf += 1
                else:
                    mf += 0
            caf_fail += cf; caf_viol += cv; hard_undef += hu; hard_viol += hv
            mut_fail += mf
            rows.append({"n_out": n_out, "m": m, "kind": kind, "n": n_seeds,
                         "caffnet_no_feasible_candidate": cf,
                         "caffnet_violations": cv,
                         "hardnet_undefined_rank_deficient": hu,
                         "hardnet_violations": hv,
                         "mutation_kmax_minus_1_failures": mf})
    # T3 cardinality / count identity
    card_ok, count_ok = True, True
    for m in range(1, 11):
        for n_out in range(1, 6):
            G = gammas(m, n_out)
            kmax = max(len(g) for g in G)
            card_ok &= (kmax == min(m, n_out))
            expected = sum(math.comb(m, k) for k in range(1, min(m, n_out) + 1))
            count_ok &= (len(G) == expected <= 2 ** m - 1)
    res = {
        "claim": 3,
        "source": "Section 3.1-3.2, Lemma 3.3, Theorem 3.4, Eq (2)",
        "command": "./.venv/bin/python verify_claim3.py",
        "n_instances_total": tot,
        "caffnet_instances_with_no_feasible_candidate": caf_fail,
        "caffnet_instances_with_constraint_violation": caf_viol,
        "caffnet_max_violation_over_all_instances": max_caf_viol,
        "hardnet_instances_undefined_due_to_rank_deficiency": hard_undef,
        "hardnet_instances_with_constraint_violation": hard_viol,
        "MUTATION_kmax_minus_1_failure_count": mut_fail,
        "MUTATION_kmax_minus_1_total": mut_tot,
        "T3_max_cardinality_equals_min_m_nout": card_ok,
        "T3_gamma_count_identity_and_2m_bound": count_ok,
        "per_config": rows,
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_config"}, indent=2))


if __name__ == "__main__":
    run()
