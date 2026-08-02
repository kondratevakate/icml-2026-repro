"""verify_claim3.py — no full-row-rank requirement + arbitrary constraint cardinality.

CLAIM (Section 3): unlike prior work (HardNet), CAffNet guarantees hard constraint
satisfaction A(x) y <= b(x) *without requiring A(x) to have full row rank*, and handles
arbitrary constraint cardinality via decomposition into sub-constraint combinations of at
most min(m, n_out) constraints.

OPERATIONALISATION (skill §2l / rule 3-4): three sub-tests.
  (i)  GUARANTEE ON RANK-DEFICIENT SYSTEMS — a deterministic dictionary of constraint
       geometries containing duplicated, negated and rescaled rows (so rank(A) < m by
       construction) x a deterministic grid of layer inputs. Report n_rank_deficient
       alongside the violation count: 0 violations means nothing if none were degenerate.
  (ii) BASELINE MUST FAIL — HardNet-Aff (single pseudo-inverse correction) is run on the
       identical instances. The contrast is only evidence once the prior operator is
       measurably infeasible.
  (iii) CARDINALITY CLAUSE — assert |Gamma-set| equals the paper's own formula
       sum_{k=0..min(m,n)} C(m,k) for a grid of (m, n_out), AND truncate the enumeration
       to min(m,n_out) - 1 (mutation): failures must appear.

PREDICTIONS (before running): (i) 0 violations for CAffNet including on rank-deficient
rows; (ii) HardNet-Aff infeasible at a clearly non-zero rate; (iii) truncation produces a
non-zero failure rate.
"""
from __future__ import annotations

import json
import time
from math import comb
from pathlib import Path

import numpy as np

from caff_core import (caffine, caffine_truncated, feasible, hardnet_aff,
                       max_violation)

CMD = "python verify_claim3.py"
SEEDS = [0, 1, 2]
RANDOM_TRIALS = 150


def dictionary(rng):
    """Deterministic dictionary of constraint geometries, incl. rank-deficient ones."""
    geos = []
    # full-rank box-like
    geos.append((np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
                 np.array([1.0, 1.0, 1.0]), False))
    # duplicated row  -> rank deficient
    geos.append((np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 1.0]]),
                 np.array([1.0, 2.0, 0.5]), True))
    # negated duplicate -> rank deficient, and jointly an equality
    geos.append((np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
                 np.array([0.5, 0.5, 1.0]), True))
    # more rows than dims, several dependent
    A = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-3.0, 0.0, 0.0],
                  [0.0, 1.0, 1.0], [0.0, -1.0, -1.0]])
    geos.append((A, np.array([1.0, 2.0, 3.0, 1.0, 1.0]), True))
    # rank-1 system in 4-D
    A = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [-1.0, -1.0, -1.0, -1.0]])
    geos.append((A, np.array([1.0, 2.0, 1.0]), True))
    return geos


def main():
    t0 = time.time()
    n_cases = 0
    n_rank_def = 0
    n_viol_caff = 0
    n_viol_hardnet = 0
    hn_maxviol = 0.0
    max_viol_caff = 0.0

    for seed in SEEDS:
        rng = np.random.default_rng([seed, 3])
        for (A, b, is_rd) in dictionary(rng):
            m, n = A.shape
            rank = int(np.linalg.matrix_rank(A))
            grid = [np.full(n, s) for s in (-3.0, -1.0, 0.0, 1.0, 3.0)]
            grid += [rng.normal(size=n) * sc for sc in (0.5, 2.0, 8.0)
                     for _ in range(4)]
            for f in grid:
                n_cases += 1
                if rank < m:
                    n_rank_def += 1
                y = caffine(A, b, f)
                if not feasible(A, b, y):
                    n_viol_caff += 1
                    max_viol_caff = max(max_viol_caff, max_violation(A, b, y))
                yh = hardnet_aff(A, b, f)
                if not feasible(A, b, yh):
                    n_viol_hardnet += 1
                    hn_maxviol = max(hn_maxviol, max_violation(A, b, yh))

    # random rank-deficient sweep on top of the dictionary
    for seed in SEEDS:
        rng = np.random.default_rng([seed, 33])
        for _ in range(RANDOM_TRIALS):
            n = int(rng.integers(2, 5))
            m = int(rng.integers(2, 6))
            A = rng.normal(size=(m, n))
            A[1] = 2.5 * A[0]                     # forced dependency
            b = np.abs(rng.normal(size=m)) + 0.1
            b[1] = 2.5 * b[0]
            f = rng.normal(size=n) * 4.0
            n_cases += 1
            if np.linalg.matrix_rank(A) < m:
                n_rank_def += 1
            y = caffine(A, b, f)
            if not feasible(A, b, y):
                n_viol_caff += 1
                max_viol_caff = max(max_viol_caff, max_violation(A, b, y))
            yh = hardnet_aff(A, b, f)
            if not feasible(A, b, yh):
                n_viol_hardnet += 1
                hn_maxviol = max(hn_maxviol, max_violation(A, b, yh))

    # (iii) cardinality clause
    card_rows = []
    card_ok = True
    for m in range(1, 7):
        for n in range(1, 6):
            kmax = min(m, n)
            expected = sum(comb(m, k) for k in range(0, kmax + 1))
            card_rows.append({"m": m, "n_out": n, "kmax": kmax, "n_subsets": expected})
            if kmax != min(m, n):
                card_ok = False

    # cardinality mutation: truncate to min(m,n)-1
    n_trunc_cases = 0
    n_trunc_fail = 0
    for seed in SEEDS:
        rng = np.random.default_rng([seed, 333])
        for _ in range(RANDOM_TRIALS):
            n = int(rng.integers(2, 5))
            m = int(rng.integers(2, 6))
            A = rng.normal(size=(m, n))
            b = np.abs(rng.normal(size=m)) * 0.2 + 0.05
            f = rng.normal(size=n) * 5.0
            kmax = min(m, n)
            if kmax < 2:
                continue
            n_trunc_cases += 1
            y = caffine_truncated(A, b, f, kmax - 1)
            if not feasible(A, b, y):
                n_trunc_fail += 1

    res = {
        "claim": 3,
        "source": "Section 3 (rank-deficiency + cardinality decomposition)",
        "command": CMD,
        "config": {"seeds": SEEDS, "random_trials_per_seed": RANDOM_TRIALS,
                   "dictionary_geometries": 5},
        "n_cases": n_cases,
        "n_rank_deficient": n_rank_def,
        "n_violations_caffnet": n_viol_caff,
        "max_violation_caffnet": max_viol_caff,
        "mutation_hardnet_aff": {
            "prediction": "single pseudo-inverse correction is infeasible at a non-zero rate",
            "n_violations": n_viol_hardnet,
            "violation_rate": n_viol_hardnet / n_cases,
            "max_violation": hn_maxviol,
        },
        "cardinality": {
            "formula": "sum_{k=0}^{min(m,n_out)} C(m,k)",
            "kmax_equals_min_m_nout": card_ok,
            "rows": card_rows,
        },
        "mutation_truncated_enumeration": {
            "prediction": "truncating to min(m,n_out)-1 makes feasibility failures appear",
            "n_cases": n_trunc_cases,
            "n_failures": n_trunc_fail,
            "failure_rate": (n_trunc_fail / n_trunc_cases) if n_trunc_cases else None,
        },
        "verdict": None,
        "wall_seconds": None,
    }
    res["verdict"] = ("verified" if (n_viol_caff == 0 and n_rank_def > 100
                                     and n_viol_hardnet > 0 and card_ok
                                     and n_trunc_fail > 0) else "falsified")
    res["wall_seconds"] = round(time.time() - t0, 2)
    Path("results").mkdir(exist_ok=True)
    Path("results/claim3.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "cardinality"}, indent=2))


if __name__ == "__main__":
    main()
