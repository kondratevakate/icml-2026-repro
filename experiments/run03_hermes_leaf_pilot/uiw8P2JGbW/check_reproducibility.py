"""
check_reproducibility.py

Re-runs every claim's computation from first principles using the pinned seed
(20260501) and asserts that the results are DETERMINISTIC and match the values
recorded in results/claim<N>.json. Run with the project venv:

    . .venv/bin/activate
    python check_reproducibility.py

Exits 0 if all claims reproduce exactly (within float tolerance); non-zero
otherwise.
"""
import json
import sys
import numpy as np

import verify_claim1 as c1
import verify_claim2 as c2
import verify_claim3 as c3
import verify_claim4 as c4
import verify_claim5 as c5
import verify_claim6 as c6

TOL = 1e-6


def approx(a, b):
    return abs(float(a) - float(b)) <= TOL


def check(claim_no, recomputed, saved, key_checks):
    """key_checks: list of (getter, saved_value) tuples to compare."""
    problems = []
    if recomputed["verdict"] != saved["verdict"]:
        problems.append(f"verdict mismatch: {recomputed['verdict']} "
                        f"vs {saved['verdict']}")
    for getter, saved_val in key_checks:
        try:
            got = getter(recomputed)
        except Exception as e:  # noqa
            problems.append(f"key missing/error: {e}")
            continue
        if not approx(got, saved_val):
            problems.append(f"value mismatch: got {got} vs saved {saved_val}")
    return problems


def main():
    problems = []

    # Claim 1
    r1 = c1.main(); s1 = json.load(open("results/claim1.json"))
    p1 = check(1, r1, s1, [
        (lambda d: d["stress_test"]["failures (Rev_A < Rev_B)"],
         s1["stress_test"]["failures (Rev_A < Rev_B)"]),
    ])
    problems += [(1, x) for x in p1]

    # Claim 2
    r2 = c2.main(); s2 = json.load(open("results/claim2.json"))
    p2 = check(2, r2, s2, [
        (lambda d: d["convexity_f"]["violations_of_convexity"],
         s2["convexity_f"]["violations_of_convexity"]),
        (lambda d: d["mean_preserving_spread"]["jensen_inequality_failures"],
         s2["mean_preserving_spread"]["jensen_inequality_failures"]),
    ])
    problems += [(2, x) for x in p2]

    # Claim 3
    r3 = c3.main(); s3 = json.load(open("results/claim3.json"))
    p3 = check(3, r3, s3, [
        (lambda d: d["welfare_monotonicity_corollary"]["failures (Welfare_A < Welfare_B)"],
         s3["welfare_monotonicity_corollary"]["failures (Welfare_A < Welfare_B)"]),
    ])
    problems += [(3, x) for x in p3]

    # Claim 4
    r4 = c4.main(); s4 = json.load(open("results/claim4.json"))
    p4 = check(4, r4, s4, [
        (lambda d: d["coarse_model"]["revenue"], s4["coarse_model"]["revenue"]),
        (lambda d: d["fine_model"]["revenue"], s4["fine_model"]["revenue"]),
        (lambda d: d["non_monotonicity"]["revenue_drop_pct"],
         s4["non_monotonicity"]["revenue_drop_pct"]),
    ])
    problems += [(4, x) for x in p4]

    # Claim 5
    r5 = c5.main(); s5 = json.load(open("results/claim5.json"))
    p5 = check(5, r5, s5, [
        (lambda d: d["coarse_model"]["revenue"], s5["coarse_model"]["revenue"]),
        (lambda d: d["fine_model"]["revenue"], s5["fine_model"]["revenue"]),
        (lambda d: d["non_monotonicity"]["revenue_drop_pct"],
         s5["non_monotonicity"]["revenue_drop_pct"]),
    ])
    problems += [(5, x) for x in p5]

    # Claim 6
    r6 = c6.main(); s6 = json.load(open("results/claim6.json"))
    p6 = check(6, r6, s6, [
        (lambda d: d["lp_lifting"]["lift_infeasible_or_objective_drop"],
         s6["lp_lifting"]["lift_infeasible_or_objective_drop"]),
        (lambda d: d["n_monotonicity_settings"], s6["n_monotonicity_settings"]),
    ])
    problems += [(6, x) for x in p6]

    if problems:
        print("REPRODUCIBILITY CHECK FAILED:")
        for pn, msg in problems:
            print(f"  Claim {pn}: {msg}")
        return 1
    print("REPRODUCIBILITY CHECK PASSED: all 6 claims reproduce deterministically "
          "under seed 20260501.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
