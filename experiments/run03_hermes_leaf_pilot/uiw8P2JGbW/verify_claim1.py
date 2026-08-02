"""
Claim 1 — Theorem 5.1 (FPA Revenue Monotonicity for tCPA, mu=1).

Anchored claim: For tCPA bidders without budget constraints using uniform
bidding with multiplier mu=1 in first-price auctions, refining the prediction
model never decreases platform revenue: Rev(M_A) >= Rev(M_B) when M_A refines
M_B.

Reproduction strategy:
  (1) Random stress test: generate many (M_B coarse, M_A refined) instances,
      compute Rev(M) = sum_C w_C max_i t_i p_i,C, confirm Rev(M_A) >= Rev(M_B)
      in every instance (modulo float tolerance).
  (2) Mutation test: drop the refinement relation (M_A not a refinement of
      M_B) and/or break calibration preservation; the monotonicity guarantee
      must break (Rev can strictly decrease).
"""
import json
import numpy as np
from common import (build_model, is_refinement, fpa_revenue_tcpa,
                    refinement_subclusters)

SEED = 20260501
TOL = 1e-9


def make_instance(rng, n_adv=4, n_users=40, n_coarse=6, n_sub=3):
    true_p = rng.random((n_adv, n_users))
    targets = rng.uniform(0.5, 5.0, size=n_adv)
    # coarse partition
    coarse = [[] for _ in range(n_coarse)]
    for u in range(n_users):
        coarse[rng.integers(n_coarse)].append(u)
    coarse = [c for c in coarse if c]
    # refined: split each coarse cluster into n_sub random sub-clusters
    fine = []
    for c in coarse:
        parts = [[] for _ in range(n_sub)]
        for u in c:
            parts[rng.integers(n_sub)].append(u)
        fine.extend(p for p in parts if p)
    MB = build_model(coarse, true_p)
    MA = build_model(fine, true_p)
    return MB, MA, targets


def main():
    rng = np.random.default_rng(SEED)
    n_instances = 2000
    failures = 0
    min_margin = np.inf
    example = None
    for k in range(n_instances):
        MB, MA, targets = make_instance(rng)
        # sanity: refinement must hold
        assert is_refinement(MA, MB), "constructed M_A must refine M_B"
        rB = fpa_revenue_tcpa(MB, targets)
        rA = fpa_revenue_tcpa(MA, targets)
        margin = rA - rB
        min_margin = min(min_margin, margin)
        if margin < -TOL:
            failures += 1
        if k == 0:
            example = (MB, MA, targets, rB, rA)
    # ---- mutation test 1: drop refinement ----
    rng2 = np.random.default_rng(SEED + 1)
    mut_fail = 0
    mut_trials = 2000
    for _ in range(mut_trials):
        MB, MA, targets = make_instance(rng2)
        # build a NON-refining M_A: a fresh random partition of same users
        true_p = MA["true_p"]
        n_users = true_p.shape[1]
        alt = [[] for _ in range(5)]
        for u in range(n_users):
            alt[rng2.integers(5)].append(u)
        alt = [c for c in alt if c]
        MAlt = build_model(alt, true_p)
        # MAlt is NOT guaranteed to refine MB; if it doesn't, monotonicity
        # need not hold.
        if not is_refinement(MAlt, MB):
            rB = fpa_revenue_tcpa(MB, targets)
            rA = fpa_revenue_tcpa(MAlt, targets)
            if rA < rB - TOL:
                mut_fail += 1
    # ---- mutation test 2: break calibration preservation ----
    rng3 = np.random.default_rng(SEED + 2)
    calib_fail = 0
    for _ in range(mut_trials):
        MB, MA, targets = make_instance(rng3)
        # corrupt MA predictions: replace with random (mis-calibrated) values
        MA_bad = dict(MA)
        MA_bad = {**MA, "pred": rng3.random(MA["pred"].shape)}
        rB = fpa_revenue_tcpa(MB, targets)
        rA = fpa_revenue_tcpa(MA_bad, targets)
        if rA < rB - TOL:
            calib_fail += 1

    verdict = "verified" if failures == 0 else "falsified"
    result = {
        "claim": 1,
        "statement": ("FPA revenue monotonicity for tCPA bidders (mu=1, no budget): "
                      "Rev(M_A) >= Rev(M_B) whenever M_A refines M_B (Theorem 5.1)."),
        "source": "Theorem 5.1, Section 5.2.1, proof Appendix A.2 (arxiv 2605.31036)",
        "verdict": verdict,
        "seed": SEED,
        "method": ("Random stress test over calibrated coarse/refined model pairs; "
                   "revenue computed as Rev(M)=sum_C w_C*max_i t_i*p_i,C."),
        "stress_test": {
            "n_instances": n_instances,
            "failures (Rev_A < Rev_B)": failures,
            "min_margin_RevA_minus_RevB": float(min_margin),
        },
        "illustrative_example": {
            "Rev(M_B)": float(example[3]),
            "Rev(M_A)": float(example[4]),
            "margin": float(example[4] - example[3]),
        },
        "mutation_tests": {
            "non_refinement": {
                "description": ("Construct M_A NOT refining M_B; monotonicity no "
                                "longer guaranteed -> Rev can decrease."),
                "trials": mut_trials,
                "instances_where_RevA_lt_RevB": mut_fail,
                "property_breaks": mut_fail > 0,
            },
            "broken_calibration": {
                "description": ("Replace calibrated predictions with random values; "
                                "Jensen step invalid -> Rev can decrease."),
                "trials": mut_trials,
                "instances_where_RevA_lt_RevB": calib_fail,
                "property_breaks": calib_fail > 0,
            },
        },
    }
    return result


if __name__ == "__main__":
    res = main()
    from common import dump_json, print_json
    dump_json(res, "results/claim1.json")
    print_json(res)
