"""
Claim 2 — Proof mechanism of Theorem 5.1.

Anchored claim: The FPA revenue monotonicity proof for tCPA bidders relies on
convexity of f(p) = max_i (t_i * p_i) via Jensen's inequality, since model
refinement corresponds to a mean-preserving spread of this convex function.

Reproduction strategy:
  (1) Numerically verify f(p)=max_i t_i p_i is convex (Jensen holds for random
      points); contrast with a non-convex function (min) where Jensen fails.
  (2) Verify that refinement induces a *mean-preserving spread*: for each coarse
      cluster, the sub-cluster predictions form a distribution (weights lambda_j)
      whose mean equals the coarse prediction (calibration preservation).
  (3) Jensen demonstration: per coarse cluster,
        sum_j lambda_j f(p_sub_j)  >=  f( sum_j lambda_j p_sub_j ) = f(p_coarse),
      which is exactly the inequality Rev(M_A) >= Rev(M_B).
  (4) Mutation: replace f by a non-convex function (min) and/or break the
      mean-preserving-spread condition; the inequality must break.
"""
import json
import numpy as np
from common import build_model, is_refinement, refinement_subclusters

SEED = 20260501
TOL = 1e-9


def f_convex(p, t):
    return float(np.max(np.asarray(t, float) * np.asarray(p, float)))


def f_concave(p, t):  # non-convex (actually concave) contrast
    return float(np.min(np.asarray(t, float) * np.asarray(p, float)))


def make_instance(rng, n_adv=4, n_users=40, n_coarse=6, n_sub=3):
    true_p = rng.random((n_adv, n_users))
    targets = rng.uniform(0.5, 5.0, size=n_adv)
    coarse = [[] for _ in range(n_coarse)]
    for u in range(n_users):
        coarse[rng.integers(n_coarse)].append(u)
    coarse = [c for c in coarse if c]
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
    # ---- (1) convexity of f ----
    n_conv = 5000
    conv_violations = 0
    nc_violations = 0  # for the concave contrast we expect violations
    for _ in range(n_conv):
        n_adv = rng.integers(2, 6)
        t = rng.uniform(0.3, 5.0, size=n_adv)
        x = rng.random(n_adv)
        y = rng.random(n_adv)
        lam = rng.random()
        mx = lam * x + (1 - lam) * y
        lhs_conv = lam * f_convex(x, t) + (1 - lam) * f_convex(y, t)
        rhs_conv = f_convex(mx, t)
        if lhs_conv + TOL < rhs_conv:
            conv_violations += 1
        # concave contrast: Jensen should frequently FAIL
        lhs_nc = lam * f_concave(x, t) + (1 - lam) * f_concave(y, t)
        rhs_nc = f_concave(mx, t)
        if lhs_nc + TOL < rhs_nc:
            nc_violations += 1

    # ---- (2)+(3) mean-preserving spread + Jensen per cluster ----
    rng2 = np.random.default_rng(SEED + 7)
    n_inst = 2000
    mps_fail = 0
    jensen_fail = 0
    min_jensen_margin = np.inf
    ex = None
    for k in range(n_inst):
        MB, MA, targets = make_instance(rng2)
        assert is_refinement(MA, MB)
        mapping = refinement_subclusters(MA, MB)
        for cb_idx, fine_idxs in enumerate(mapping):
            # coarse prediction vector
            p_coarse = MB["pred"][:, cb_idx]
            w_coarse = MB["weights"][cb_idx]
            # sub-cluster weights (within coarse cluster) and predictions
            sub_w = np.array([MA["weights"][fj] for fj in fine_idxs])
            lam = sub_w / w_coarse
            # (2) mean-preserving spread: mean of sub predictions == coarse
            p_mean = sum(lam[j] * MA["pred"][:, fine_idxs[j]]
                         for j in range(len(fine_idxs)))
            if not np.allclose(p_mean, p_coarse, atol=1e-9):
                mps_fail += 1
            # (3) Jensen: sum lambda_j f(p_sub_j) >= f(p_coarse)
            lhs = sum(lam[j] * f_convex(MA["pred"][:, fine_idxs[j]], targets)
                      for j in range(len(fine_idxs)))
            rhs = f_convex(p_coarse, targets)
            margin = lhs - rhs
            min_jensen_margin = min(min_jensen_margin, margin)
            if lhs + TOL < rhs:
                jensen_fail += 1
        if k == 0:
            ex = (targets.tolist(), float(min_jensen_margin))

    # ---- (4) mutation: non-convex f or broken MPS ----
    # mutation A: use a non-convex function where Jensen reverses
    rng3 = np.random.default_rng(SEED + 9)
    mutA = 0
    for _ in range(n_inst):
        MB, MA, targets = make_instance(rng3)
        mapping = refinement_subclusters(MA, MB)
        for cb_idx, fine_idxs in enumerate(mapping):
            p_coarse = MB["pred"][:, cb_idx]
            w_coarse = MB["weights"][cb_idx]
            sub_w = np.array([MA["weights"][fj] for fj in fine_idxs])
            lam = sub_w / w_coarse
            lhs = sum(lam[j] * f_concave(MA["pred"][:, fine_idxs[j]], targets)
                      for j in range(len(fine_idxs)))
            rhs = f_concave(p_coarse, targets)
            if lhs + TOL < rhs:  # Jensen fails for non-convex f
                mutA += 1
                break
    # mutation B: break MPS (sub predictions no longer average to coarse)
    rng4 = np.random.default_rng(SEED + 11)
    mutB = 0
    for _ in range(n_inst):
        MB, MA, targets = make_instance(rng4)
        MA_bad = {**MA, "pred": rng4.random(MA["pred"].shape)}
        mapping = refinement_subclusters(MA_bad, MB)
        for cb_idx, fine_idxs in enumerate(mapping):
            p_coarse = MB["pred"][:, cb_idx]
            w_coarse = MB["weights"][cb_idx]
            sub_w = np.array([MA_bad["weights"][fj] for fj in fine_idxs])
            lam = sub_w / w_coarse
            lhs = sum(lam[j] * f_convex(MA_bad["pred"][:, fine_idxs[j]], targets)
                      for j in range(len(fine_idxs)))
            rhs = f_convex(p_coarse, targets)
            if lhs + TOL < rhs:
                mutB += 1
                break

    verdict = "verified" if (conv_violations == 0 and mps_fail == 0
                             and jensen_fail == 0) else "falsified"
    result = {
        "claim": 2,
        "statement": ("FPA revenue monotonicity proof uses convexity of "
                      "f(p)=max_i(t_i p_i) + Jensen over a mean-preserving spread "
                      "induced by refinement (Theorem 5.1, Section 5)."),
        "source": "Theorem 5.1 proof, Section 5.2.1 / Appendix A.2 (arxiv 2605.31036)",
        "verdict": verdict,
        "seed": SEED,
        "method": ("(a) random Jensen checks on f; (b) per-cluster verification of "
                   "mean-preserving spread (calibration) and the Jensen inequality "
                   "sum_j lambda_j f(p_sub_j) >= f(p_coarse)."),
        "convexity_f": {
            "random_jensen_trials": n_conv,
            "violations_of_convexity": conv_violations,
            "nonconvex_min_jensen_violations_(expected>0)": nc_violations,
        },
        "mean_preserving_spread": {
            "instances": n_inst,
            "mps_calibration_failures": mps_fail,
            "jensen_inequality_failures": jensen_fail,
            "min_jensen_margin": float(min_jensen_margin),
        },
        "mutation_tests": {
            "nonconvex_f": {
                "description": "Replace f by non-convex min; Jensen must fail.",
                "jensen_failures_observed": mutA,
                "property_breaks": mutA > 0,
            },
            "broken_mps": {
                "description": ("Break calibration so sub predictions don't average "
                                "to coarse; Jensen inequality must fail."),
                "jensen_failures_observed": mutB,
                "property_breaks": mutB > 0,
            },
        },
    }
    return result


if __name__ == "__main__":
    res = main()
    from common import dump_json, print_json
    dump_json(res, "results/claim2.json")
    print_json(res)
