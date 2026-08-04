"""Claim 1 (Theorem 5.1): FPA + tCPA, no budgets, uniform bidding mu=1.
Model refinement (calibration-preserving) never decreases platform revenue.

Test: 20000 random refined/coarse instance pairs; check Rev(fine) >= Rev(coarse) - tol.
Mutation A: break calibration (predictions perturbed off the conditional mean).
Mutation B: replace convex max-aggregator with the concave min-aggregator.
"""
import json, pathlib
import numpy as np
from common import (RNG_DEFAULT, random_refined_instance, coarsen, fp_revenue)

OUT = pathlib.Path(__file__).parent / "results" / "claim1.json"
N = 20000


def main():
    rng = np.random.default_rng(RNG_DEFAULT)
    gaps, viol = [], 0
    mut_a_viol = mut_b_viol = 0
    for _ in range(N):
        t, mass, pf, group = random_refined_instance(rng)
        mc, pc = coarsen(mass, pf, group)
        rf, rc = fp_revenue(t, mass, pf), fp_revenue(t, mc, pc)
        gaps.append(rf - rc)
        if rf < rc - 1e-12:
            viol += 1
        # Mutation A: miscalibrated coarse predictions (multiplicative noise, mean broken)
        pc_bad = np.clip(pc * rng.uniform(0.7, 1.3, size=pc.shape), 0, 1)
        if fp_revenue(t, mass, pf) < fp_revenue(t, mc, pc_bad) - 1e-12:
            mut_a_viol += 1
        # Mutation B: concave aggregator min_i t_i p_i  (Jensen reverses)
        bf = (mass * (pf * t).min(axis=1)).sum()
        bc = (mc * (pc * t).min(axis=1)).sum()
        if bf < bc - 1e-12:
            mut_b_viol += 1
    gaps = np.array(gaps)
    res = {
        "claim": 1,
        "source": "Theorem 5.1",
        "seed": RNG_DEFAULT,
        "n_instances": N,
        "violations_of_Rev_fine_ge_Rev_coarse": int(viol),
        "mean_revenue_gain": float(gaps.mean()),
        "min_revenue_gain": float(gaps.min()),
        "max_revenue_gain": float(gaps.max()),
        "frac_strict_gain": float((gaps > 1e-12).mean()),
        "mutation_A_miscalibrated_violations": int(mut_a_viol),
        "mutation_B_concave_min_aggregator_violations": int(mut_b_viol),
        "verdict": "verified" if viol == 0 and mut_a_viol > 0 and mut_b_viol > 0 else "inconclusive",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
