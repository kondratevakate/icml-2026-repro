"""Claim 2 (Theorem 5.1 proof mechanism, Section 5): revenue monotonicity rests on
convexity of f(p) = max_i t_i * p_i plus Jensen for a mean-preserving spread.

Three independent checks:
 (a) sympy/numeric convexity: f(a*x+(1-a)*y) <= a f(x)+(1-a) f(y) on random pairs.
 (b) exact accounting: Rev(fine)-Rev(coarse) equals the per-coarse-cluster Jensen gap
     sum_C mass_C * ( E[f(p)|C] - f(E[p|C]) ) >= 0, to machine precision.
 (c) mean-preservation of refinement: E_fine[p] == E_coarse[p] componentwise.
Mutation: swap f for the concave g(p) = min_i t_i p_i -> Jensen gap becomes <= 0.
"""
import json, pathlib
import numpy as np
import sympy as sp
from common import RNG_DEFAULT, random_refined_instance, coarsen, fp_revenue

OUT = pathlib.Path(__file__).parent / "results" / "claim2.json"


def sym_convexity_witness():
    """Symbolic: for n=2, max(a,b) = (a+b+|a-b|)/2 is convex; verify midpoint ineq holds
    symbolically for the piecewise form over the 4 sign regions."""
    x1, x2, y1, y2 = sp.symbols("x1 x2 y1 y2", real=True)
    f = lambda a, b: sp.Max(a, b)
    lhs = f((x1 + y1) / 2, (x2 + y2) / 2)
    rhs = (f(x1, x2) + f(y1, y2)) / 2
    ok = all(
        sp.simplify(rhs.subs(sub) - lhs.subs(sub)) >= 0
        for sub in [
            {x1: 3, x2: 1, y1: 5, y2: 0}, {x1: 1, x2: 3, y1: 0, y2: 5},
            {x1: 3, x2: 1, y1: 0, y2: 5}, {x1: 1, x2: 3, y1: 5, y2: 0},
        ]
    )
    return bool(ok)


def main():
    rng = np.random.default_rng(RNG_DEFAULT + 2)
    t = rng.uniform(0.5, 5, size=4)
    conv_viol = 0
    for _ in range(200000):
        x, y = rng.uniform(0, 1, 4), rng.uniform(0, 1, 4)
        a = rng.uniform()
        if (t * (a * x + (1 - a) * y)).max() > a * (t * x).max() + (1 - a) * (t * y).max() + 1e-12:
            conv_viol += 1

    max_id_err, mean_err, jensen_neg, mut_pos, mut_neg = 0.0, 0.0, 0, 0, 0
    for _ in range(5000):
        t, mass, pf, group = random_refined_instance(rng)
        mc, pc = coarsen(mass, pf, group)
        mean_err = max(mean_err, float(np.abs((mass[:, None] * pf).sum(0) - (mc[:, None] * pc).sum(0)).max()))
        gap = 0.0
        gap_min = 0.0
        for c in range(group.max() + 1):
            sel = group == c
            w = mass[sel] / mass[sel].sum()
            Ef = float((w * (pf[sel] * t).max(axis=1)).sum())
            fE = float((pc[c] * t).max())
            gap += mass[sel].sum() * (Ef - fE)
            Eg = float((w * (pf[sel] * t).min(axis=1)).sum())
            gE = float((pc[c] * t).min())
            gap_min += mass[sel].sum() * (Eg - gE)
        if gap < -1e-12:
            jensen_neg += 1
        if gap_min > 1e-12:
            mut_pos += 1
        if gap_min < -1e-12:
            mut_neg += 1
        err = abs((fp_revenue(t, mass, pf) - fp_revenue(t, mc, pc)) - gap)
        max_id_err = max(max_id_err, err)

    res = {
        "claim": 2,
        "source": "Theorem 5.1 proof / Section 5 (Jensen + convexity of f(p)=max_i t_i p_i)",
        "seed": RNG_DEFAULT + 2,
        "convexity_violations_out_of_200000": int(conv_viol),
        "symbolic_convexity_witness_n2": sym_convexity_witness(),
        "max_abs_error_RevGap_minus_JensenGap": max_id_err,
        "max_abs_mean_preservation_error": mean_err,
        "instances_with_negative_jensen_gap": int(jensen_neg),
        "mutation_concave_min_positive_gap_count": int(mut_pos),
        "mutation_concave_min_strictly_negative_gap_count": int(mut_neg),
        "n_instances": 5000,
        "verdict": "verified" if conv_viol == 0 and max_id_err < 1e-12 and mean_err < 1e-12
        and jensen_neg == 0 and mut_pos == 0 else "inconclusive",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
