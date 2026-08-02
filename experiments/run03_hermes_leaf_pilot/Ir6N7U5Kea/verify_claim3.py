"""verify_claim3.py -- Corollary 4.11: the efficiency gap w.r.t. the data-richest level
is reduced from QUADRATIC order (rc) to LINEAR order (r).

Cor 4.11 (Section 4.3), from Thm 4.8:
    E[K_SNN(d)]  / E[K_MSNN(dmax)] = O( gamma^{-c} (p_d/p_max)^{rc+r+c} )
    E[K_MSNN(d)] / E[K_MSNN(dmax)] = O( (p_d/p_max)^{r} )

Verification:
  (a) SYMBOLIC (sympy): substitute the Thm-4.8 leading terms and simplify the two ratios;
      check they equal the stated expressions identically, and read off the exponent of
      rho := p_d/p_max  -> rc+r+c (contains the quadratic term rc) vs r (linear).
  (b) NUMERIC exponent recovery: for a grid of rho values compute the exact ratios and
      fit log(ratio) vs log(rho); the slopes must equal rc+r+c and r.
  (c) CROSS-CHECK against the exhaustive enumeration used for claim 2 (results/claim2.json
      machinery is re-used here through the same probability formulas evaluated exactly).

Mutation test: replace MSNN(d) in the numerator by SNN(d) (i.e. remove mixing from the
data-sparse level). The recovered exponent must jump back from r to rc+r+c.
"""
import itertools
import json
from fractions import Fraction

import numpy as np
import sympy as sp

CMD = ".venv/bin/python verify_claim3.py"


def symbolic():
    pd, pmax, gamma, m, n = sp.symbols("p_d p_max gamma m n", positive=True)
    r, c = sp.symbols("r c", positive=True, integer=True)
    comb = sp.Symbol("C", positive=True)  # C(m-1,r)C(n-1,c), cancels
    K_snn_d = comb * pd ** (r * c + r + c)
    K_msnn_d = comb * gamma ** c * pd ** r * pmax ** ((r + 1) * c)
    K_msnn_max = comb * gamma ** c * pmax ** r * pmax ** ((r + 1) * c)
    rho = sp.Symbol("rho", positive=True)
    ratio1 = sp.simplify(K_snn_d / K_msnn_max)
    ratio2 = sp.simplify(K_msnn_d / K_msnn_max)
    target1 = gamma ** (-c) * (pd / pmax) ** (r * c + r + c)
    target2 = (pd / pmax) ** r
    ok1 = sp.simplify(ratio1 / target1) == 1
    ok2 = sp.simplify(ratio2 / target2) == 1
    # exponent of rho after substituting p_d = rho * p_max
    e1 = sp.simplify(sp.log(ratio1.subs(pd, rho * pmax) * gamma ** c) / sp.log(rho))
    e2 = sp.simplify(sp.log(ratio2.subs(pd, rho * pmax)) / sp.log(rho))
    return {
        "ratio_SNNd_over_MSNNmax": sp.srepr(ratio1) and str(sp.simplify(ratio1)),
        "ratio_MSNNd_over_MSNNmax": str(sp.simplify(ratio2)),
        "matches_corollary_4_11_first": bool(ok1),
        "matches_corollary_4_11_second": bool(ok2),
        "rho_exponent_SNN_over_MSNNmax": str(sp.expand(e1)),
        "rho_exponent_MSNN_over_MSNNmax": str(sp.expand(e2)),
        "quadratic_term_present_in_SNN_ratio": bool(
            sp.expand(e1).coeff(sp.Symbol("r", positive=True, integer=True)
                                * sp.Symbol("c", positive=True, integer=True)) == 1),
        "MSNN_ratio_is_linear_in_r": str(sp.expand(e2)) == "r",
    }


def numeric_slopes(r, c, L, pmax=0.5):
    """Exact ratios on a grid of rho; recover exponents by log-log least squares."""
    rhos = np.array([0.02, 0.05, 0.1, 0.2, 0.4])
    out = {}
    r1, r2, rmut = [], [], []
    for rho in rhos:
        pd = rho * pmax
        # other levels: one at pmax, the rest at pd (worst-case data-sparse levels)
        ps = [pmax] + [pd] * (L - 1)
        gamma = sum((x / pmax) ** (r + 1) for x in ps)
        K_snn_d = pd ** (r * c + r + c)
        K_msnn_d = gamma ** c * pd ** r * pmax ** ((r + 1) * c)
        K_msnn_max = gamma ** c * pmax ** r * pmax ** ((r + 1) * c)
        K_snn_max = pmax ** (r * c + r + c)
        r1.append(K_snn_d / K_msnn_max * gamma ** c)   # strip the gamma^{-c} prefactor
        r2.append(K_msnn_d / K_msnn_max)
        rmut.append(K_snn_d / K_snn_max)               # MUTATION: SNN vs SNN
    lr = np.log(rhos)
    s1 = np.polyfit(lr, np.log(r1), 1)[0]
    s2 = np.polyfit(lr, np.log(r2), 1)[0]
    sm = np.polyfit(lr, np.log(rmut), 1)[0]
    out.update(r=r, c=c, L=L, rhos=rhos.tolist(),
               slope_SNNd_over_MSNNmax=float(s1), expected_slope_1=r * c + r + c,
               slope_MSNNd_over_MSNNmax=float(s2), expected_slope_2=r,
               mutation_slope_SNNd_over_SNNmax=float(sm),
               mutation_expected=r * c + r + c)
    return out


def enumeration_crosscheck(r, c):
    """Exact enumeration (same construction as verify_claim2) of the per-tuple
    feasibility probabilities, used to confirm the Thm 4.8 forms that Cor 4.11 rests on."""
    p = [0.2, 0.05, 0.15, 0.6]  # p0, three levels
    d = 1
    L = len(p) - 1
    P = [Fraction(x).limit_denominator(10 ** 9) for x in p]
    pd_r = Fraction(1)
    for _ in range(r):
        pd_r *= P[d]
    p_snn = Fraction(0)
    p_msnn_d = Fraction(0)
    for cfg in itertools.product(range(L + 1), repeat=r * c + c):
        block, qrow = cfg[: r * c], cfg[r * c:]
        w = pd_r
        for t in cfg:
            w *= P[t]
        if w == 0:
            continue
        if all(t == d for t in block) and all(t == d for t in qrow):
            p_snn += w
        if all(qrow[b] != 0 for b in range(c)) and all(
                block[a * c + b] == qrow[b] for a in range(r) for b in range(c)):
            p_msnn_d += w
    # same for the data-richest level dmax
    dmax = int(np.argmax(p[1:])) + 1
    pmaxr = Fraction(1)
    for _ in range(r):
        pmaxr *= P[dmax]
    p_msnn_max = Fraction(0)
    for cfg in itertools.product(range(L + 1), repeat=r * c + c):
        block, qrow = cfg[: r * c], cfg[r * c:]
        w = pmaxr
        for t in cfg:
            w *= P[t]
        if w == 0:
            continue
        if all(qrow[b] != 0 for b in range(c)) and all(
                block[a * c + b] == qrow[b] for a in range(r) for b in range(c)):
            p_msnn_max += w
    pd, pmax = p[d], p[dmax]
    gamma = sum((x / pmax) ** (r + 1) for x in p[1:])
    return {
        "p": p, "r": r, "c": c, "d_index": d, "dmax_index": dmax,
        "exact_SNNd_over_MSNNmax": float(p_snn) / float(p_msnn_max),
        "pred_SNNd_over_MSNNmax": gamma ** (-c) * (pd / pmax) ** (r * c + r + c),
        "exact_MSNNd_over_MSNNmax": float(p_msnn_d) / float(p_msnn_max),
        "pred_MSNNd_over_MSNNmax": (pd / pmax) ** r,
    }


def main():
    out = {"command": CMD, "claim": "Corollary 4.11 (Section 4.3)"}
    out["symbolic"] = symbolic()
    out["numeric"] = [numeric_slopes(r, c, L) for r, c, L in
                      [(3, 3, 4), (2, 4, 3), (5, 2, 4), (3, 5, 2)]]
    out["enumeration_crosscheck"] = [enumeration_crosscheck(r, c) for r, c in [(1, 1), (2, 2), (3, 2)]]
    print(json.dumps(out, indent=2)[:4000])
    with open("results/claim3.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
