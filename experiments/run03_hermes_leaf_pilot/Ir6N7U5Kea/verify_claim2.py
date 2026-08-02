"""verify_claim2.py -- Corollary 4.10 (MSNN/SNN expected-subgroup ratio).

Claim: E[K_MSNN(d)]/E[K_SNN(d)] = (1+o(1)) [ sum_{d'} (p_d'/p_d)^{r+1} ]^c   (Cor 4.10),
backed by Thm 4.8:
    E[K_SNN(d)]  ~ C(m-1,r) C(n-1,c) p_d^{rc+r+c}
    E[K_MSNN(d)] ~ C(m-1,r) C(n-1,c) gamma^c p_d^r p_max^{(r+1)c},
    gamma = sum_d' (p_d'/p_max)^{r+1}

Method (exact rational arithmetic, no seeds needed):
  For one candidate anchor tuple (AR of size r, AC of size c) for target entry (i,j) at
  level d, feasibility depends only on the treatments of the (r+1)x(c+1) block minus the
  (i,j) cell -- rc + r + c cells.  We enumerate all (|L|+1)^(rc+c) configurations of the
  anchor block and the q-row, and separately all (|L|+1)^r configurations of the target
  column, accumulating exact probabilities of
    SNN-feasible  : D_ab = d for all a in AR,b in AC ; D_aj = d ; D_ib = d
    MSNN-feasible : D_aj = d for all a in AR ; for each b in AC, D_ib = d(b) != 0 and
                    D_ab = d(b) for all a in AR      (Alg 3: B_ab = 1{D_ab=D_ib, D_aj=d})
  and compare with the closed forms.  Expected counts over the whole matrix follow by
  linearity: E[#feasible tuples] = C(m-1,r)C(n-1,c) * P; a Monte-Carlo count on a small
  matrix confirms that step.

Mutation test: restrict the mixed construction to a SINGLE treatment level for every
column (removing the "mixed" ingredient).  The MSNN probability must then collapse onto
the SNN probability, i.e. mutation_single_level_ratio == 1.
"""
import itertools
import json
import math
from fractions import Fraction

import numpy as np

CMD = ".venv/bin/python verify_claim2.py"


def exact_probs(p, d_idx, r, c):
    """Exhaustive enumeration. p[0] = P(unobserved), p[1:] = P(level).
    Returns (P_snn, P_msnn, P_msnn_restricted_to_single_level)."""
    L = len(p) - 1
    P = [Fraction(x).limit_denominator(10 ** 9) for x in p]
    d = d_idx
    # (a) exhaustive enumeration of the r cells of the target column D_aj (a in AR):
    #     both algorithms require all == d.  Confirms surviving mass == p_d^r.
    p_xcol = Fraction(0)
    for cfg in itertools.product(range(L + 1), repeat=r):
        if all(t == d for t in cfg):
            w = Fraction(1)
            for t in cfg:
                w *= P[t]
            p_xcol += w
    assert abs(float(p_xcol) - p[d] ** r) < 1e-15
    # (b) exhaustive enumeration of the remaining rc + c cells (anchor block + q row).
    p_snn = Fraction(0)
    p_msnn = Fraction(0)
    p_mut = Fraction(0)
    for cfg in itertools.product(range(L + 1), repeat=r * c + c):
        block = cfg[: r * c]   # D_ab, row-major over AR x AC
        qrow = cfg[r * c:]     # D_ib, b in AC
        w = p_xcol
        for t in cfg:
            w *= P[t]
        if w == 0:
            continue
        if all(t == d for t in block) and all(t == d for t in qrow):
            p_snn += w
        ok = all(qrow[b] != 0 for b in range(c)) and all(
            block[a * c + b] == qrow[b] for a in range(r) for b in range(c)
        )
        if ok:
            p_msnn += w
            if all(qrow[b] == d for b in range(c)):
                p_mut += w
    return p_snn, p_msnn, p_mut


def mc_count(p, d_idx, r, c, m, n, seeds):
    """Monte-Carlo: mean number of feasible (AR,AC) tuples on an m x n matrix,
    counted exhaustively over all C(m-1,r)C(n-1,c) tuples, target entry (0,0)."""
    L = len(p) - 1
    snn_counts = []
    msnn_counts = []
    rows = list(range(1, m))
    cols = list(range(1, n))
    for s in seeds:
        rng = np.random.default_rng(s)
        D = rng.choice(np.arange(L + 1), size=(m, n), p=p)
        ksnn = kmsnn = 0
        for AR in itertools.combinations(rows, r):
            if not all(D[a, 0] == d_idx for a in AR):
                continue
            for AC in itertools.combinations(cols, c):
                sub = D[np.ix_(AR, AC)]
                q = D[0, list(AC)]
                if np.all(sub == d_idx) and np.all(q == d_idx):
                    ksnn += 1
                if np.all(q != 0) and np.all(sub == q[None, :]):
                    kmsnn += 1
        snn_counts.append(ksnn)
        msnn_counts.append(kmsnn)
    a = np.array(snn_counts, float); b = np.array(msnn_counts, float)
    return (a.mean(), a.std(ddof=1) / np.sqrt(len(a)),
            b.mean(), b.std(ddof=1) / np.sqrt(len(b)))


def main():
    out = {"command": CMD, "claim": "Corollary 4.10 (Section 4.3)", "cases": []}
    settings = [
        # (name, p = [p0, p_low, p_med, p_high, p_vhigh], target level index, r, c)
        ("paper_MCAR_low_r1c1", [0.115, 0.01, 0.025, 0.05, 0.8], 1, 1, 1),
        ("paper_MCAR_low_r2c1", [0.115, 0.01, 0.025, 0.05, 0.8], 1, 2, 1),
        ("paper_MCAR_low_r1c2", [0.115, 0.01, 0.025, 0.05, 0.8], 1, 1, 2),
        ("paper_MCAR_med_r2c2", [0.115, 0.01, 0.025, 0.05, 0.8], 2, 2, 2),
        ("uniform3_r2c2", [0.4, 0.2, 0.2, 0.2], 1, 2, 2),
        ("skew3_r3c2", [0.2, 0.05, 0.15, 0.6], 1, 3, 2),
    ]
    for name, p, d, r, c in settings:
        assert abs(sum(p) - 1) < 1e-12
        p_snn, p_msnn, p_mut = exact_probs(p, d, r, c)
        pd = p[d]
        pmax = max(p[1:])
        gamma = sum((pk / pmax) ** (r + 1) for pk in p[1:])
        f_snn = pd ** (r * c + r + c)
        f_msnn = gamma ** c * pd ** r * pmax ** ((r + 1) * c)
        ratio_emp = float(p_msnn) / float(p_snn)
        ratio_cor410 = sum((pk / pd) ** (r + 1) for pk in p[1:]) ** c
        case = {
            "name": name, "p": p, "d_index": d, "r": r, "c": c,
            "n_configs_enumerated": len(p) ** (r * c + c) + len(p) ** r,
            "P_snn_exact": float(p_snn), "P_snn_thm48": f_snn,
            "P_msnn_exact": float(p_msnn), "P_msnn_thm48": f_msnn,
            "rel_err_snn": abs(float(p_snn) - f_snn) / f_snn,
            "rel_err_msnn": abs(float(p_msnn) - f_msnn) / f_msnn,
            "ratio_exact": ratio_emp, "ratio_corollary_4_10": ratio_cor410,
            "rel_err_ratio": abs(ratio_emp - ratio_cor410) / ratio_cor410,
            "mutation_single_level_ratio": float(p_mut) / float(p_snn),
        }
        out["cases"].append(case)
        print(f"{name}: P_snn={float(p_snn):.6e} (thm {f_snn:.6e}) "
              f"P_msnn={float(p_msnn):.6e} (thm {f_msnn:.6e}) "
              f"ratio={ratio_emp:.6f} vs Cor4.10 {ratio_cor410:.6f} "
              f"mutation_ratio={case['mutation_single_level_ratio']:.6f}")

    p = [0.4, 0.2, 0.2, 0.2]
    r, c, m, n, d = 2, 1, 9, 9, 1
    seeds = list(range(4000))
    mc_snn, se_snn, mc_msnn, se_msnn = mc_count(p, d, r, c, m, n, seeds)
    p_snn, p_msnn, _ = exact_probs(p, d, r, c)
    comb = math.comb(m - 1, r) * math.comb(n - 1, c)
    out["monte_carlo"] = {
        "p": p, "r": r, "c": c, "m": m, "n": n, "d_index": d, "n_seeds": len(seeds),
        "mean_feasible_snn_tuples": mc_snn, "se_snn": se_snn, "pred_snn": comb * float(p_snn),
        "mean_feasible_msnn_tuples": mc_msnn, "se_msnn": se_msnn, "pred_msnn": comb * float(p_msnn),
    }
    print("MC:", out["monte_carlo"])
    with open("results/claim2.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
