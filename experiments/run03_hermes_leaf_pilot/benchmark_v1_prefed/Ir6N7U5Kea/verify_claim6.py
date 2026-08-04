"""verify_claim6.py -- Section 3.2-3.3 / Algorithms 2-3: mixed anchor rows/columns are
built as bipartite cliques of B = [1{D_ab = D_ib, D_aj = d}] that SPAN several treatment
levels while the target column x(d) stays at the target treatment level d, and the
construction is valid only because of Assumption 2.5 (shared latent row factors).

Tests (MCAR design of Appendix B, many seeds):
  T1 structure: for every anchor set returned by Algorithm 3, check
      (a) D_aj = d for all a in MAR            (target row's same-treatment data preserved)
      (b) D_ab = D_ib for all a in MAR, b in MAC   (column treatment matches the q entry)
      (c) fraction of anchor sets whose MAC spans >= 2 distinct treatment levels ("mixed"),
          against the SNN construction, which is single-level by definition.
  T2 mechanism: estimate A_ij^{(d)} with those mixed anchors under
      (i)  Assumption 2.5 holding  (shared row factors U, per-level column factors), and
      (ii) MUTATION: Assumption 2.5 violated (row factors drawn independently per level).
      Under (i) the relative error must be small; under (ii) it must blow up.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from msnn import (F_SCALES, P_MCAR, build_B, build_B_snn, estimate_from_anchors,  # noqa: E402
                  feasible, greedy_biclique, make_ground_truth)

CMD = ".venv/bin/python verify_claim6.py"
SIGMA_REL = 1e-3
R_LATENT = 3


def make_ground_truth_violating(m, n, r, f_scales, rng):
    """Same generator but with INDEPENDENT row factors per treatment level (breaks A2.5)."""
    A = np.empty((len(f_scales), m, n))
    V = rng.normal(size=(n, r))
    for d in range(len(f_scales)):
        Ud = rng.normal(size=(m, r))
        M = Ud @ V.T
        A[d] = M / np.abs(M).max() * f_scales[d]
    return A


def run(seed, m=80, n=80, d=1, max_rows=8, max_cols=6, p_vec=None):
    rng = np.random.default_rng(seed)
    A, U, V, lam = make_ground_truth(m, n, R_LATENT, F_SCALES, rng)
    Abad = make_ground_truth_violating(m, n, R_LATENT, F_SCALES,
                                       np.random.default_rng(seed + 777))
    p_vec = P_MCAR if p_vec is None else np.asarray(p_vec)
    D = rng.choice(np.arange(len(F_SCALES) + 1), size=(m, n), p=p_vec)
    recs = []
    targets = [(i, j) for i in range(0, m, 7) for j in range(0, n, 7)]
    for (i, j) in targets:
        B = build_B(D, i, j, d)
        MAR, MAC = greedy_biclique(B, max_rows=max_rows, max_cols=max_cols,
                                   rng=np.random.default_rng(seed))
        if MAR.size < 2 or MAC.size < 2:
            continue
        dc0 = D[i, MAC]
        w0 = 1.0 / F_SCALES[dc0 - 1]
        S_chk = np.array([[A[db - 1][a, b] for b, db in zip(MAC, dc0)] for a in MAR]) * w0[None, :]
        q_chk = np.array([A[db - 1][i, b] for b, db in zip(MAC, dc0)]) * w0
        x_chk = A[d - 1][MAR, j]
        if not feasible(S_chk, q_chk, x_chk, rtol=1e-6):   # Section 5.1 feasibility filter
            continue
        dcols = D[i, MAC]
        struct_a = bool(np.all(D[MAR, j] == d))
        struct_b = bool(np.all(D[np.ix_(MAR, MAC)] == dcols[None, :]))
        mixed = int(len(set(dcols.tolist())))
        # SNN construction for the same target
        Bs = build_B_snn(D, i, j, d)
        MARs, MACs = greedy_biclique(Bs, max_rows=max_rows, max_cols=max_cols,
                                     rng=np.random.default_rng(seed))
        snn_levels = int(len(set(D[i, MACs].tolist()))) if MACs.size else 0

        def est(Asrc):
            w = 1.0 / F_SCALES[dcols - 1]
            S = np.array([[Asrc[db - 1][a, b] for b, db in zip(MAC, dcols)] for a in MAR])
            q = np.array([Asrc[db - 1][i, b] for b, db in zip(MAC, dcols)])
            S = S + rng.normal(size=S.shape) * (SIGMA_REL * F_SCALES[dcols - 1])[None, :]
            q = q + rng.normal(size=q.shape) * SIGMA_REL * F_SCALES[dcols - 1]
            x = Asrc[d - 1][MAR, j] + rng.normal(size=MAR.size) * SIGMA_REL * F_SCALES[d - 1]
            ah, _, _ = estimate_from_anchors(S * w[None, :], q * w, x)
            truth = Asrc[d - 1][i, j]
            return abs((ah - truth) / truth)

        recs.append({"i": int(i), "j": int(j), "n_MAR": int(MAR.size), "n_MAC": int(MAC.size),
                     "struct_target_col_all_d": struct_a, "struct_col_treatment_match": struct_b,
                     "n_levels_in_MAC": mixed, "n_levels_in_SNN_MAC": snn_levels,
                     "snn_MAC_size": int(MACs.size), "snn_MAR_size": int(MARs.size),
                     "rel_err_A25_holds": est(A), "rel_err_A25_violated": est(Abad)})
    return recs


def main():
    seeds = list(range(25))
    recs = []
    for s in seeds:
        recs.extend(run(s))
    # balanced MCAR: shows the construction can span several levels when no level dominates
    uni = [0.2, 0.2, 0.2, 0.2, 0.2]
    recs_uni = []
    for s in seeds:
        recs_uni.extend(run(s, p_vec=uni))
    arr_ok = np.array([r["rel_err_A25_holds"] for r in recs])
    arr_bad = np.array([r["rel_err_A25_violated"] for r in recs])
    out = {
        "command": CMD,
        "claim": "Mixed anchors via bipartite cliques, Assumption 2.5 (Sections 3.2-3.3, Alg 2-3)",
        "n_seeds": len(seeds), "n_anchor_sets": len(recs),
        "frac_struct_target_col_all_d": float(np.mean([r["struct_target_col_all_d"] for r in recs])),
        "frac_struct_col_treatment_match": float(np.mean([r["struct_col_treatment_match"] for r in recs])),
        "frac_MAC_spanning_multiple_levels": float(np.mean([r["n_levels_in_MAC"] >= 2 for r in recs])),
        "mean_levels_in_MAC": float(np.mean([r["n_levels_in_MAC"] for r in recs])),
        "mean_levels_in_SNN_MAC": float(np.mean([r["n_levels_in_SNN_MAC"] for r in recs])),
        "mean_MAC_size": float(np.mean([r["n_MAC"] for r in recs])),
        "mean_SNN_MAC_size": float(np.mean([r["snn_MAC_size"] for r in recs])),
        "median_rel_err_A25_holds": float(np.median(arr_ok)),
        "median_rel_err_A25_violated_MUTATION": float(np.median(arr_bad)),
        "mean_rel_err_A25_holds": float(np.mean(arr_ok)),
        "mean_rel_err_A25_violated_MUTATION": float(np.mean(arr_bad)),
        "frac_runs_mutation_worse": float(np.mean(arr_bad > arr_ok)),
        "balanced_MCAR_p": uni,
        "balanced_n_anchor_sets": len(recs_uni),
        "balanced_frac_MAC_spanning_multiple_levels":
            float(np.mean([r["n_levels_in_MAC"] >= 2 for r in recs_uni])),
        "balanced_mean_levels_in_MAC": float(np.mean([r["n_levels_in_MAC"] for r in recs_uni])),
        "balanced_frac_struct_col_treatment_match":
            float(np.mean([r["struct_col_treatment_match"] for r in recs_uni])),
        "balanced_median_rel_err_A25_holds":
            float(np.median([r["rel_err_A25_holds"] for r in recs_uni])),
        "balanced_median_rel_err_A25_violated_MUTATION":
            float(np.median([r["rel_err_A25_violated"] for r in recs_uni])),
        "records_head": recs[:20],
    }
    print(json.dumps({k: v for k, v in out.items() if k != "records_head"}, indent=2))
    with open("results/claim6.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
