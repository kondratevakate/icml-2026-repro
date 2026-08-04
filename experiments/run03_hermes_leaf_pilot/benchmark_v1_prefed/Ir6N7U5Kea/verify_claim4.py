"""verify_claim4.py -- Table 1 (MCAR simulation study, Section 5.1 / Appendix B).

Paper setting: m=300, n=100, r=3, sigma_rel=1e-3, K=1, 10 repetitions,
L = {low, medium, high, very high} with f = 1, 5, 25, 625 and
p_MCAR = (0.115 unobserved, 0.01 low, 0.025 medium, 0.05 high, 0.8 very high).
Reported: FR = #feasible / (m n)  and  MRE = mean |(Ahat - A)/A| over feasible entries,
for target levels low / medium / high, for SNN (Alg 1) and MSNN (Alg 2 + Alg 3).

Deviations from the paper (documented, unavoidable):
  * `maxBiclique` in Algorithm 3 is not specified in the paper (it notes exact search is
    intractable).  We use the documented greedy column-growing heuristic in msnn.py.
  * The candidate rows for a target (i,j,d) are exactly {a != i : D_aj = d} and the
    candidate columns {b != j : D_ib != 0} -- this is an exact restriction of B, not an
    approximation.
  * FR / MRE are estimated on a random SUBSAMPLE of `n_targets` entries per repetition
    (uniformly over the m*n grid) instead of all 30000 entries, for runtime reasons.
    FR is therefore a binomial estimate of the paper's quantity; its Monte-Carlo standard
    error is reported.

Mutation test: `--mutation` restricts MSNN's mixed anchor columns to the target level
only (removing the cross-treatment integration).  MSNN must then collapse to SNN's
feasible rate.
"""
import argparse
import json
import time

import numpy as np

from msnn import F_SCALES, P_MCAR, estimate_from_anchors, feasible, greedy_biclique, make_ground_truth

CMD = ".venv/bin/python verify_claim4.py --reps 10 --targets 1500"
SIGMA_REL = 1e-3
R_LATENT = 3
MAX_ROWS, MAX_COLS = 10, 10          # paper: search is unaffordable beyond ~10


def anchors_for(D, i, j, d, mixed, rng, single_level=False):
    """Algorithm 3 (mixed=True) or its SNN restriction (mixed=False).
    Returns (MAR, MAC, dcols) or None."""
    m, n = D.shape
    rows = np.flatnonzero(D[:, j] == d)
    rows = rows[rows != i]
    if rows.size == 0:
        return None
    Di = D[i, :]
    if mixed and not single_level:
        colmask = Di != 0
    else:
        colmask = Di == d
    colmask[j] = False
    cols = np.flatnonzero(colmask)
    if cols.size == 0:
        return None
    B = D[np.ix_(rows, cols)] == Di[cols][None, :]
    MARi, MACi = greedy_biclique(B, max_rows=MAX_ROWS, max_cols=MAX_COLS, rng=rng)
    if MARi.size == 0 or MACi.size == 0:
        return None
    MAR, MAC = rows[MARi], cols[MACi]
    return MAR, MAC, Di[MAC]


RTOLS = (1e-3, 1e-2, 1e-1)


def run_rep(seed, d, mixed, n_targets, m=300, n=100, single_level=False):
    rng = np.random.default_rng(seed)
    A, U, V, lam = make_ground_truth(m, n, R_LATENT, F_SCALES, rng)
    D = rng.choice(np.arange(len(F_SCALES) + 1), size=(m, n), p=P_MCAR)
    Y = np.full((m, n), np.nan)
    for dd in range(1, len(F_SCALES) + 1):
        mask = D == dd
        Y[mask] = A[dd - 1][mask] + rng.normal(scale=SIGMA_REL * F_SCALES[dd - 1],
                                               size=int(mask.sum()))
    idx = rng.choice(m * n, size=n_targets, replace=False)
    nfeas = {rt: 0 for rt in RTOLS}
    rel_errs = {rt: [] for rt in RTOLS}
    sizes = []
    for t in idx:
        i, j = divmod(int(t), n)
        got = anchors_for(D, i, j, d, mixed, rng, single_level)
        if got is None:
            continue
        MAR, MAC, dcols = got
        if MAR.size * MAC.size <= 1 or (MAR.size == 1 and MAC.size == 1):
            continue
        w = 1.0 / F_SCALES[dcols - 1]
        S = Y[np.ix_(MAR, MAC)] * w[None, :]
        q = Y[i, MAC] * w
        x = Y[MAR, j]
        if not np.all(np.isfinite(S)) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(x)):
            continue
        passes = [rt for rt in RTOLS if feasible(S, q, x, rtol=rt)]
        if not passes:
            continue
        tau = np.linalg.svd(S, compute_uv=False)
        lam_rank = min(int(np.sum(tau > 1e-2 * tau[0])), R_LATENT)
        ah, _, _ = estimate_from_anchors(S, q, x, lam_rank=max(lam_rank, 1))
        truth = A[d - 1][i, j]
        sizes.append((int(MAR.size), int(MAC.size)))
        for rt in passes:
            nfeas[rt] += 1
            rel_errs[rt].append(abs((ah - truth) / truth))
    fr = {rt: nfeas[rt] / n_targets for rt in RTOLS}
    mre = {rt: (float(np.mean(rel_errs[rt])) if rel_errs[rt] else None) for rt in RTOLS}
    return fr, mre, nfeas, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--targets", type=int, default=1500)
    ap.add_argument("--out", default="results/claim4.json")
    args = ap.parse_args()

    paper = {  # Table 1
        "SNN": {1: (0.03, 0.806), 2: (1.20, 0.577), 3: (11.34, 0.515)},
        "MSNN": {1: (4.69, 3.91e-2), 2: (63.73, 1.18e-3), 3: (99.29, 7.05e-4)},
    }
    levels = {1: "low(p=0.01)", 2: "medium(p=0.025)", 3: "high(p=0.05)"}
    out = {"command": CMD, "claim": "Table 1, MCAR simulation (Section 5.1)",
           "config": {"m": 300, "n": 100, "r": R_LATENT, "sigma_rel": SIGMA_REL, "K": 1,
                      "reps": args.reps, "targets_per_rep": args.targets,
                      "p_MCAR": P_MCAR.tolist(), "f_scales": F_SCALES.tolist(),
                      "max_anchor_rows": MAX_ROWS, "max_anchor_cols": MAX_COLS},
           "paper_table1": {k: {levels[d]: v for d, v in dv.items()} for k, dv in paper.items()},
           "results": {}}
    t0 = time.time()
    for algo, mixed in (("SNN", False), ("MSNN", True), ("MSNN_MUTATION_single_level", True)):
        single = algo.endswith("single_level")
        for d, lname in levels.items():
            frs = {rt: [] for rt in RTOLS}
            mres = {rt: [] for rt in RTOLS}
            sz = []
            for rep in range(args.reps):
                fr, mre, nf, sizes = run_rep(1000 * d + rep, d, mixed, args.targets,
                                             single_level=single)
                for rt in RTOLS:
                    frs[rt].append(fr[rt] * 100)
                    if mre[rt] is not None:
                        mres[rt].append(mre[rt])
                sz.extend(sizes)
            rec = {"by_feasibility_tolerance": {
                       str(rt): {"FR_percent_mean": float(np.mean(frs[rt])),
                                 "FR_percent_sd": float(np.std(frs[rt], ddof=1)),
                                 "MRE_mean": float(np.mean(mres[rt])) if mres[rt] else None,
                                 "MRE_sd": float(np.std(mres[rt], ddof=1)) if len(mres[rt]) > 1 else None,
                                 "n_reps_with_feasible": len(mres[rt])}
                       for rt in RTOLS},
                   "mean_MAR": float(np.mean([a for a, b in sz])) if sz else None,
                   "mean_MAC": float(np.mean([b for a, b in sz])) if sz else None,
                   "paper_FR_percent": paper.get(algo, {}).get(d, (None, None))[0],
                   "paper_MRE": paper.get(algo, {}).get(d, (None, None))[1]}
            out["results"].setdefault(algo, {})[lname] = rec
            print(algo, lname, json.dumps(rec), flush=True)
    out["elapsed_sec"] = time.time() - t0
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
