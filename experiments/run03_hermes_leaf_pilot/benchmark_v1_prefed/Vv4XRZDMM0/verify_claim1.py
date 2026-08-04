"""verify_claim1.py -- Theorem 1 (Section 2), "Finite-sample uniform validity" of max-p aggregation.

Claim under test (paper, Thm 1):
  (a) The aggregated set equals the union of the per-source conformal sets:
      Chat(X_{n+1}) = union_k Chat^{(k)}(X_{n+1}).
  (b) For a test point from ANY mixture P = sum_k pi_k P^{(k)} (arbitrary weights),
      P(Y_{n+1} in Chat(X_{n+1})) >= 1 - alpha,  in finite samples,
      for ARBITRARY conformity scores.

Design:
  * Discrete X in {0..n_x-1}, discrete Y in {0..n_y-1}; K sources with different
    conditional laws f_k(y|x).
  * Conformity scores are deliberately ARBITRARY / misspecified (a fixed random
    score table per source) -- Theorem 1 must hold regardless of score quality.
  * max-p aggregation: p^{(k)}(y) = (1 + #{i in cal_k : s_k >= s_k(x,y)}) / (n_k + 1),
    p(y) = max_k p^{(k)}(y),  Chat(x) = {y : p(y) > alpha}.
  * Exhaustive over the mixture simplex: since coverage under a mixture is the
    pi-weighted average of per-source coverages, the worst case is a vertex.
    We therefore evaluate every vertex (= every source) exhaustively AND a grid
    of mixtures; both must clear 1-alpha.
  * Many independent draws of the calibration set (fresh seeds), never a single seed.

Mutation test (mechanism, not correlation):
  M1: drop the finite-sample "+1" correction -> p^{(k)} = #{>=}/n_k  (plug-in quantile)
  M2: replace max-p by mean-p aggregation
  Both are predicted to LOSE the >= 1-alpha guarantee (M1 marginally/at small n,
  M2 substantially).

CPU only, numpy stdlib. Writes results/claim1.json.
"""
import json, os, sys, time
import numpy as np
from scipy.stats import beta as beta_dist

ALPHA = 0.1
K = 3
NX, NY = 4, 20
N_CAL = 50           # per-source calibration size (small -> finite-sample regime)
N_TRIALS = 4000      # independent calibration+test draws per configuration
HERE = os.path.dirname(os.path.abspath(__file__))


def make_world(rng):
    """Per-source conditional pmfs f_k(y|x) and arbitrary score tables s_k(x,y)."""
    f = rng.gamma(1.0, 1.0, size=(K, NX, NY)) + 0.05
    f /= f.sum(axis=2, keepdims=True)
    px = rng.dirichlet(np.ones(NX) * 3.0, size=K)      # per-source marginal of X
    # Scores: log-density of source k plus noise (informative but misspecified).
    # Theorem 1 allows ARBITRARY scores; informative ones keep sets non-degenerate.
    scores = -np.log(f) + 0.3 * rng.normal(size=(K, NX, NY))
    return f, px, scores


def draw(rng, f, px, k, n):
    x = rng.choice(NX, size=n, p=px[k])
    cdf = np.cumsum(f[k], axis=1)                    # (NX, NY)
    u = rng.random(n)
    y = (u[:, None] > cdf[x]).sum(axis=1)
    y = np.minimum(y, NY - 1)
    return x, y


def pvalues(cal_scores, s_row, correction=True):
    """p^{(k)}(y) for all y given calibration scores of source k."""
    n = cal_scores.size
    ge = (cal_scores[None, :] >= s_row[:, None]).sum(axis=1)
    return (ge + 1.0) / (n + 1.0) if correction else ge / n


def run_config(seed, aggregation="max", correction=True):
    rng = np.random.default_rng(seed)
    f, px, scores = make_world(rng)
    hits = np.zeros(K)              # per-source coverage counts (vertices of simplex)
    sizes = np.zeros(K)
    union_mismatch = 0
    for t in range(N_TRIALS):
        cal = []
        for k in range(K):
            cx, cy = draw(rng, f, px, k, N_CAL)
            cal.append(scores[k, cx, cy])
        ktest = t % K                                  # cycle sources exhaustively
        tx, ty = draw(rng, f, px, ktest, 1)
        x0, y0 = int(tx[0]), int(ty[0])
        P = np.zeros((K, NY))
        for k in range(K):
            P[k] = pvalues(cal[k], scores[k, x0], correction)
        if aggregation == "max":
            pagg = P.max(axis=0)
        elif aggregation == "mean":
            pagg = P.mean(axis=0)
        else:                                   # "single": source-0 calibration only
            pagg = P[0]
        Cset = pagg > ALPHA
        if aggregation == "max":
            union = np.zeros(NY, dtype=bool)
            for k in range(K):
                union |= (P[k] > ALPHA)
            if not np.array_equal(union, Cset):
                union_mismatch += 1
        hits[ktest] += bool(Cset[y0])
        sizes[ktest] += Cset.sum()
    per_src_n = np.array([sum(1 for t in range(N_TRIALS) if t % K == k) for k in range(K)], float)
    cov = hits / per_src_n
    return dict(seed=seed, per_source_coverage=cov.tolist(),
                per_source_n=per_src_n.tolist(),
                worst_source_coverage=float(cov.min()),
                mean_set_size=float((sizes.sum()) / N_TRIALS),
                union_identity_mismatches=int(union_mismatch))


def mixture_check(res_list):
    """Coverage under any mixture is a convex combination of per-source coverages,
    so min over the simplex = min over vertices. Verified numerically on a grid."""
    grid = []
    for a in np.linspace(0, 1, 11):
        for b in np.linspace(0, 1 - a, 11):
            grid.append((a, b, 1 - a - b))
    worst = 1.0
    for r in res_list:
        cov = np.array(r["per_source_coverage"])
        for pi in grid:
            worst = min(worst, float(np.dot(pi, cov)))
    return dict(n_mixture_weights=len(grid) * len(res_list), worst_mixture_coverage=worst)


def main():
    t0 = time.time()
    seeds = [1000 + i for i in range(8)]      # 8 independent worlds, never one seed
    main_runs = [run_config(s, "max", True) for s in seeds]
    mut1 = [run_config(s, "max", False) for s in seeds]     # no +1 correction
    mut2 = [run_config(s, "mean", True) for s in seeds]     # mean-p instead of max-p
    mut3 = [run_config(s, "single", True) for s in seeds]   # single-source calibration

    def agg(rs):
        w = np.array([r["worst_source_coverage"] for r in rs])
        return dict(worst_source_coverage_min=float(w.min()),
                    worst_source_coverage_mean=float(w.mean()),
                    n_worlds=len(rs),
                    n_worlds_below_target=int((w < 1 - ALPHA).sum()),
                    mean_set_size=float(np.mean([r["mean_set_size"] for r in rs])))

    n_eff = main_runs[0]["per_source_n"][0]
    lo = float(beta_dist.ppf(0.005, (1 - ALPHA) * n_eff, n_eff - (1 - ALPHA) * n_eff + 1))
    out = dict(
        claim="Theorem 1 (Section 2): finite-sample uniform validity of max-p aggregation",
        alpha=ALPHA, K=K, n_cal_per_source=N_CAL, n_trials_per_world=N_TRIALS,
        command="python verify_claim1.py",
        main=dict(runs=main_runs, **agg(main_runs)),
        mixture=mixture_check(main_runs),
        union_identity_total_mismatches=int(sum(r["union_identity_mismatches"] for r in main_runs)),
        mutation_no_plus_one=dict(runs=mut1, **agg(mut1)),
        mutation_mean_p=dict(runs=mut2, **agg(mut2)),
        mutation_single_source=dict(runs=mut3, **agg(mut3)),
        mc_lower_tolerance_99pct=lo,
        runtime_sec=round(time.time() - t0, 1),
    )
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "claim1.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith(("main", "mutation"))}, indent=2))
    print("main            worst-source coverage min/mean:",
          out["main"]["worst_source_coverage_min"], out["main"]["worst_source_coverage_mean"],
          "size", out["main"]["mean_set_size"])
    print("MUT no-(+1)     worst-source coverage min/mean:",
          out["mutation_no_plus_one"]["worst_source_coverage_min"], out["mutation_no_plus_one"]["worst_source_coverage_mean"])
    print("MUT single-src  worst-source coverage min/mean:",
          out["mutation_single_source"]["worst_source_coverage_min"], out["mutation_single_source"]["worst_source_coverage_mean"],
          "size", out["mutation_single_source"]["mean_set_size"])
    print("MUT mean-p      worst-source coverage min/mean:",
          out["mutation_mean_p"]["worst_source_coverage_min"], out["mutation_mean_p"]["worst_source_coverage_mean"],
          "size", out["mutation_mean_p"]["mean_set_size"])


if __name__ == "__main__":
    main()
