"""verify_claim1.py -- Theorem 4.5 (finite-sample entry-wise bound) and Theorem 4.6
(asymptotic normality, error rate K_MSNN^{-1/2}), Section 4.2.

Setting (Assumption 2.5 tensor model, Appendix-B scales f = 1,5,25,625, sigma_rel = 1e-3):
for a target entry (i,j) at the data-sparse level d = "low", we build K_MSNN subgroups by
hand exactly as Algorithm 2 requires -- MAC shared, MAR partitioned into K disjoint groups,
every anchor column b carrying a (random) treatment level d(b) that matches the target
row's own treatment D_ib, and every target-column entry D_aj = d -- and run Algorithm 2
with weights w(b,d(b)) = 1/f(d(b)).

Tests (all over many seeds, no single-seed conclusions):
  T1 normality: the studentized statistic  K (Ahat - A) / sqrt(sum_k sigma~_k^2)
     must be standard normal (Kolmogorov-Smirnov test against N(0,1)).
  T2 rate: sd(Ahat - A) as a function of K must scale as K^{-1/2}
     (log-log slope ~ -0.5).
  T3 finite-sample bound (Thm 4.5): |Ahat - A| / f(d) must stay below the theorem's
     rate  (1/K){error + [sum_k ||beta~||_2^2]^{1/2}}  times a constant of order 1.

Mutations:
  M1 (mechanism): break the column-treatment matching required by Algorithm 2/3 -- the
     anchor block columns are taken at a treatment level different from the one of the
     corresponding q entry.  Because latent COLUMN factors differ per level, the estimator
     acquires a bias floor: the K^{-1/2} rate must be destroyed (slope -> ~0) and the
     studentized statistic must fail the KS test.
  M2 (weights): drop w = 1/f(d(b)) (use w = 1) under the strongly heterogeneous scales
     f = 1..625.  Reports the conditioning/error change (Remark 3.1).
"""
import json
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
from msnn import F_SCALES, estimate_from_anchors, make_ground_truth  # noqa: E402

CMD = ".venv/bin/python verify_claim1.py"
SIGMA_REL = 1e-3
R_LATENT = 3


def one_run(seed, K, R=6, C=8, n=200, d=1, mismatch=False, use_weights=True):
    """Returns (Ahat - A, studentized statistic, |Ahat-A|/f(d), thm45_rate)."""
    rng = np.random.default_rng(seed)
    m = K * R + 1
    A, U, V, lam = make_ground_truth(m, n, R_LATENT, F_SCALES, rng)
    L = len(F_SCALES)
    i, j = 0, 0
    # anchor columns: shared MAC, each with its own treatment level d(b)
    MAC = np.arange(1, C + 1)
    dcols = rng.integers(1, L + 1, size=C)                 # d(b) in {1..L}
    num, den2, b2sum, err_terms = 0.0, 0.0, 0.0, 0.0
    Ahat_parts = []
    for k in range(K):
        MAR = np.arange(1 + k * R, 1 + (k + 1) * R)
        w = np.array([1.0 / F_SCALES[db - 1] if use_weights else 1.0 for db in dcols])
        # noiseless (population) quantities
        S0 = np.array([[A[db - 1][a, b] for b, db in zip(MAC, dcols)] for a in MAR])
        q0 = np.array([A[db - 1][i, b] for b, db in zip(MAC, dcols)])
        if mismatch:  # MUTATION M1: block columns at a *different* level than q
            dblk = 1 + (dcols % L)
            S0 = np.array([[A[db - 1][a, b] for b, db in zip(MAC, dblk)] for a in MAR])
        S0w, q0w = S0 * w[None, :], q0 * w
        # noisy observations
        Sn = S0 + rng.normal(size=S0.shape) * (SIGMA_REL * F_SCALES[dcols - 1])[None, :]
        qn = q0 + rng.normal(size=q0.shape) * SIGMA_REL * F_SCALES[dcols - 1]
        x = A[d - 1][MAR, j] + rng.normal(size=R) * SIGMA_REL * F_SCALES[d - 1]
        Ahat_k, beta_hat, _ = estimate_from_anchors(Sn * w[None, :], qn * w, x)
        Ahat_parts.append(Ahat_k)
        # population beta~ = projection of beta onto Col(E[S_w]) -> least squares fit
        beta_t, *_ = np.linalg.lstsq(S0w.T, q0w, rcond=None)
        sig = SIGMA_REL * F_SCALES[d - 1]                  # sigma_lj^{(d)}
        den2 += np.sum((beta_t * sig) ** 2)
        b2sum += float(beta_t @ beta_t)
        rk = np.linalg.matrix_rank(S0w)
        e1 = np.sqrt(rk) / C ** 0.25
        e2 = rk ** 1.5 * np.abs(beta_t).sum() * np.sqrt(np.log(C * R)) / np.sqrt(min(C, R))
        err_terms += e1 + e2
    Ahat = float(np.mean(Ahat_parts))
    Atrue = A[d - 1][i, j]
    diff = Ahat - Atrue
    stat = K * diff / np.sqrt(den2)
    thm45 = (err_terms + np.sqrt(b2sum)) / K            # rate inside O_p(.), times f(d)
    return diff, stat, abs(diff) / F_SCALES[d - 1], thm45


def sweep(Ks, seeds, **kw):
    sds, stats_all, ratios = {}, [], {}
    shape_ks, infl = {}, {}
    for K in Ks:
        diffs, sts, rat = [], [], []
        for s in seeds:
            dfk, st, ad, t45 = one_run(s + 1000 * K, K, **kw)
            diffs.append(dfk)
            sts.append(st)
            rat.append(ad / t45)
        sds[K] = float(np.std(diffs, ddof=1))
        # T4: shape test -- standardize by the EMPIRICAL sd instead of the theoretical
        # sigma~ (Thm 4.6 condition (iv) is not attainable at these finite sizes),
        # and record the variance-inflation factor sd_emp / (theoretical sd).
        dz = (np.array(diffs) - np.mean(diffs)) / np.std(diffs, ddof=1)
        shape_ks[K] = float(stats.kstest(dz, "norm").pvalue)
        infl[K] = float(np.std(diffs, ddof=1) / (np.std(sts, ddof=1) ** 0 * 1.0)
                        ) if False else float(np.std(sts, ddof=1))
        ratios[K] = float(np.max(rat))
        stats_all.extend(sts)
    slope = float(np.polyfit(np.log(list(sds)), np.log(list(sds.values())), 1)[0])
    ks = stats.kstest(stats_all, "norm")
    return {"sd_by_K": {str(k): v for k, v in sds.items()},
            "loglog_slope": slope,
            "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
            "n_statistics": len(stats_all),
            "max_ratio_absdiff_over_thm45_rate": {str(k): v for k, v in ratios.items()},
            "studentized_mean": float(np.mean(stats_all)),
            "studentized_sd": float(np.std(stats_all, ddof=1)),
            "shape_ks_pvalue_by_K_empirical_standardization":
                {str(k): v for k, v in shape_ks.items()},
            "variance_inflation_by_K": {str(k): v for k, v in infl.items()}}


def main():
    Ks = [1, 2, 4, 8, 16, 32]
    seeds = list(range(300))
    out = {"command": CMD,
           "claim": "Theorem 4.5 + Theorem 4.6 (Section 4.2)",
           "config": {"K_values": Ks, "n_seeds": len(seeds), "MAR_per_group": 6,
                      "MAC": 8, "n_cols": 200, "r": R_LATENT, "sigma_rel": SIGMA_REL,
                      "f_scales": F_SCALES.tolist(), "target_level": "low (f=1)"},
           "main": sweep(Ks, seeds),
           "anchor_size_sweep_K8": {f"R{R}_C{C}": sweep([8], list(range(200)), R=R, C=C)
                                    for R, C in [(6, 8), (12, 16), (24, 32), (48, 64)]},
           "mutation_M1_column_treatment_mismatch": sweep(Ks, seeds, mismatch=True),
           "mutation_M2_no_weights": sweep(Ks, seeds, use_weights=False)}
    print(json.dumps(out, indent=2))
    with open("results/claim1.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
