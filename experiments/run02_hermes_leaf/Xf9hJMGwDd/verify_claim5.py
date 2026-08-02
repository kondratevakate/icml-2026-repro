"""verify_claim5.py -- Figures 4/5, Section 5.1: on simulated adjacent-support data
Semi-knockoffs maintains type-I error control while achieving HIGHER POWER than HRT,
and derandomization with 5 permutations under masked correlation further increases power.

*** THIS SCRIPT IS A `toy` REPRODUCTION, BY CONSTRUCTION. ***
The paper's black box m_hat in Figures 4/5 is a neural network / gradient boosting /
random forest. sklearn is not available under the task's numpy/scipy/sympy restriction,
so m_hat here is a RIDGE regression. The split-vs-no-split power mechanism is argued in
the paper to be model-independent (HRT must hold out data, SKO must not), so the
comparison is still informative about the *mechanism* -- but it is NOT the paper's
configuration and the verdict is capped at `toy`.

Two settings, both taken verbatim from the paper:
  S1 "adjacent support" (Sec 5.1 / App F.5.1):
      X ~ N(0, Sigma), Sigma_ij = 0.6^|i-j|; first 0.25p coords of beta in [1,2];
      y = beta'X + eps.
  S2 "masked correlation" (App F.4.2 / Figs 16-18):
      one relevant coordinate l, y = X_l + 0.5 eps1; a correlated NULL variable
      X_{l-1} = X_l + 0.5 eps2.

Methods compared, all at the same nominal alpha:
  SKO-1   : Semi-knockoffs, sign test, n_perm = 1, model fitted on ALL n samples.
  SKO-5   : derandomized Semi-knockoffs, n_perm = 5 (the paper's "5 permutations").
  HRT     : model fitted on a 50/50 train split; conditional resampling of X^j on the
            held-out half; p-value = empirical rank of the observed loss among K resamples.

Reported: type-I error on null coords, power on non-null coords, over many seeds.
"""
import json, time
import numpy as np
from common import (ar1_cov, gen_adjacent, oracle_nu, oracle_rho, ridge_fit, ridge_pred,
                    sko_pair_losses, sign_test_pval)

ALPHA = 0.05
N_SEEDS = 300
K_HRT = 200


# ------------------------------------------------------------------ settings
def setting_adjacent(n, p, rng):
    X, y, beta, Sigma, sn = gen_adjacent(n, p, rng)
    return X, y, beta, Sigma, sn


def setting_masked(n, p, rng):
    """App F.4.2: unique relevant coordinate l; X_{l-1} := X_l + 0.5 eps2 (null but
    highly correlated with the relevant one)."""
    Sigma = ar1_cov(p, 0.6)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    l = int(rng.integers(1, p))
    X[:, l - 1] = X[:, l] + 0.5 * rng.standard_normal(n)
    y = X[:, l] + 0.5 * rng.standard_normal(n)
    beta = np.zeros(p); beta[l] = 1.0
    # empirical covariance is used as the oracle Sigma here because the masking step
    # changes the design covariance away from the AR(1) form.
    Semp = np.cov(np.column_stack([X, y]).T)
    return X, y, beta, Semp, l


# ------------------------------------------------------------------- methods
def sko_pvals(X, y, Sigma_full, beta, sn, n_perm, rng, masked=False):
    n, p = X.shape
    theta = ridge_fit(X, y)                       # NO train-test split
    pred = lambda Z: ridge_pred(theta, Z)
    pv = np.zeros(p)
    for j in range(p):
        if masked:
            nu, rho = emp_cond_means(X, y, j)
        else:
            nu = oracle_nu(X, Sigma_full, j)
            rho = oracle_rho(X, y, Sigma_full, beta, sn, j)
        L1, L2 = sko_pair_losses(X, y, j, nu, rho, pred, rng, n_perm)
        pv[j] = sign_test_pval(L1, L2)
    return pv


def emp_cond_means(X, y, j):
    """Gaussian conditional means from the empirical joint covariance of (X, y).
    Used in the masked setting where the analytic Sigma no longer applies."""
    n, p = X.shape
    Z = np.column_stack([X, y])
    C = np.cov(Z.T)
    idx = np.delete(np.arange(p), j)
    # nu: X^j | X^-j
    w1 = np.linalg.solve(C[np.ix_(idx, idx)], C[j, idx])
    nu = X[:, idx] @ w1
    # rho: X^j | X^-j, y
    idy = np.append(idx, p)
    w2 = np.linalg.solve(C[np.ix_(idy, idy)], C[j, idy])
    rho = Z[:, idy] @ w2
    return nu, rho


def hrt_pvals(X, y, Sigma_full, beta, sn, rng, masked=False):
    """Tansey et al. HRT: fit on train half, resample X^j on the held-out half,
    p-value = (1 + #{resampled loss <= observed loss}) / (1 + K)."""
    n, p = X.shape
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    theta = ridge_fit(X[tr], y[tr])
    pred = lambda Z: ridge_pred(theta, Z)
    Xte, yte = X[te], y[te]
    obs_all = np.mean((yte - pred(Xte)) ** 2)
    pv = np.zeros(p)
    for j in range(p):
        if masked:
            nu, _ = emp_cond_means(X, y, j)
            nu = nu[te]
        else:
            nu = oracle_nu(Xte, Sigma_full, j)
        e = Xte[:, j] - nu
        cnt = 0
        for _ in range(K_HRT):
            Z = Xte.copy(); Z[:, j] = nu + rng.permutation(e)
            if np.mean((yte - pred(Z)) ** 2) <= obs_all:
                cnt += 1
        pv[j] = (1 + cnt) / (1 + K_HRT)
    return pv


# ------------------------------------------------------------------- driver
def run_setting(name, n, p, n_seeds=N_SEEDS):
    acc = {m: {"t1": [], "pow": []} for m in ["SKO-1", "SKO-5", "HRT"]}
    masked = (name == "masked")
    for s in range(n_seeds):
        rng = np.random.default_rng(90_000 + s)
        if masked:
            X, y, beta, Semp, l = setting_masked(n, p, rng)
            Sig, sn = None, None
        else:
            X, y, beta, Sig, sn = setting_adjacent(n, p, rng)
        nn = beta != 0
        res = {
            "SKO-1": sko_pvals(X, y, Sig, beta, sn, 1, rng, masked),
            "SKO-5": sko_pvals(X, y, Sig, beta, sn, 5, rng, masked),
            "HRT":   hrt_pvals(X, y, Sig, beta, sn, rng, masked),
        }
        for m, pv in res.items():
            acc[m]["t1"].append(np.mean(pv[~nn] <= ALPHA))
            acc[m]["pow"].append(np.mean(pv[nn] <= ALPHA))
    return {m: {"type_I_error": float(np.mean(v["t1"])),
                "type_I_mcse": float(np.std(v["t1"]) / np.sqrt(n_seeds)),
                "power": float(np.mean(v["pow"])),
                "power_mcse": float(np.std(v["pow"]) / np.sqrt(n_seeds))}
            for m, v in acc.items()}


if __name__ == "__main__":
    t0 = time.time()
    out = {
        "claim": 5,
        "source": "Figures 4 and 5, Section 5.1; masked-correlation setting App F.4.2",
        "command": "./.venv/bin/python verify_claim5.py",
        "VERDICT_CAP": "toy -- black box is ridge, not the paper's NN/GB/RF (sklearn unavailable)",
        "config": {"alpha": ALPHA, "n_seeds": N_SEEDS, "K_HRT_resamples": K_HRT,
                   "adjacent": {"n": 200, "p": 12}, "masked": {"n": 200, "p": 12}},
        "adjacent_support": run_setting("adjacent", 200, 12),
        "masked_correlation": run_setting("masked", 200, 12),
        "predictions": {
            "type_I": "all methods <= alpha",
            "power": "power(SKO) > power(HRT) because HRT loses half the data",
            "derandomization": "power(SKO-5) >= power(SKO-1) in the masked setting"},
    }
    out["wall_seconds"] = round(time.time() - t0, 2)
    with open("results/claim5.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
