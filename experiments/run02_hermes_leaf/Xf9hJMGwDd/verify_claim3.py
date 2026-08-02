"""verify_claim3.py -- Theorem 4.1 (Optimization stability), Section 4.2, arXiv:2601.23124v2:

    for j in H_0,   || theta~^j - theta_hat ||_2  <=  O_P( sqrt( log(1/delta) / n ) )

where theta_hat minimises the l2-regularised empirical risk of Eq. (2),
    R_n(theta) = (1/n) sum_i l(theta' chi_i, z_i) + lambda ||theta||^2,
over R^p, theta^{-j}_hat minimises it over R^{p-1} (j-th coordinate removed), and
theta~^j = (0, theta^{-j}_hat) is the restricted optimiser padded with a zero.

For the quadratic loss (one of the two losses the paper says it verifies the regularity
assumptions for) the minimiser is CLOSED FORM,
    theta_hat = (chi'chi/n + lambda I)^{-1} chi'z/n,
so there is zero optimiser noise and the measured quantity is exact.

Setting = Figure 1 of the paper: z = chi beta + eps, beta 0.25-sparse with important
features grouped in blocks of 5, p=50, chi ~ N(0, Sigma), Sigma_ij = 0.6^|i-j|,
noise level ||chi beta||/2.

The claim has TWO functional contents, tested separately:
  (T1) rate in n     : the (1-delta)-quantile of the norm must decay like n^{-1/2}
                       -> log-log regression slope ~ -0.5.
  (T2) rate in delta : at fixed n, quantile(1-delta) must grow like sqrt(log(1/delta))
                       -> regressing quantile on sqrt(log(1/delta)) must be near-linear
                          through the origin (high R^2).

MUTATION: run the identical estimator on a NON-NULL coordinate j (beta_j != 0).
Theorem 4.1 requires j in H_0; if the mechanism is the nullity of the feature (and not
merely ridge shrinkage), the quantity must STOP decaying and plateau near |beta_j|.
Predicted log-log slope ~ 0, not -0.5.

Usage: ./.venv/bin/python verify_claim3.py
"""
import json, time
import numpy as np
from common import ar1_cov

P = 50
LAM = 1e-2
NS = [125, 250, 500, 1000, 2000, 4000]
REPS = 250
_CHOL = None
DELTAS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]


def gen_fig1(n, p, rng):
    """Figure 1 DGP: beta is 0.25-sparse, important features in blocks of 5,
    noise level ||chi beta|| / 2."""
    global _CHOL
    if _CHOL is None:
        _CHOL = np.linalg.cholesky(ar1_cov(p, 0.6))
    chi = rng.standard_normal((n, p)) @ _CHOL.T
    n_imp = int(round(0.25 * p))
    n_blocks = n_imp // 5
    starts = rng.choice(np.arange(0, p - 5, 5), size=n_blocks, replace=False)
    beta = np.zeros(p)
    for s in starts:
        beta[s:s + 5] = rng.uniform(1.0, 2.0, size=5)
    signal = chi @ beta
    sigma = np.linalg.norm(signal) / (2 * np.sqrt(n))
    z = signal + sigma * rng.standard_normal(n)
    return chi, z, beta


def ridge(chi, z, lam=LAM):
    n, p = chi.shape
    return np.linalg.solve(chi.T @ chi / n + lam * np.eye(p), chi.T @ z / n)


def stability_norm(chi, z, j, lam=LAM):
    """|| theta~^j - theta_hat ||_2, exact."""
    p = chi.shape[1]
    th_full = ridge(chi, z, lam)
    idx = np.delete(np.arange(p), j)
    th_sub = ridge(chi[:, idx], z, lam)
    th_tilde = np.zeros(p)
    th_tilde[idx] = th_sub
    return float(np.linalg.norm(th_tilde - th_full))


def collect(null_feature=True):
    """Returns {n: array of REPS stability norms}."""
    res = {}
    for n in NS:
        vals = []
        for r in range(REPS):
            rng = np.random.default_rng((1 if null_feature else 2) * 10**7 + n * 1000 + r)
            chi, z, beta = gen_fig1(n, P, rng)
            pool = np.where(beta == 0)[0] if null_feature else np.where(beta != 0)[0]
            j = int(rng.choice(pool))
            vals.append(stability_norm(chi, z, j))
        res[n] = np.array(vals)
    return res


def loglog_slope(res, delta=0.1):
    ns = np.array(sorted(res))
    q = np.array([np.quantile(res[n], 1 - delta) for n in ns])
    slope, intercept = np.polyfit(np.log(ns), np.log(q), 1)
    pred = np.exp(intercept) * ns ** slope
    r2 = 1 - np.sum((np.log(q) - np.log(pred)) ** 2) / np.sum((np.log(q) - np.log(q).mean()) ** 2)
    return float(slope), float(r2), {int(n): float(v) for n, v in zip(ns, q)}


def delta_fit(res, n):
    """T2: quantile(1-delta) vs sqrt(log(1/delta)), linear through the origin."""
    v = res[n]
    x = np.array([np.sqrt(np.log(1 / d)) for d in DELTAS])
    q = np.array([np.quantile(v, 1 - d) for d in DELTAS])
    c = float(np.sum(x * q) / np.sum(x * x))          # least squares through origin
    r2 = 1 - np.sum((q - c * x) ** 2) / np.sum((q - q.mean()) ** 2)
    return {"slope_through_origin": c, "R2": float(r2),
            "deltas": DELTAS, "quantiles": [float(t) for t in q]}


if __name__ == "__main__":
    t0 = time.time()
    null_res = collect(True)
    nonnull_res = collect(False)

    s_null, r2_null, q_null = loglog_slope(null_res)
    s_mut, r2_mut, q_mut = loglog_slope(nonnull_res)

    out = {
        "claim": 3,
        "source": "Theorem 4.1 (Optimization stability), Eq. (3), Section 4.2; DGP = Figure 1",
        "command": "./.venv/bin/python verify_claim3.py",
        "config": {"p": P, "lambda": LAM, "loss": "quadratic (closed-form ridge, exact minimiser)",
                   "n_grid": NS, "reps_per_n": REPS, "deltas": DELTAS,
                   "dgp": "Figure 1: z=chi*beta+eps, beta 0.25-sparse in blocks of 5, "
                          "chi~N(0,0.6^|i-j|), noise ||chi beta||/2"},
        "T1_rate_in_n_NULL_feature": {
            "loglog_slope": s_null, "R2": r2_null, "predicted_slope": -0.5,
            "quantile_0.9_by_n": q_null},
        "T2_rate_in_delta_NULL_feature": {str(n): delta_fit(null_res, n) for n in [500, 2000, 4000]},
        "MUTATION_non_null_feature": {
            "loglog_slope": s_mut, "R2": r2_mut,
            "prediction": "slope ~ 0 (plateau), because Thm 4.1 requires j in H_0",
            "quantile_0.9_by_n": q_mut},
        "ratio_nonnull_over_null_at_largest_n": float(
            q_mut[max(NS)] / q_null[max(NS)]),
        "medians_null_by_n": {int(n): float(np.median(v)) for n, v in null_res.items()},
        "medians_nonnull_by_n": {int(n): float(np.median(v)) for n, v in nonnull_res.items()},
    }
    out["wall_seconds"] = round(time.time() - t0, 2)
    with open("results/claim3.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
