"""Shared helpers: data generation, oracle conditional expectations, ridge black box,
semi-knockoff statistic, and the knockoff threshold of Eq. (1).

All formulas traced to arXiv:2601.23124v2:
  - nu_j(x^-j) = E[X^j | X^-j]                        (Sec 4.2, Gaussian closed form)
  - rho_j(x^-j, y) = E[X^j | X^-j, y]                 (Sec 3)
  - Xtilde_1^(j)j = nu(X^-j) + eps_{j,1,pi1(i)}       (Algorithm 1)
  - Xtilde_2^(j)j = rho(X^-j,y) + eps_{j,2,pi2(i)}    (Algorithm 1)
  - threshold T_q                                     (Eq. 1, Sec 2.3.2)
"""
import numpy as np
from scipy.stats import binomtest, wilcoxon


# ---------------------------------------------------------------- data
def ar1_cov(p, rho=0.6):
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def gen_adjacent(n, p, rng, sigma_noise=1.0, frac=0.25, corr=0.6):
    """Sec 5.1 / App F.5.1: X ~ N(0, Sigma), Sigma_ij = 0.6^|i-j|;
    first 0.25p coords of beta in [1,2], rest zero; y = beta'X + eps."""
    Sigma = ar1_cov(p, corr)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    beta = np.zeros(p)
    k = max(1, int(round(frac * p)))
    beta[:k] = rng.uniform(1.0, 2.0, size=k)
    y = X @ beta + sigma_noise * rng.standard_normal(n)
    return X, y, beta, Sigma, sigma_noise


# ------------------------------------------- oracle conditional expectations
def oracle_nu(X, Sigma, j):
    """E[X^j | X^-j] for a centred Gaussian design (exact)."""
    idx = np.delete(np.arange(Sigma.shape[0]), j)
    s_j_mj = Sigma[j, idx]
    S_mj = Sigma[np.ix_(idx, idx)]
    w = np.linalg.solve(S_mj, s_j_mj)
    return X[:, idx] @ w


def oracle_rho(X, y, Sigma, beta, sigma_noise, j):
    """E[X^j | X^-j, y] exactly: (X, y) is jointly Gaussian because y = beta'X + eps."""
    p = Sigma.shape[0]
    idx = np.delete(np.arange(p), j)
    # joint covariance of (X^-j, y) and cross-cov with X^j
    cov_X_y = Sigma @ beta                       # Cov(X, y)
    var_y = beta @ Sigma @ beta + sigma_noise ** 2
    C = np.empty((p, p))                          # cond. block for (X^-j, y)
    C[: p - 1, : p - 1] = Sigma[np.ix_(idx, idx)]
    C[: p - 1, p - 1] = cov_X_y[idx]
    C[p - 1, : p - 1] = cov_X_y[idx]
    C[p - 1, p - 1] = var_y
    c = np.concatenate([Sigma[j, idx], [cov_X_y[j]]])
    w = np.linalg.solve(C, c)
    Z = np.column_stack([X[:, idx], y])
    return Z @ w


# ---------------------------------------------------------------- black box
def ridge_fit(X, y, lam=1e-2):
    n, p = X.shape
    Xc = np.column_stack([X, np.ones(n)])
    return np.linalg.solve(Xc.T @ Xc / n + lam * np.eye(p + 1), Xc.T @ y / n)


def ridge_pred(theta, X):
    return np.column_stack([X, np.ones(X.shape[0])]) @ theta


# ------------------------------------------------- semi-knockoff imputations
def sko_pair_losses(X, y, j, nu, rho, predict, rng, n_perm=1):
    """Algorithm 1: build Xtilde_1 and Xtilde_2 and return per-sample squared losses.
    `nu`, `rho` are length-n vectors of the fitted/oracle conditional expectations."""
    e1 = X[:, j] - nu
    e2 = X[:, j] - rho
    L1, L2 = [], []
    for _ in range(n_perm):
        X1 = X.copy(); X1[:, j] = nu + rng.permutation(e1)
        X2 = X.copy(); X2[:, j] = rho + rng.permutation(e2)
        L1.append((y - predict(X1)) ** 2)
        L2.append((y - predict(X2)) ** 2)
    return np.concatenate(L1), np.concatenate(L2)


def hrt_style_losses(X, y, j, nu, predict, rng, n_perm=1):
    """MUTATION component: asymmetric statistic l(m(Xtilde1)) - l(m(X)) evaluated on the
    training data (i.e. the HRT statistic with the train-test split removed).
    The paper (Sec 3) states this is exactly what breaks exchangeability under the null."""
    e1 = X[:, j] - nu
    L1, L0 = [], []
    for _ in range(n_perm):
        X1 = X.copy(); X1[:, j] = nu + rng.permutation(e1)
        L1.append((y - predict(X1)) ** 2)
        L0.append((y - predict(X)) ** 2)
    return np.concatenate(L1), np.concatenate(L0)


def sign_test_pval(La, Lb):
    """Exact nonparametric paired sign test, alternative 'greater' (semi_KO.py, p_val='sign_test')."""
    d = La - Lb
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0
    return binomtest(int(np.sum(nz > 0)), nz.size, alternative="greater").pvalue


def wilcoxon_pval(La, Lb):
    try:
        return wilcoxon(La, Lb, alternative="greater").pvalue
    except ValueError:
        return 1.0


# ---------------------------------------------------------------- threshold
def knockoff_threshold(W, q=0.1, offset=1):
    """Eq. (1). offset=1 is the paper's procedure; offset=0 is the mutation."""
    t_grid = np.sort(np.concatenate([[0.0], np.abs(W[W != 0])]))
    for t in t_grid:
        num = offset + np.sum(W <= -t)
        den = max(np.sum(W >= t), 1)
        if num / den <= q:
            return t
    return np.inf


def fdp(selected, non_null):
    if len(selected) == 0:
        return 0.0
    false = sum(1 for j in selected if j not in non_null)
    return false / len(selected)
