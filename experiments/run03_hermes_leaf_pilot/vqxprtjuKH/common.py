"""Shared helpers for reproducing the 6 anchored claims of arXiv:2502.18463
'Allocating Variance to Maximize Expectation' (Paes Leme, Stein, Teng, Worah).

All computations are CPU-only using numpy/scipy.  No GPU, no ML libraries.
Conventions
-----------
- VarAlloc:   independent Gaussians, maximize E[max_i X_i],  sum_i sigma_i^2 = 1.
- CorrVarAlloc: (X_1..X_n) ~ N(mu, Sigma), Sigma PSD, sum_i Sigma_ii = 1.
- GraphVarAlloc: m subsets S_j subset [n], maximize E[ sum_j max_{i in S_j} X_i ],
  sum_i sigma_i^2 = 1 (independent case).
"""
import numpy as np
from scipy import stats
from scipy import integrate

PHI = stats.norm.cdf          # standard normal CDF
PHIinv = stats.norm.ppf


# ---------------------------------------------------------------------------
# Exact 1-D integrals for independent zero-mean variables (Lemma 2.1 setting)
# ---------------------------------------------------------------------------
def emax0_exact(stds):
    """E[ max(0, max_i Y_i) ] for independent zero-mean Y_i ~ N(0, sigma_i^2).

    Exact 1-D integral:  int_0^inf (1 - prod_i Phi(-t/sigma_i)) dt.
    Used to verify Lemma 2.1 (small-variance contribution bound).
    """
    stds = np.asarray(stds, dtype=float)
    stds = stds[stds > 0]
    if stds.size == 0:
        return 0.0

    def integrand(t):
        # P(max_i Y_i > t) = 1 - prod_i P(Y_i <= t) = 1 - prod_i Phi(t/sigma_i)
        with np.errstate(divide="ignore"):
            logprod = np.sum(np.log(PHI(t / stds)))
        return 1.0 - np.exp(logprod)

    # integrate to a safe upper cutoff
    t_max = max(20.0, 50.0 * stds.max())
    val, _ = integrate.quad(integrand, 0.0, t_max, limit=200)
    return val


def emax_exact(stds):
    """E[ max_i Y_i ] for independent zero-mean Y_i ~ N(0, sigma_i^2).

    By symmetry E[max Y] = 2 * E[max(0, max Y)] - E[|max Y|] ... we use the
    identity  E[max Y] = int_0^inf P(max>t) dt - int_0^inf P(max<-t) dt.
    For zero-mean symmetric variables P(max<-t) = P(min>t) = prod_i Phi(-t/sigma_i)
    (all below -t).  So E[max] = int_0^inf [1 - prod Phi(-t/s_i) - prod Phi(-t/s_i)] dt
                        = int_0^inf [1 - 2*prod Phi(-t/s_i)] dt.
    """
    stds = np.asarray(stds, dtype=float)
    stds = stds[stds > 0]
    if stds.size == 0:
        return 0.0

    def integrand(t):
        # E[max] = int_0^inf [P(max>t) - P(max<-t)] dt
        # P(max>t)   = 1 - prod_i Phi(t/sigma_i)
        # P(max<-t)  = prod_i Phi(-t/sigma_i) = prod_i (1 - Phi(t/sigma_i))
        with np.errstate(divide="ignore"):
            lp = np.sum(np.log(PHI(t / stds)))          # prod Phi(t/sigma_i)
            lm = np.sum(np.log(1.0 - PHI(t / stds)))     # prod (1 - Phi(t/sigma_i))
        return 1.0 - np.exp(lp) - np.exp(lm)

    t_max = max(20.0, 50.0 * stds.max())
    val, _ = integrate.quad(integrand, 0.0, t_max, limit=200)
    return val


# ---------------------------------------------------------------------------
# Monte-Carlo estimators
# ---------------------------------------------------------------------------
def emax_mc(means, stds, nsamples=200000, seed=0, rng=None):
    """MC estimate of E[ max_i X_i ] for independent X_i ~ N(mu_i, sigma_i^2)."""
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed)
    z = rng.standard_normal((nsamples, means.size))
    X = means + stds * z
    return float(X.max(axis=1).mean())


def esummax_mc(means, stds, sets, nsamples=200000, seed=0, rng=None):
    """MC estimate of E[ sum_j max_{i in S_j} X_i ] (GraphVarAlloc objective,
    independent case).  `sets` is a list of arrays of indices."""
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed)
    z = rng.standard_normal((nsamples, means.size))
    X = means + stds * z
    total = np.zeros(nsamples)
    for S in sets:
        S = np.asarray(S, dtype=int)
        total += X[:, S].max(axis=1)
    return float(total.mean())


def emax_mc_corr(cov, nsamples=200000, seed=0, rng=None, means=None):
    """MC estimate of E[ max_i X_i ] for X ~ N(means, cov) (correlated)."""
    cov = np.asarray(cov, dtype=float)
    if means is None:
        means = np.zeros(cov.shape[0])
    else:
        means = np.asarray(means, dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(means, cov, size=nsamples)
    return float(X.max(axis=1).mean())


def esummax_mc_corr(cov, sets, nsamples=200000, seed=0, rng=None, means=None):
    """MC estimate of E[ sum_j max_{i in S_j} X_i ] for correlated X~N(means,cov)."""
    cov = np.asarray(cov, dtype=float)
    if means is None:
        means = np.zeros(cov.shape[0])
    else:
        means = np.asarray(means, dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(means, cov, size=nsamples)
    total = np.zeros(nsamples)
    for S in sets:
        S = np.asarray(S, dtype=int)
        total += X[:, S].max(axis=1)
    return float(total.mean())


# ---------------------------------------------------------------------------
# Instance construction helpers
# ---------------------------------------------------------------------------
def all_subsets_of_size(n, k):
    """Return list of all subsets (as numpy arrays) of {0..n-1} of size k."""
    from itertools import combinations
    return [np.array(c, dtype=int) for c in combinations(range(n), k)]


def er_blocks_cov(diag, signs):
    """Build the block-diagonal covariance used in the paper's Figs 1-2.

    n = 2*B variables grouped into B blocks of size 2.  Within block b the two
    variables are perfectly correlated with sign s_b in {+1,-1}:
        Sigma_{2b,   2b}   = diag[2b]
        Sigma_{2b+1, 2b+1} = diag[2b+1]
        Sigma_{2b, 2b+1}   = s_b * sqrt(diag[2b]*diag[2b+1])
    `diag` length 2B, `signs` length B.
    """
    diag = np.asarray(diag, dtype=float)
    n = diag.size
    B = n // 2
    cov = np.zeros((n, n))
    for i in range(n):
        cov[i, i] = diag[i]
    for b in range(B):
        i, j = 2 * b, 2 * b + 1
        cov[i, j] = cov[j, i] = signs[b] * np.sqrt(diag[i] * diag[j])
    return cov


def conc_alloc(t, n, total=1.0):
    """Canonical 'concentrate on t variables, equal variance' allocation.
    Returns stds array (length n) with t entries = sqrt(total/t), rest 0."""
    stds = np.zeros(n)
    if t <= 0:
        return stds
    t = min(t, n)
    stds[:t] = np.sqrt(total / t)
    return stds
