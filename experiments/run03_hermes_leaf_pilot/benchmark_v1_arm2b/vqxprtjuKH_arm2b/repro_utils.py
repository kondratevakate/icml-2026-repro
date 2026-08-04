"""Shared numerics for vqxprtjuKH reproduction (CPU only, numpy/scipy).

GraphVarAlloc: X ~ N(0, Sigma), trace(Sigma) = 1, sets S_1..S_m subset [n];
OBJ = E sum_j max_{i in S_j} X_i.
Independent case is computed exactly by deterministic quadrature; correlated cases
by Monte Carlo with common random numbers (deterministic given the seed).
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

SEED = 20260803
T = np.linspace(-7.0, 9.0, 1201)


def emax_indep(sig2):
    """E[max_i X_i] for independent zero-mean Gaussians with variances sig2."""
    sd = np.sqrt(np.maximum(np.asarray(sig2, float), 0.0))
    live = sd > 1e-9
    if live.sum() == 0:
        return 0.0
    if live.sum() == 1:
        return 0.0  # single zero-mean Gaussian
    logF = norm.logcdf(T[None, :] / sd[live][:, None]).sum(axis=0)
    if (~live).any():
        logF = logF + np.where(T >= 0, 0.0, -np.inf)  # point mass at 0
    F = np.exp(logF)
    pos = T >= 0
    return float(np.trapezoid((1.0 - F)[pos], T[pos]) - np.trapezoid(F[~pos], T[~pos]))


def graph_obj_indep(sig2, sets):
    sig2 = np.asarray(sig2, float)
    return float(sum(emax_indep(sig2[list(S)]) for S in sets))


def make_Z(n, nsamp=20000, seed=SEED):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((nsamp, n))
    return np.vstack([Z, -Z])


def sigma_from_var(sig2, mode):
    """Build Sigma with diagonal sig2 and 2x2 block correlations as in Figure 1:
    Sigma_{2i+1,2i+2} = +/- sqrt(Sigma_ii Sigma_jj) (perfect +/- correlation)."""
    n = len(sig2)
    S = np.diag(np.asarray(sig2, float))
    if mode == "independent":
        return S
    sgn = 1.0 if mode == "positive" else -1.0
    for i in range(0, n - 1, 2):
        S[i, i + 1] = S[i + 1, i] = sgn * np.sqrt(sig2[i] * sig2[i + 1])
    return S


def sqrtm_psd(S):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return V @ (np.sqrt(np.clip(w, 0, None))[:, None] * V.T)


def graph_obj_mc(sig2, sets, mode, Z):
    X = Z @ sqrtm_psd(sigma_from_var(sig2, mode))
    return float(sum(X[:, list(S)].max(axis=1).mean() for S in sets))


def simplex(theta):
    e = np.exp(theta - theta.max())
    return e / e.sum()


def optimize(sets, n, mode="independent", Z=None, restarts=3, seed=SEED, maxiter=2500):
    """Maximise OBJ over {sig2 >= 0, sum sig2 = 1}. Returns (best_value, best_sig2)."""
    rng = np.random.default_rng(seed)
    if mode == "independent":
        f = lambda th: -graph_obj_indep(simplex(th), sets)
    else:
        f = lambda th: -graph_obj_mc(simplex(th), sets, mode, Z)
    best, bx = -np.inf, None
    for r in range(restarts):
        th0 = np.zeros(n) if r == 0 else rng.normal(0, 1.5, n)
        res = minimize(f, th0, method="Nelder-Mead",
                       options=dict(maxiter=maxiter, fatol=1e-9, xatol=1e-6))
        if -res.fun > best:
            best, bx = -res.fun, simplex(res.x)
    return best, bx


def er_closed_neighbourhoods(n, p, rng):
    """G(n,p) Erdos-Renyi graph; sets S_j = closed neighbourhood of vertex j."""
    A = (rng.random((n, n)) < p)
    A = np.triu(A, 1)
    A = A + A.T
    return [tuple(sorted(set(np.flatnonzero(A[j]).tolist() + [j]))) for j in range(n)]
