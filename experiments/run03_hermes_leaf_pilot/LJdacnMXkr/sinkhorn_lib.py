"""Shared utilities: symmetric Sinkhorn normalization of diffusion kernels.

Reference: "Sinkhorn Normalization of Diffusion Kernels" (OpenReview LJdacnMXkr,
arXiv 2507.06161), Theorem 4.1 / 4.2.  Implemented from first principles.
"""
import numpy as np

SEED = 20260802


def rng(offset=0):
    return np.random.default_rng(SEED + offset)


def pdist2(X, Y=None):
    Y = X if Y is None else Y
    d2 = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T
    return np.maximum(d2, 0.0)


def gaussian_kernel(X, eps, Y=None):
    return np.exp(-pdist2(X, Y) / (eps**2))


def exponential_kernel(X, eps, Y=None):
    return np.exp(-np.sqrt(pdist2(X, Y)) / eps)


def mahalanobis_kernel(X, Sigmas, eps):
    """Covariance-aware (GMM) kernel: symmetric, positive entries.
    k_ij = exp(-0.5 * (x_i-x_j)^T (Si+Sj)^{-1} (x_i-x_j) / eps^2)."""
    n = X.shape[0]
    K = np.empty((n, n))
    for i in range(n):
        diff = X[i] - X                       # (n,d)
        S = Sigmas[i][None, :, :] + Sigmas    # (n,d,d)
        sol = np.linalg.solve(S, diff[:, :, None])[:, :, 0]
        K[i] = np.exp(-0.5 * np.einsum('nd,nd->n', diff, sol) / eps**2)
    return 0.5 * (K + K.T)


def sym_sinkhorn(K, m=None, tol=1e-12, max_iter=10000, track=False):
    """Symmetric Sinkhorn scaling (Thm 4.1).

    Find d>0 with S = diag(d) K diag(d) symmetric and S 1 = m, so that
    P = diag(m)^{-1} S is a diffusion operator: row sums 1 (mass conservation),
    positive entries, and self-adjoint w.r.t. <u,v>_m = sum_i m_i u_i v_i
    because M P = S = S^T = (M P)^T.

    Returns P, d, history of max relative row-sum error.
    """
    n = K.shape[0]
    m = np.ones(n) if m is None else np.asarray(m, float)
    d = np.ones(n)
    hist = []
    for it in range(max_iter):
        r = K @ d
        err = np.max(np.abs(d * r - m) / m)
        hist.append(err)
        if err < tol:
            break
        d = np.sqrt(d * m / r)
    S = (d[:, None] * K) * d[None, :]
    P = S / m[:, None]
    if track:
        return P, d, np.array(hist)
    return P, d


def row_normalize(K):
    return K / K.sum(1, keepdims=True)


def sym_normalize(K):
    s = K.sum(1)
    return K / np.sqrt(np.outer(s, s))


def props(P, m=None):
    n = P.shape[0]
    m = np.ones(n) if m is None else np.asarray(m, float)
    M = np.diag(m)
    sym = np.max(np.abs(M @ P - (M @ P).T))
    mass = np.max(np.abs(P.sum(1) - 1.0))
    pos = float(P.min())
    A = np.diag(np.sqrt(m)) @ P @ np.diag(1 / np.sqrt(m))
    ev = np.linalg.eigvalsh(0.5 * (A + A.T))
    return dict(self_adjoint_residual=float(sym), mass_error=float(mass),
                min_entry=pos, lam_min=float(ev.min()), lam_max=float(ev.max()))


def dump(path, obj):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)
    print(json.dumps(obj, indent=2, default=float))
