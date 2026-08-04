"""Shared first-principles implementation of symmetric Sinkhorn normalization
of diffusion kernels (paper: LJdacnMXkr, arXiv 2507.06161).

Setup (Theorem 4.1 as stated in the anchored claim):
  W : symmetric, entrywise-positive smoothing/affinity matrix
  m : positive mass vector (mass-weighted inner product <x,y>_m = x^T diag(m) y)

Symmetric Sinkhorn seeks d > 0 with
      diag(d) W diag(d) 1 = m
so that S = diag(d) W diag(d) is symmetric, positive, with row sums = m.
The diffusion operator is then
      T = diag(m)^{-1} S
which satisfies T 1 = 1 (mass conservation) and diag(m) T = S = S^T, i.e. T is
self-adjoint w.r.t. <.,.>_m.

Fixed-point iteration:  d <- sqrt( d * m / (W d) )   (symmetric Sinkhorn)
"""
import numpy as np


def gaussian_kernel(X, eps, metric_inv=None):
    if metric_inv is None:
        D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    else:
        Dv = X[:, None, :] - X[None, :, :]
        D2 = np.einsum('ijk,kl,ijl->ij', Dv, metric_inv, Dv)
    return np.exp(-D2 / (2.0 * eps ** 2))


def exponential_kernel(X, eps):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    return np.exp(-D / eps)


def sym_sinkhorn(W, m, tol=1e-12, max_iter=10000, return_hist=False):
    """Return d, iters, history of relative normalization error."""
    n = W.shape[0]
    d = np.ones(n)
    hist = []
    it = 0
    for it in range(1, max_iter + 1):
        r = d * (W @ d)              # row sums of diag(d)W diag(d)
        err = np.max(np.abs(r - m) / m)
        hist.append(err)
        if err < tol:
            break
        d = np.sqrt(d * m / (W @ d))
    r = d * (W @ d)
    hist.append(np.max(np.abs(r - m) / m))
    if return_hist:
        return d, it, np.array(hist)
    return d, it


def sinkhorn_operator(W, m):
    d, it = sym_sinkhorn(W, m)
    S = (d[:, None] * W) * d[None, :]
    T = S / m[:, None]
    return T, S, d, it


def row_normalized(W):
    return W / W.sum(1, keepdims=True)


def sym_normalized(W):
    dg = W.sum(1)
    return W / np.sqrt(np.outer(dg, dg))


def props(T, m, tol=1e-8):
    """Check the four Theorem-4.1 properties."""
    n = T.shape[0]
    M = np.diag(m)
    sym_err = np.max(np.abs(M @ T - (M @ T).T)) / np.max(np.abs(M @ T))
    mass_err = np.max(np.abs(T @ np.ones(n) - 1.0))
    pos = bool(np.all(T > 0))
    ev = np.linalg.eigvals(np.diag(m ** 0.5) @ T @ np.diag(m ** -0.5))
    ev = np.sort(np.real(ev))
    return dict(
        self_adjoint_m=bool(sym_err < tol), sym_err=float(sym_err),
        mass_conserving=bool(mass_err < tol), mass_err=float(mass_err),
        positive=pos,
        eig_min=float(ev[0]), eig_max=float(ev[-1]),
        spectral_damping=bool(ev[0] > -tol and ev[-1] < 1 + tol),
        max_imag=float(np.max(np.abs(np.imag(
            np.linalg.eigvals(np.diag(m ** 0.5) @ T @ np.diag(m ** -0.5)))))),
    )
