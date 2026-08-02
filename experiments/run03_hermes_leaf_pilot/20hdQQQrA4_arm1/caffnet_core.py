"""Minimal numpy reimplementation of the CAffNet CAffine layer (Eqs. 2, 8, 9, 12).

arm1 (control) — written from notes_paper.md only.
"""
import itertools
import numpy as np


def gamma_set(m, n_out, kmax=None):
    """Gamma = union_{k=1}^{min(m,n_out)} Gamma_k  (Eq. 2)."""
    kmax = min(m, n_out) if kmax is None else kmax
    out = []
    for k in range(1, kmax + 1):
        out.extend(itertools.combinations(range(m), k))
    return out


def P_gamma(f, A, b, gam, w):
    """Eq. (8): P_gamma = f - A_g^+ (A_g f - b_g) + (I - A_g^+ A_g) w."""
    idx = list(gam)
    Ag, bg = A[idx, :], b[idx]
    Ap = np.linalg.pinv(Ag)
    n = A.shape[1]
    return f - Ap @ (Ag @ f - bg) + (np.eye(n) - Ap @ Ag) @ w


def caffnet(f, A, b, w, p=2, kmax=None, tol=1e-9, return_gamma=False):
    """Eq. (12): f if feasible, else nearest feasible candidate from S_P (Eq. 9)."""
    if np.all(A @ f - b <= tol):
        return (f, None) if return_gamma else f
    best, best_d, best_g = None, np.inf, None
    for gam in gamma_set(A.shape[0], A.shape[1], kmax):
        y = P_gamma(f, A, b, gam, w)
        if np.all(A @ y - b <= tol):
            d = np.linalg.norm(y - f, p)
            if d < best_d:
                best, best_d, best_g = y, d, gam
    if best is None:                      # should not happen under Assumption 3.2
        return (f, None) if return_gamma else f
    return (best, best_g) if return_gamma else best


def hardnet_like(f, A, b):
    """HardNet-style single pseudo-inverse correction using the *violated* rows only
    (no combination search, no null-space term) — the baseline CAffNet generalises."""
    v = np.where(A @ f - b > 0)[0]
    if v.size == 0:
        return f
    Av, bv = A[v, :], b[v]
    return f - np.linalg.pinv(Av) @ (Av @ f - bv)


def random_feasible_instance(rng, n_out, m, rank=None, redundant=False):
    """Random (A, b, y0) with A(y0) <= b strictly; optionally rank-deficient/redundant."""
    if rank is None:
        rank = min(m, n_out)
    U = rng.normal(size=(m, rank))
    V = rng.normal(size=(rank, n_out))
    A = U @ V
    if redundant and m >= 2:
        A[-1, :] = 2.0 * A[0, :]          # exactly linearly dependent row
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    A = A / nrm
    y0 = rng.normal(size=n_out)
    b = A @ y0 + rng.uniform(0.05, 1.0, size=m)   # y0 strictly feasible
    return A, b, y0


def sample_in_ball(rng, n, radius, p=2):
    v = rng.normal(size=n)
    v = v / max(np.linalg.norm(v, p), 1e-12)
    return v * radius * rng.uniform(0.0, 0.999) ** (1.0 / n)


def seed_of(*parts):
    """Deterministic seed across processes (python's hash() is salted for str/tuples)."""
    import zlib
    return zlib.crc32(repr(parts).encode()) % (2**32)
