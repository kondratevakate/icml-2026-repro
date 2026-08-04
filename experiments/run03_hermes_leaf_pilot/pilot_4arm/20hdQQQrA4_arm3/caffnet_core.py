"""caffnet_core.py — faithful numpy implementation of CAffNet's CAffine layer.

Eq. (8), (9), (12) of arXiv:2605.24437v1 (see notes_paper.md).
Pure numpy, CPU, no training here — this is the deterministic projection mechanism.
"""
import itertools
import numpy as np


def gamma_set(m, n_out, kmax=None):
    """Gamma = union_{k=1..min(m,n_out)} Gamma_k  (Sec. 3.1)."""
    kk = min(m, n_out) if kmax is None else kmax
    out = []
    for k in range(1, kk + 1):
        out.extend(itertools.combinations(range(m), k))
    return out


def P_gamma(f_theta, w_phi, A, b, g):
    """Eq. (8): P_g = f - Ag^dag (Ag f - bg) + (I - Ag^dag Ag) w."""
    Ag = A[list(g), :]
    bg = b[list(g)]
    Agd = np.linalg.pinv(Ag)
    n = A.shape[1]
    return f_theta - Agd @ (Ag @ f_theta - bg) + (np.eye(n) - Agd @ Ag) @ w_phi


def caffnet_output(f_theta, w_phi, A, b, p=2.0, kmax=None, select="min", tol=1e-9,
                   use_nullspace=True):
    """Eq. (9) + (12). Returns (y, info)."""
    m, n = A.shape
    if np.all(A @ f_theta <= b + tol):
        return f_theta, {"case": "feasible_f_theta", "n_candidates": 0, "n_feasible": 0}
    w = w_phi if use_nullspace else np.zeros_like(w_phi)
    cands, feas = [], []
    for g in gamma_set(m, n, kmax):
        y = P_gamma(f_theta, w, A, b, g)
        cands.append(y)
        if np.all(A @ y <= b + tol):
            feas.append(y)
    if not feas:
        return None, {"case": "no_feasible_candidate", "n_candidates": len(cands),
                      "n_feasible": 0}
    d = [np.linalg.norm(y - f_theta, ord=p) for y in feas]
    idx = int(np.argmax(d)) if select == "max" else int(np.argmin(d))
    return feas[idx], {"case": "projected", "n_candidates": len(cands),
                       "n_feasible": len(feas), "dist": float(d[idx])}


def hardnet_aff(f_theta, A, b, tol=1e-12):
    """HardNet-Aff style single pseudo-inverse correction for one-sided A y <= b.

    y = f - A^dag ReLU(A f - b)  (Min & Azizan 2025; exact under the full-row-rank
    assumption that CAffNet drops).
    """
    v = np.maximum(A @ f_theta - b, 0.0)
    return f_theta - np.linalg.pinv(A) @ v


def random_feasible_system(rng, m, n, dep_rows=0, scale=1.0):
    """Random A y <= b with a non-empty feasible set; optional linearly dependent rows."""
    A = rng.normal(size=(m, n)) * scale
    for i in range(dep_rows):
        if m >= 2:
            src = rng.integers(0, m)
            dst = (src + 1 + i) % m
            A[dst] = rng.choice([1.0, -1.0, 2.0]) * A[src]
    y0 = rng.normal(size=n)
    b = A @ y0 + np.abs(rng.normal(size=m)) * 0.5
    return A, b, y0
