"""Reference numpy implementation of CAffNet's CAffine layer (Eqs 5, 8, 9, 12).

Source: notes_paper.md (paper Sec 3.1-3.2). CPU / numpy only.
"""
import itertools
import numpy as np


def gammas(m, n_out, kmax=None):
    """Index combinations Gamma = U_{k=1..min(m,n_out)} Gamma_k  (Eq 2)."""
    kmax = min(m, n_out) if kmax is None else kmax
    out = []
    for k in range(1, kmax + 1):
        out.extend(itertools.combinations(range(m), k))
    return out


def P_gamma(f, A, b, g, w):
    """Eq (8): projection onto sub-constraint gamma with null-space term w."""
    Ag = A[list(g), :]
    bg = b[list(g)]
    Agp = np.linalg.pinv(Ag)
    n = A.shape[1]
    return f - Agp @ (Ag @ f - bg) + (np.eye(n) - Agp @ Ag) @ w


def caffnet(f, A, b, w=None, kmax=None, p=2, tol=1e-9):
    """Eq (12). Returns (y, info)."""
    m, n = A.shape
    w = np.zeros(n) if w is None else w
    if np.all(A @ f - b <= tol):
        return f.copy(), {"branch": "unconstrained", "gamma": None, "n_feasible": None}
    best, bg, nfeas = None, None, 0
    for g in gammas(m, n, kmax):
        y = P_gamma(f, A, b, g, w)
        if np.all(A @ y - b <= tol):
            nfeas += 1
            d = np.linalg.norm(y - f, p)
            if best is None or d < best[0]:
                best, bg = (d, y), g
    if best is None:
        return None, {"branch": "FAILED", "gamma": None, "n_feasible": 0}
    return best[1], {"branch": "projected", "gamma": bg, "n_feasible": nfeas}


def hardnet(f, A, b, tol=1e-9):
    """HardNet-style projection: requires A with full row rank (uses (A A^T)^-1)."""
    if np.all(A @ f - b <= tol):
        return f.copy(), "unconstrained"
    m = A.shape[0]
    G = A @ A.T
    if np.linalg.matrix_rank(A) < m:
        return None, "rank_deficient"  # (A A^T) singular -> formula undefined
    return f - A.T @ np.linalg.solve(G, np.maximum(A @ f - b, 0.0)), "projected"


def random_feasible_problem(rng, n_out, m, scale=1.0):
    """Random A,b with guaranteed non-empty S(x): pick interior point y0, set b = A y0 + s."""
    A = rng.normal(size=(m, n_out)) * scale
    y0 = rng.normal(size=n_out)
    b = A @ y0 + np.abs(rng.normal(size=m)) * 0.5 + 1e-3
    return A, b, y0
