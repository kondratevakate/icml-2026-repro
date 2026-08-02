"""Independent numpy reimplementation of the CAffine layer (CAffNet, OpenReview 20hdQQQrA4).

Written from the paper's Section 3 equations only (arm2 run). No official code was executed.

CAffine projection for a candidate sub-constraint set Gamma:

    P_Gamma(x) = f(x) - A_Gamma^+ (A_Gamma f(x) - b_Gamma)
                       + (I - A_Gamma^+ A_Gamma) w(x)

Selection rule: enumerate sub-constraint sets of cardinality at most
min(m, n_out) (as the paper prescribes), keep the candidates that satisfy
the FULL constraint system A y <= b, and return the feasible candidate
closest to f(x).

Baseline for the "unlike prior work (HardNet)" contrast: a single
pseudo-inverse correction using the violated rows only, with no
enumeration and no feasibility selection.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

TOL = 1e-9


def pinv(M):
    return np.linalg.pinv(M)


def proj_gamma(A, b, f, w, idx):
    """Eq. (CAffine) restricted to sub-constraint set `idx` (tuple of rows)."""
    n = f.shape[0]
    if len(idx) == 0:
        return f + w
    Ag = A[list(idx), :]
    bg = b[list(idx)]
    Agp = pinv(Ag)
    return f - Agp @ (Ag @ f - bg) + (np.eye(n) - Agp @ Ag) @ w


def feasible(A, b, y, tol=1e-7):
    return bool(np.all(A @ y - b <= tol))


def max_violation(A, b, y):
    return float(np.max(np.maximum(A @ y - b, 0.0)))


def caffine(A, b, f, w=None, w_scales=(1.0, 0.5, 0.25, 0.0), return_info=False):
    """Full CAffine layer: enumerate sub-constraint sets, select a feasible projection.

    w_scales implements the practical safeguard that the trainable null-space
    term is shrunk when no feasible candidate exists at full scale; with w = 0
    (or w_scales == (0.0,)) the enumeration is exactly the active-set
    enumeration of the Euclidean projection, so feasibility is guaranteed
    whenever the polyhedron is non-empty.
    """
    m, n = A.shape
    if w is None:
        w = np.zeros(n)
    kmax = min(m, n)
    subsets = [()]
    for k in range(1, kmax + 1):
        subsets.extend(combinations(range(m), k))

    if feasible(A, b, f) and np.allclose(w, 0.0):
        if return_info:
            return f, {"case": 1, "gamma": (), "n_candidates": len(subsets),
                       "n_feasible": 1, "w_scale": 0.0}
        return f

    best, best_d, best_g, best_s, nfeas = None, np.inf, None, None, 0
    for s in w_scales:
        for g in subsets:
            y = proj_gamma(A, b, f, s * w, g)
            if feasible(A, b, y):
                nfeas += 1
                d = float(np.linalg.norm(y - f))
                if d < best_d:
                    best, best_d, best_g, best_s = y, d, g, s
        if best is not None:
            break
    if best is None:                       # recorded, never silently patched
        best, best_g, best_s = f, None, None
    if return_info:
        return best, {"case": 2, "gamma": best_g, "n_candidates": len(subsets),
                      "n_feasible": nfeas, "w_scale": best_s,
                      "found": best_g is not None}
    return best


def caffine_truncated(A, b, f, kmax, w=None):
    """Cardinality-mutation arm: enumerate only sub-constraint sets of size <= kmax."""
    m, n = A.shape
    if w is None:
        w = np.zeros(n)
    subsets = [()]
    for k in range(1, kmax + 1):
        subsets.extend(combinations(range(m), k))
    best, best_d = None, np.inf
    for g in subsets:
        y = proj_gamma(A, b, f, w, g)
        if feasible(A, b, y):
            d = float(np.linalg.norm(y - f))
            if d < best_d:
                best, best_d = y, d
    return f if best is None else best


def hardnet_aff(A, b, f):
    """HardNet-Aff style baseline: one pseudo-inverse correction on the violated rows.

    This is the prior-work operator the paper contrasts against; it assumes the
    selected rows behave like a full-row-rank equality system.
    """
    v = A @ f - b
    idx = np.where(v > TOL)[0]
    if idx.size == 0:
        return f
    Ag, bg = A[idx, :], b[idx]
    return f - pinv(Ag) @ (Ag @ f - bg)


# ---------------------------------------------------------------- generators
def random_constraints(rng, m, n, rank_deficient=False, scale=1.0):
    """Random polyhedron A y <= b guaranteed non-empty (contains the origin)."""
    A = rng.normal(size=(m, n)) * scale
    if rank_deficient and m >= 2:
        # duplicate / negate / rescale rows so rank(A) < m by construction
        A[1] = 2.0 * A[0]
        if m >= 3:
            A[2] = -1.5 * A[0]
    b = np.abs(rng.normal(size=m)) + 0.25   # origin strictly feasible
    return A, b


def feasible_point(A, b, rng, on_boundary=False):
    """A feasible target f_t; optionally sitting exactly on an active constraint."""
    n = A.shape[1]
    y = np.zeros(n)
    for _ in range(64):
        cand = rng.normal(size=n) * 0.5
        if feasible(A, b, cand):
            y = cand
            break
    if on_boundary:
        k = int(rng.integers(A.shape[0]))
        a = A[k]
        slack = b[k] - a @ y
        y = y + slack * a / (a @ a + 1e-12)
        if not feasible(A, b, y):
            y = np.zeros(n)
    return y
