"""Vectorised CAffine layer for the piecewise benchmark (claim 4).

The constraint matrix A(x) takes only two values (x >= 0 / x < 0), so every
sub-constraint set's pseudo-inverse and Jacobian can be precomputed once and applied to
the whole batch. b(x) still varies per sample and enters linearly.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from caff_core import pinv


class VecCaff:
    def __init__(self, A):
        self.A = A
        m, n = A.shape
        self.subsets = [()] + [c for k in range(1, min(m, n) + 1)
                               for c in combinations(range(m), k)]
        self.J = []
        self.Agp = []
        for g in self.subsets:
            if len(g) == 0:
                self.J.append(np.eye(n))
                self.Agp.append(None)
            else:
                Ag = A[list(g)]
                Agp = pinv(Ag)
                self.J.append(np.eye(n) - Agp @ Ag)
                self.Agp.append(Agp)

    def apply(self, F, B, tol=1e-9):
        """F: (N,n) layer inputs, B: (N,m) offsets. Returns (Y, J_per_sample)."""
        N, n = F.shape
        best_y = F.copy()
        best_J = np.repeat(np.eye(n)[None], N, axis=0)
        best_d = np.full(N, np.inf)
        for gi, g in enumerate(self.subsets):
            J = self.J[gi]
            if len(g) == 0:
                Y = F.copy()
            else:
                Y = F @ J.T + B[:, list(g)] @ self.Agp[gi].T
            feas = np.all(Y @ self.A.T - B <= tol, axis=1)
            d = np.linalg.norm(Y - F, axis=1)
            upd = feas & (d < best_d)
            best_y[upd] = Y[upd]
            best_J[upd] = J
            best_d[upd] = d[upd]
        return best_y, best_J
