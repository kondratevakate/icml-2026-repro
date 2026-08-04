"""Hypergraph container + LP (1) solver + LP-rounding set K (paper Sec.4.1)."""
import numpy as np
from scipy.optimize import linprog


class Hypergraph:
    def __init__(self, hyperedges, weights):
        # hyperedges: list of tuples of vertex ids (ints); weights: list of floats
        self.V = sorted({v for e in hyperedges for v in e})
        self.vid = {v: i for i, v in enumerate(self.V)}  # global id -> index 0..n-1
        self.n = len(self.V)
        self.hyperedges = [tuple(sorted(self.vid[v] for v in e)) for e in hyperedges]
        self.weights = [float(w) for w in weights]
        self.m = len(self.hyperedges)
        self.W = float(sum(self.weights))
        # edge->vertex incidence (for LP constraints)
        self.e2v = [list(e) for e in self.hyperedges]

    def induced_weight(self, K):
        """e(K) = total weight of hyperedges fully contained in K (K is a set of vertex indices)."""
        tot = 0.0
        for e, w in zip(self.hyperedges, self.weights):
            if all(v in K for v in e):
                tot += w
        return tot

    def coverage_loss(self, K):
        return self.W - self.induced_weight(K)

    def solve_lp(self, eps):
        """Solve LP (1): minimize sum x_v s.t. sum w_e z_e >= (1-eps)W, z_e <= x_v (v in e),
        0<=x,z<=1. Returns (x_star, z_star, obj)."""
        n, m = self.n, self.m
        nv = n + m
        c = np.concatenate([np.ones(n), np.zeros(m)])  # minimize sum_v x_v only (LP (1))
        rows = []
        for e_idx, e in enumerate(self.hyperedges):
            for v in e:
                row = np.zeros(nv)
                row[n + e_idx] = 1.0
                row[v] = -1.0
                rows.append(row)
        cov = np.zeros(nv)
        for e_idx, w in enumerate(self.weights):
            cov[n + e_idx] = -w
        rows.append(cov)
        A_ub = np.array(rows)
        b_ub = np.zeros(A_ub.shape[0])
        b_ub[-1] = -(1.0 - eps) * self.W
        bounds = [(0.0, 1.0)] * nv
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            raise RuntimeError("LP failed: " + res.message)
        sol = res.x
        x = sol[:n]
        z = sol[n:]
        return x, z, float(res.fun)

    def lp_rounding_set(self, eps, kappa):
        """Return K = {v : x*_v >= rho}, rho = kappa/(1+kappa)."""
        x, z, obj = self.solve_lp(eps)
        rho = kappa / (1.0 + kappa)
        K = set(int(i) for i in range(self.n) if x[i] >= rho - 1e-9)
        return K, x, z, obj, rho
