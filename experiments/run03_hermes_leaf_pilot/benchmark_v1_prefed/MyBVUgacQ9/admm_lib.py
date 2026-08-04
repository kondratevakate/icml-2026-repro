"""admm_lib.py -- exact-block ADMM (Algorithm 1 of arXiv:2603.11919) for
multi-affine quadratic equality constrained problems (eq. 1-3).

Problem:
    min_{x,z}  sum_i [ (mu_i/2)||x_i||^2 + I_{X_i}(x_i) ]  +  (mu_z/2)||z||^2
    s.t.       A(x) + Q z = 0,      (A(x))_j = 0.5 x^T C_j x + d_j^T x + e_j

with C_j symmetric and zero diagonal blocks (Definition 2.1), so that A is affine in
each block when the others are fixed.  Every subproblem is solved EXACTLY (closed form,
or exhaustive active-set enumeration for small boxes), as Theorems 3.1-3.3 assume.

Only numpy is used.  CPU only.
"""
import numpy as np
import itertools


class MultiAffineProblem:
    def __init__(self, blocks, C, d, e, Q, mu_x, mu_z, box=None):
        """blocks: list of index arrays partitioning x.
        C: (nc, nx, nx) symmetric, zero diagonal blocks.  d: (nc, nx).  e: (nc,).
        Q: (nc, nz).  mu_x: (nx,) diagonal quadratic weights.  mu_z: (nz,).
        box: None, or list (per block) of (lo, hi) arrays or None -> polyhedral I_i.
        """
        self.blocks = [np.asarray(b, dtype=int) for b in blocks]
        self.C = np.asarray(C, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.e = np.asarray(e, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.mu_x = np.asarray(mu_x, dtype=float)
        self.mu_z = np.asarray(mu_z, dtype=float)
        self.nx = self.C.shape[1]
        self.nc = self.C.shape[0]
        self.nz = self.Q.shape[1]
        self.box = box
        # sanity: zero diagonal blocks (multi-affinity)
        for j in range(self.nc):
            for b in self.blocks:
                assert np.allclose(self.C[j][np.ix_(b, b)], 0.0), \
                    "C_%d has nonzero diagonal block -> not multi-affine" % j
            assert np.allclose(self.C[j], self.C[j].T), "C_%d not symmetric" % j

    # ---- problem functions -------------------------------------------------
    def A(self, x):
        return np.array([0.5 * x @ self.C[j] @ x + self.d[j] @ x + self.e[j]
                         for j in range(self.nc)])

    def f(self, x):
        return 0.5 * np.sum(self.mu_x * x ** 2)

    def phi(self, z):
        return 0.5 * np.sum(self.mu_z * z ** 2)

    def cnorm(self):
        """||C|| := max_j ||C_j||  (spectral norm), Theorem 3.2."""
        return max(np.linalg.norm(self.C[j], 2) for j in range(self.nc))

    def lagrangian(self, x, z, w, rho):
        g = self.A(x) + self.Q @ z
        return self.f(x) + self.phi(z) + w @ g + 0.5 * rho * g @ g

    def feasible(self, x, tol=1e-9):
        if self.box is None:
            return True
        for b, bx in zip(self.blocks, self.box):
            if bx is None:
                continue
            lo, hi = bx
            if np.any(x[b] < lo - tol) or np.any(x[b] > hi + tol):
                return False
        return True

    # ---- block linearisation ----------------------------------------------
    def _affine_in_block(self, x, b):
        """A(x_b ; x_-b) = M x_b + c   (exact, thanks to multi-affinity)."""
        v = x.copy()
        v[b] = 0.0
        M = np.empty((self.nc, len(b)))
        c = np.empty(self.nc)
        for j in range(self.nc):
            M[j] = (self.C[j] @ v)[b] + self.d[j][b]
            c[j] = 0.5 * v @ self.C[j] @ v + self.d[j] @ v + self.e[j]
        return M, c

    # ---- exact subproblem solves ------------------------------------------
    @staticmethod
    def _solve_box_qp(H, g, lo, hi):
        """min 0.5 y^T H y + g^T y s.t. lo<=y<=hi, H spd, dim small -> exact
        by enumerating active sets (all faces)."""
        n = len(g)
        best, bestval = None, np.inf
        for pattern in itertools.product([0, 1, 2], repeat=n):
            free = [i for i in range(n) if pattern[i] == 0]
            y = np.empty(n)
            for i in range(n):
                if pattern[i] == 1:
                    y[i] = lo[i]
                elif pattern[i] == 2:
                    y[i] = hi[i]
            if free:
                fx = np.array(free)
                fixed = np.array([i for i in range(n) if pattern[i] != 0], dtype=int)
                rhs = -g[fx]
                if len(fixed):
                    rhs = rhs - H[np.ix_(fx, fixed)] @ y[fixed]
                try:
                    y[fx] = np.linalg.solve(H[np.ix_(fx, fx)], rhs)
                except np.linalg.LinAlgError:
                    continue
            if np.any(y < lo - 1e-12) or np.any(y > hi + 1e-12):
                continue
            val = 0.5 * y @ H @ y + g @ y
            if val < bestval:
                bestval, best = val, y.copy()
        assert best is not None
        return np.clip(best, lo, hi)

    def x_block_update(self, x, z, w, rho, i):
        b = self.blocks[i]
        M, c = self._affine_in_block(x, b)
        r = c + self.Q @ z
        H = np.diag(self.mu_x[b]) + rho * (M.T @ M)
        g = M.T @ w + rho * (M.T @ r)
        if self.box is None or self.box[i] is None:
            return np.linalg.solve(H, -g)
        lo, hi = self.box[i]
        lo = np.broadcast_to(np.asarray(lo, float), (len(b),)).copy()
        hi = np.broadcast_to(np.asarray(hi, float), (len(b),)).copy()
        if len(b) <= 8:
            return self._solve_box_qp(H, g, lo, hi)
        # fallback: projected gradient (not used in this study)
        y = np.clip(x[b], lo, hi)
        L = np.linalg.eigvalsh(H).max()
        for _ in range(20000):
            y = np.clip(y - (H @ y + g) / L, lo, hi)
        return y

    def z_update(self, x, w, rho):
        Ax = self.A(x)
        H = np.diag(self.mu_z) + rho * (self.Q.T @ self.Q)
        g = self.Q.T @ w + rho * (self.Q.T @ Ax)
        return np.linalg.solve(H, -g)

    # ---- Algorithm 1 -------------------------------------------------------
    def run(self, x0, z0, w0, rho, iters=400, record=True, store_traj=False):
        x, z, w = np.array(x0, float), np.array(z0, float), np.array(w0, float)
        hist = {"L": [], "res": [], "viol": [], "x": [], "z": [], "w": []}
        for k in range(iters):
            if record:
                hist["L"].append(self.lagrangian(x, z, w, rho))
                g = self.A(x) + self.Q @ z
                hist["viol"].append(float(np.linalg.norm(g)))
                if store_traj:
                    hist["x"].append(x.copy())
                    hist["z"].append(z.copy())
                    hist["w"].append(w.copy())
            xp, zp = x.copy(), z.copy()
            for i in range(len(self.blocks)):
                x[self.blocks[i]] = self.x_block_update(x, z, w, rho, i)
            z = self.z_update(x, w, rho)
            w = w + rho * (self.A(x) + self.Q @ z)
            if record:
                hist["res"].append(float(np.linalg.norm(np.concatenate(
                    [x - xp, z - zp]))))
            if not np.all(np.isfinite(np.concatenate([x, z, w]))):
                break
        for key in ("L", "res", "viol"):
            hist[key] = np.array(hist[key])
        return x, z, w, hist


# --------------------------------------------------------------------------
# rate diagnostics
# --------------------------------------------------------------------------
def contraction_factor(seq, floor=1e-13, min_pts=8):
    """Given a positive decreasing error sequence, estimate the asymptotic
    per-iteration factor c = lim e_{k+1}/e_k from a log-linear fit on the tail
    that stays above `floor` (numerical noise level).  Returns (factor, npts, r2).
    factor << 1  -> linear (geometric) convergence.
    factor -> 1  -> sublinear."""
    e = np.asarray(seq, float)
    idx = np.where((e > floor) & np.isfinite(e) & (e > 0))[0]
    if len(idx) < min_pts:
        return float("nan"), len(idx), float("nan")
    # tail = last 60% of the usable range, skipping the first 20% (transient)
    lo = idx[int(0.35 * len(idx))]
    hi = idx[-1]
    ks = np.arange(lo, hi + 1)
    ok = e[ks] > floor
    ks = ks[ok]
    if len(ks) < min_pts:
        return float("nan"), len(ks), float("nan")
    y = np.log(e[ks])
    p, res = np.polyfit(ks, y, 1, full=True)[:2]
    yhat = np.polyval(p, ks)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(np.exp(p[0])), int(len(ks)), float(r2)


def sublinear_witness(gap):
    """max_k k*gap_k over the tail and its trend -- o(1/k) requires k*gap -> 0."""
    g = np.asarray(gap, float)
    k = np.arange(len(g))
    prod = k * g
    n = len(g)
    tail = prod[int(0.5 * n):]
    return {"max_k_gap_tail": float(np.nanmax(tail)),
            "last_k_gap": float(prod[-1]),
            "k_gap_decreasing": bool(np.nanmax(prod[int(0.75 * n):]) <=
                                     np.nanmax(prod[int(0.5 * n):int(0.75 * n)]) + 1e-18)}


# --------------------------------------------------------------------------
# Section-5 toy problem (Figure 2)
#   min (mu_x/2)(x1^2+x2^2+x3^2+x4^2) + (mu_z/2) z^2
#   s.t. x1 x2 - x3 x4 + q z + 1 = 0
# --------------------------------------------------------------------------
def toy_problem(q, mu_x=1.0, mu_z=1.0, box=None, c_scale=1.0):
    C = np.zeros((1, 4, 4))
    C[0, 0, 1] = C[0, 1, 0] = 1.0 * c_scale
    C[0, 2, 3] = C[0, 3, 2] = -1.0 * c_scale
    d = np.zeros((1, 4))
    e = np.array([1.0])
    Q = np.array([[float(q)]])
    return MultiAffineProblem(blocks=[[0], [1], [2], [3]], C=C, d=d, e=e, Q=Q,
                              mu_x=mu_x * np.ones(4), mu_z=mu_z * np.ones(1),
                              box=box)


# Example 2.2 of the paper
def example22():
    C = np.zeros((2, 2, 2))
    C[0, 0, 1] = C[0, 1, 0] = 1.0
    C[1, 0, 1] = C[1, 1, 0] = -1.0
    d = np.array([[1.0, 0.0], [0.0, 1.0]])
    e = np.array([1.0, 1.0])
    Q = np.eye(2)
    return MultiAffineProblem(blocks=[[0], [1]], C=C, d=d, e=e, Q=Q,
                              mu_x=2.0 * np.ones(2), mu_z=2.0 * np.ones(2))


def rho_threshold(prob, L_phi, mu_phi):
    """Theorem 3.1 threshold:
    rho >= max{4 L_phi^2/(mu_phi lmin+(Q^T Q)), 4 L_phi^2/(mu_phi sqrt(lmin+(Q Q^T)))}"""
    Q = prob.Q
    def lmin_pos(M):
        ev = np.linalg.eigvalsh(M)
        pos = ev[ev > 1e-12]
        return float(pos.min()) if len(pos) else float("nan")
    a = lmin_pos(Q.T @ Q)
    b = lmin_pos(Q @ Q.T)
    return max(4 * L_phi ** 2 / (mu_phi * a), 4 * L_phi ** 2 / (mu_phi * np.sqrt(b)))
