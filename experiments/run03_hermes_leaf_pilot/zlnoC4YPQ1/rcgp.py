"""rcgp.py — from-scratch CPU implementation of GP / RCGP (P-IMQ) posteriors and the
BO algorithms of "Robust Bayesian Optimisation with Unbounded Corruptions" (arXiv 2511.15315).

Everything follows notes_paper.md:
  - RCGP posterior: Eqs (5)-(6), J_w = diag(sn^2/(2 w_i^2)), m_w = [sn^2 d/dy log w_i^2]
  - P-IMQ weight: Definition 3.1, W = sn/sqrt(2)
  - Psi(n) = sqrt(1 + (n kappa/sn^2)(1 + n kappa/sn^2))   (Sec 4.1, Lemma D.11)
Numpy only.
"""
import numpy as np

# ---------------------------------------------------------------- kernel
def rbf(X1, X2, lengthscale=0.1, outputscale=1.0):
    X1 = np.atleast_2d(np.asarray(X1, float).reshape(-1, 1))
    X2 = np.atleast_2d(np.asarray(X2, float).reshape(-1, 1))
    d2 = (X1 - X2.T) ** 2
    return outputscale * np.exp(-0.5 * d2 / lengthscale ** 2)


def psi(n_c, kappa=1.0, sn2=1.0):
    """Psi(T_c) from Sec 4.1 (Lemma D.11). Psi(0) = 1."""
    z = n_c * kappa / sn2
    return np.sqrt(1.0 + z * (1.0 + z))


# ---------------------------------------------------------------- P-IMQ weights
def pimq(y, g, L, c, sn):
    """Definition 3.1. Returns (w, m_w_factor) where m_w = sn^2 * dlog(w^2)/dy."""
    y = np.asarray(y, float)
    g = np.asarray(g, float) * np.ones_like(y)
    W = sn / np.sqrt(2.0)
    r = y - g
    a = np.abs(r)
    out = a > L
    w = np.full_like(y, W)
    u = np.where(out, a - L, 0.0)
    w[out] = W * (1.0 + (u[out] ** 2) / c ** 2) ** -0.5
    # d/dy log w^2 = -2u/(c^2 + u^2) * sign(r)  outside plateau, 0 inside
    dlog = np.zeros_like(y)
    dlog[out] = -2.0 * u[out] / (c ** 2 + u[out] ** 2) * np.sign(r[out])
    return w, sn ** 2 * dlog


def posterior(Xtr, ytr, Xte, sn, kpar, weights=None, mw=None):
    """RCGP posterior (Eqs 5-6). weights=None -> standard GP (J_w = I, m_w = 0)."""
    Xtr = np.asarray(Xtr, float).ravel()
    ytr = np.asarray(ytr, float).ravel()
    n = len(ytr)
    K = rbf(Xtr, Xtr, **kpar)
    if weights is None:
        Jw = np.ones(n)
        mw = np.zeros(n)
    else:
        W = sn / np.sqrt(2.0)
        Jw = sn ** 2 / (2.0 * weights ** 2)
        Jw = Jw / (sn ** 2 / (2.0 * W ** 2))  # normalise so w=W gives Jw=1 exactly
    A = K + sn ** 2 * np.diag(Jw) + 1e-10 * np.eye(n)
    ks = rbf(Xtr, Xte, **kpar)
    Ainv_y = np.linalg.solve(A, ytr - mw)
    mu = ks.T @ Ainv_y
    v = np.linalg.solve(A, ks)
    var = kpar.get("outputscale", 1.0) - np.sum(ks * v, axis=0)
    return mu, np.sqrt(np.maximum(var, 1e-12))


# ---------------------------------------------------------------- objective
def forrester(x):
    x = np.asarray(x, float)
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def objective(x):
    """Maximisation target: f = -forrester on [0,1]. Optimum ~ 6.0207 at x* ~ 0.7572."""
    return -forrester(x)


GRID = np.linspace(0.0, 1.0, 201)
XSTAR = GRID[int(np.argmax(objective(GRID)))]
FSTAR = float(np.max(objective(GRID)))


def beta_prime(t, ngrid=len(GRID), delta=0.1):
    """GP-UCB confidence multiplier (Srinivas et al. 2010, finite domain)."""
    return 2.0 * np.log(ngrid * (t ** 2) * np.pi ** 2 / (6.0 * delta))


class Adversary:
    """Sec 5.3 / Appendix I.4.2: greedy clairvoyant, frequency budget only."""

    def __init__(self, budget, near=0.1, far=0.4, low=-10.0, high=25.0, xstar=XSTAR):
        self.budget, self.near, self.far = budget, near, far
        self.low, self.high, self.xstar = low, high, xstar
        self.used = 0

    def __call__(self, x, y):
        if self.used >= self.budget:
            return y, False
        d = abs(x - self.xstar)
        if d < self.near:
            self.used += 1
            return self.low, True
        if d > self.far:
            self.used += 1
            return self.high, True
        return y, False
