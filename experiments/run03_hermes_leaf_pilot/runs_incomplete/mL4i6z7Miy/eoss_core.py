"""
eoss_core.py -- Core machinery to reproduce the anchored claims of
"Momentum Further Constrains Sharpness at the Edge of Stochastic Stability"
(OpenReview mL4i6z7Miy / arXiv 2604.14108) from first principles.

All quantities are computed with numpy only (CPU).  The mathematical model is
the paper's own *random quadratic* / "random loss" linearization:

  * Each sample i has a quadratic loss L_i(w) = 1/2 w^T H_i w with an
    interpolating minimizer at the origin (so grad L_i(0)=0, Hessian = H_i).
  * Mini-batch of size b:  H_b = (1/b) sum_{j in B} H_j.
  * Linearized SGDM recursion (paper Eq. 9):
        v_t = beta v_{t-1} + H_b x_{t-1}
        x_t = x_{t-1} - eta v_t
    with x_t = theta_t - theta*.
  * Batch Sharpness (Def. 3.1, paper Eq. 7):
        BS(theta) = E_B[ g_B(theta)^T H_B(theta) g_B(theta) / ||g_B(theta)||^2 ]
    where on the quadratic g_B = H_B x and H_B = H_B, so the per-batch
    quantity is the directional curvature x^T H_B^3 x / x^T H_B^2 x.

Mean-square stability (MSS) is studied through the second-moment operator
(Theorem 4.1 / Appendix D):  z_t = [x_t; v_t], z_t = A_t z_{t-1},
A_t = [[I - eta H_t, -eta beta I], [H_t, beta I]].  MSS <=> rho(E[A_t otimes A_t]) < 1.
The slow (Schur-reduced) operator governs the near-unit modes and reduces, in
the noise-dominated regime, to
        T_slow = I - eta_eff K + eta_eff^2 G ,   eta_eff = eta / (1 - beta),
with K = Hbar otimes I + I otimes Hbar,  G = E[H_t otimes H_t]
(Theorem 4.1, paper Eq. 11).
"""

import numpy as np


# --------------------------------------------------------------------------
# Random-quadratic model
# --------------------------------------------------------------------------
class RandomQuadratic:
    """Quadratic loss L(w)=1/2 sum_i (a_i . w)^2 with per-sample Hessian H_i = a_i a_i^T.

    This is the canonical Edge-of-Stochastic-Stability 'random loss' model:
    gradients vanish at the origin, the full Hessian is Hbar = (1/N) sum a_i a_i^T,
    and every mini-batch Hessian is H_b = (1/b) sum_{i in B} a_i a_i^T.
    The spectrum of Hbar is controlled by the singular/eigen values of A.
    """

    def __init__(self, A, seed=None):
        self.A = np.asarray(A, dtype=float)          # (N, d)
        self.N, self.d = self.A.shape
        self.Hbar = (self.A.T @ self.A) / self.N     # (d, d) full Hessian

    def full_loss(self, w):
        # 1/2 w^T Hbar w
        return 0.5 * w @ self.Hbar @ w

    def batch_grad_hess(self, w, idx):
        Ab = self.A[idx]                              # (b, d)
        proj = Ab @ w                                 # (b,)
        g = (Ab.T @ proj) / len(idx)                 # (d,)  gradient of 1/2 (a_i.w)^2
        H = (Ab.T @ Ab) / len(idx)                   # (d,d) batch Hessian
        return g, H

    def batch_sharpness(self, w, b, n_mc, rng):
        """Def. 3.1 Monte-Carlo estimate of Batch Sharpness at w."""
        vals = []
        for _ in range(n_mc):
            idx = rng.choice(self.N, size=b, replace=False)
            g, H = self.batch_grad_hess(w, idx)
            g2 = float(g @ g)
            if g2 > 1e-14:
                gHg = float(g @ H @ g)
                vals.append(gHg / g2)
        return float(np.mean(vals)) if vals else float('nan')


# --------------------------------------------------------------------------
# 1-D second-moment operator (curvature-sampling model, paper Eq. 25)
# --------------------------------------------------------------------------
def A_t_1d(h, eta, beta):
    """Augmented 2x2 step matrix for scalar curvature h (paper Eq. 25)."""
    return np.array([[1.0 - eta * h, -eta * beta],
                     [h,             beta]])


def THB_1d(a, sigma, b, eta, beta, n_mc=4000, rng=None):
    """E[A_t o A_t] (4x4) for curvature-sampling model in 1-D.

    Per-sample curvature j ~ N(a, sigma^2); batch curvature h = mean of b draws.
    MSS <=> spectral radius of the returned operator < 1.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    acc = np.zeros((4, 4))
    s = sigma / np.sqrt(b)                 # variance of batch-mean curvature
    for _ in range(n_mc):
        h = rng.normal(a, s)
        A = A_t_1d(h, eta, beta)
        acc += np.kron(A, A)
    return acc / n_mc


def rho_THB_1d(a, sigma, b, eta, beta, n_mc=4000, rng=None):
    T = THB_1d(a, sigma, b, eta, beta, n_mc=n_mc, rng=rng)
    return np.max(np.abs(np.linalg.eigvals(T)))


def critical_curvature_1d(sigma, b, eta, beta, n_mc=3000, rng=None,
                          a_max=120.0, tol=2e-3):
    """Largest curvature a for which SGDM is mean-square stable.

    Returns the UPPER stability boundary a* where rho(THB)=1 (the
    'sharpness plateau' the system can tolerate).  Returns 0.0 if the system
    is unstable even at a->0 (too-small-batch / over-noisy regime), and
    a_max if it is still stable at a_max (very flat).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    rho = lambda a: rho_THB_1d(a, sigma, b, eta, beta, n_mc=n_mc, rng=rng)

    # Coarse scan to locate the upper (rho crosses 1 from below to above) boundary.
    grid = np.concatenate([np.linspace(1e-3, 5.0, 40),
                           np.linspace(5.0, a_max, 120)])
    rgrid = np.array([rho(a) for a in grid])
    # find last index where rho<1
    stable = np.where(rgrid < 1.0)[0]
    if len(stable) == 0:
        return 0.0                       # unstable everywhere
    i_top = stable[-1]
    a_lo = grid[i_top]
    a_hi = grid[min(i_top + 1, len(grid) - 1)]
    if a_hi <= a_lo or rgrid[min(i_top + 1, len(grid) - 1)] < 1.0:
        return a_lo                     # stable up to a_max
    for _ in range(50):
        a_mid = 0.5 * (a_lo + a_hi)
        if rho(a_mid) < 1.0:
            a_lo = a_mid
        else:
            a_hi = a_mid
        if a_hi - a_lo < tol:
            break
    return 0.5 * (a_lo + a_hi)


def second_moment_operator_1d(a, sigma, b, eta, beta):
    """Exact 3x3 second-moment operator M (acting on [E[x^2], E[xv], E[v^2]]) for the
    1-D curvature-sampling SGDM model.  MSS <=> rho(M) < 1."""
    a2s2 = a**2 + sigma**2 / b          # E[h_t^2]
    M11 = (1.0 - eta * a)**2
    M12 = np.array([[-2.0 * eta * beta * (1.0 - eta * a), eta**2 * beta**2]])
    M21 = np.array([[a - eta * a2s2],
                    [a2s2]])
    M22 = np.array([[beta * (1.0 - eta * a), 0.0],
                    [2.0 * beta * a, beta**2]])
    return M11, M12, M21, M22


def Tslow_1d(a, sigma, b, eta, beta):
    """Schur-reduced slow operator (paper Appendix D): T_slow = M11 + M12 (I-M22)^{-1} M21.
    In the small-eta / noise-dominated regime this equals 1 - 2*eta_eff*a + eta_eff^2*(a^2+sigma^2/b)
    with eta_eff = eta/(1-beta) (Theorem 4.1 / paper Eq. 11)."""
    M11, M12, M21, M22 = second_moment_operator_1d(a, sigma, b, eta, beta)
    I = np.eye(M22.shape[0])
    return M11 + M12 @ np.linalg.inv(I - M22) @ M21


def Tslow_theorem41_1d(a, sigma, b, eta, beta):
    """Theorem 4.1 closed form (paper Eq. 11) for 1-D: 1 - 2 eta_eff a + eta_eff^2 (a^2 + sigma^2/b)."""
    eta_eff = eta / (1.0 - beta)
    return 1.0 - 2.0 * eta_eff * a + eta_eff**2 * (a**2 + sigma**2 / b)


# --------------------------------------------------------------------------
# d-dimensional operators (Theorem 4.1 / Appendix D)
# --------------------------------------------------------------------------
def A_t_ddim(H, eta, beta):
    d = H.shape[0]
    I = np.eye(d)
    return np.block([[I - eta * H, -eta * beta * I],
                     [H,             beta * I]])


def THB_ddim(H_sampler, b, eta, beta, n_mc=1500, rng=None):
    """E[A_t o A_t] (4d^2 x 4d^2). H_sampler() -> batch Hessian H_b of shape (d,d)."""
    if rng is None:
        rng = np.random.default_rng(0)
    d = None
    acc = None
    for _ in range(n_mc):
        H = H_sampler()
        A = A_t_ddim(H, eta, beta)
        M = np.kron(A, A)
        if acc is None:
            d = H.shape[0]
            acc = np.zeros_like(M)
        acc += M
    return acc / n_mc, d


def slow_operator(Hbar, G, eta, beta):
    """Theorem 4.1 slow operator:  T_slow = I - eta_eff K + eta_eff^2 G."""
    d = Hbar.shape[0]
    I = np.eye(d)
    K = np.kron(Hbar, I) + np.kron(I, Hbar)
    eta_eff = eta / (1.0 - beta)
    return np.eye(d * d) - eta_eff * K + eta_eff ** 2 * G


def rho_slow(Hbar, G, eta, beta):
    T = slow_operator(Hbar, G, eta, beta)
    return np.max(np.abs(np.linalg.eigvals(T)))


# --------------------------------------------------------------------------
# Optimizer steps on the quadratic model (for dynamical / empirical checks)
# --------------------------------------------------------------------------
def sgdm_step(x, v, model, b, eta, beta, rng):
    idx = rng.choice(model.N, size=b, replace=False)
    g, H = model.batch_grad_hess(x, idx)
    v = beta * v + g
    x = x - eta * v
    return x, v


def sgdn_step(x, v, model, b, eta, beta, rng):
    # Nesterov (paper Eq. 6): v_{t+1}=beta v_t + g(theta_t - beta eta v_t); theta+= -eta v
    look = x - beta * eta * v
    idx = rng.choice(model.N, size=b, replace=False)
    g, H = model.batch_grad_hess(look, idx)
    v = beta * v + g
    x = x - eta * v
    return x, v


def sgd_step(x, model, b, eta, rng):
    idx = rng.choice(model.N, size=b, replace=False)
    g, H = model.batch_grad_hess(x, idx)
    x = x - eta * g
    return x, g


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def make_model_spectrum(eigs, seed=0):
    """Build a RandomQuadratic whose full Hessian Hbar has spectrum `eigs`."""
    rng = np.random.default_rng(seed)
    d = len(eigs)
    # random orthogonal basis
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    S = np.diag(eigs)
    Hbar = Q @ S @ Q.T
    # find A with A^T A = Hbar  (A = Hbar^{1/2} R, R random rotation)
    Hsq = Q @ np.diag(np.sqrt(eigs)) @ Q.T
    R, _ = np.linalg.qr(rng.normal(size=(d, d)))
    A = Hsq @ R
    N = 500
    A_big = A + rng.normal(scale=1e-3, size=(N, d))  # (N,d); renders Hbar = (1/N) sum
    # better: make A rows so that (1/N) A^T A = Hbar exactly-ish; use N copies scaling
    return RandomQuadratic(A_big, seed=seed), Hbar
