"""Data-generating process for the FunCQNet simulation study (Appendix D.1).

Faithful to the paper's specification:
  - Two functional predictors X1(t), X2(t) on [0, 1].
      X1(t) = sum_{k=1..K} zeta_k B_k(t),  K = 20 cubic B-spline basis,
              zeta_k ~ N(0, sigma_zeta^2), sigma_zeta^2 = 4.
      X2(t) = | sum_{k=1..K} xi_k U_k phi_k(t) |,
              phi_1 = 1, phi_k(t) = sqrt(2) cos((k-1) pi t) for k > 1,
              xi_k = (-1)^{k+1} k^{-1}, U_k ~ U[-3, 3].
  - Scalar predictors Z1 ~ U(1, 2), Z2, Z3 ~ Bernoulli(0.5).
  - Three scenarios (S1 homoscedastic + interaction, S2 heteroscedastic +
    interaction, S3 heteroscedastic without interaction) as in the paper.
  - Censoring: Y = min(T, C), delta = I(T <= C), with C conditionally
    independent of T. The paper does not fix the law of C, only the target
    censoring rate. We draw C ~ Exp(rate) with the rate calibrated per run so
    that the empirical censoring fraction matches {0, 10, 30, 50}%. This choice
    is documented explicitly (see README, "Deviations from the paper").

All functions of t are evaluated on a common dense grid so the functional
integrals int X_d(t) alpha_d(t, Z) dt are computed by the trapezoidal rule.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


# ---------------------------------------------------------------------------
# Basis systems
# ---------------------------------------------------------------------------
def bspline_basis(grid: np.ndarray, K: int = 20) -> np.ndarray:
    """Cubic B-spline basis with K functions on K-4 equally spaced interior
    knots in [0, 1]. Returns (len(grid), K)."""
    degree = 3
    n_interior = K - 4
    interior = np.linspace(0, 1, n_interior + 2)[1:-1]
    knots = np.concatenate(([0] * (degree + 1), interior, [1] * (degree + 1)))
    B = np.zeros((len(grid), K))
    for k in range(K):
        c = np.zeros(K)
        c[k] = 1.0
        B[:, k] = BSpline(knots, c, degree, extrapolate=False)(grid)
    return np.nan_to_num(B)


def cos_basis(grid: np.ndarray, K: int = 20) -> np.ndarray:
    """phi_1 = 1, phi_k(t) = sqrt(2) cos((k-1) pi t). Returns (len(grid), K)."""
    Phi = np.empty((len(grid), K))
    Phi[:, 0] = 1.0
    for k in range(2, K + 1):
        Phi[:, k - 1] = np.sqrt(2.0) * np.cos((k - 1) * np.pi * grid)
    return Phi


# ---------------------------------------------------------------------------
# True coefficient functions alpha_d(t, Z)
# ---------------------------------------------------------------------------
def psi11(t, Z):  # used by alpha_1 in S1/S2
    Z1, Z2, Z3 = Z[:, 0:1], Z[:, 1:2], Z[:, 2:3]
    return (2.0 / 3.0) * (Z1 * t ** 3 + Z2 * np.exp(-t) + Z3 * np.cos(t))


def psi21(t, Z):
    Z1, Z2, Z3 = Z[:, 0:1], Z[:, 1:2], Z[:, 2:3]
    return (1.0 / 5.0) * (Z1 * t ** 2 + 2 * Z2 * np.cos(3 * np.pi * t) + 3 * Z3 * np.exp(-t))


def psi11_noZ(t):
    return (2.0 / 3.0) * (t ** 3 + np.exp(-t) + np.cos(t))


def psi21_noZ(t):
    return (1.0 / 5.0) * (t ** 2 + 2 * np.cos(3 * np.pi * t) + 3 * np.exp(-t))


def true_alpha(scenario: str, grid: np.ndarray, Z: np.ndarray):
    """Return (alpha1, alpha2) each shaped (n, len(grid)) = alpha_d(t, Z_i)."""
    n = Z.shape[0]
    t = grid[None, :]
    if scenario in ("S1", "S2"):
        a1 = psi11(t, Z)                       # (n, T)
        a2 = psi21(t, Z)                       # tau=0.5 -> F^{-1}(0.5)=0, so psi22 term drops
    elif scenario == "S3":
        a1 = np.tile(psi11_noZ(grid), (n, 1))
        a2 = np.tile(psi21_noZ(grid), (n, 1))
    else:
        raise ValueError(scenario)
    return a1, a2


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample(scenario: str, n: int, cens_rate: float, rng: np.random.Generator,
           n_grid: int = 100, K: int = 20, beta: float = 0.5,
           sigma_eps2: float = 0.01):
    """Generate one simulated dataset. tau is fixed at 0.5 (median), so the
    F^{-1}(tau) recentering term is zero and the heteroscedastic psi22 term in
    S2/S3 vanishes at the median."""
    grid = np.linspace(0, 1, n_grid)
    dt = grid[1] - grid[0]

    # scalar predictors
    Z1 = rng.uniform(1, 2, n)
    Z2 = rng.integers(0, 2, n).astype(float)
    Z3 = rng.integers(0, 2, n).astype(float)
    Z = np.column_stack([Z1, Z2, Z3])

    # functional predictors on the grid
    B = bspline_basis(grid, K)                 # (T, K)
    Phi = cos_basis(grid, K)                   # (T, K)
    zeta = rng.normal(0, 2.0, size=(n, K))     # sigma_zeta = 2 (var 4)
    X1 = zeta @ B.T                            # (n, T)
    xi = np.array([(-1) ** (k + 1) / k for k in range(1, K + 1)])
    U = rng.uniform(-3, 3, size=(n, K))
    X2 = np.abs((U * xi[None, :]) @ Phi.T)     # (n, T)

    # true coefficient functions and the linear predictor (median, tau=0.5)
    a1, a2 = true_alpha(scenario, grid, Z)
    lin = beta + np.trapz(X1 * a1, grid, axis=1) + np.trapz(X2 * a2, grid, axis=1)

    eps = rng.normal(0, np.sqrt(sigma_eps2), n)
    if scenario == "S2":
        # heteroscedastic: variance scaled by a positive functional of X2
        scale = 1.0 + 0.5 * np.trapz(X2, grid, axis=1) / np.trapz(np.ones_like(grid), grid)
        eps = eps * scale
    logT = lin + eps
    T = np.exp(logT)

    # censoring calibrated to the target rate
    if cens_rate <= 0:
        C = np.full(n, np.inf)
    else:
        C = _calibrate_censoring(T, cens_rate, rng)
    Y = np.minimum(T, C)
    delta = (T <= C).astype(float)

    return dict(grid=grid, dt=dt, Z=Z, X1=X1, X2=X2, B=B, Phi=Phi,
                T=T, C=C, Y=Y, delta=delta, logY=np.log(Y),
                cens_rate=float(1 - delta.mean()))


def _calibrate_censoring(T, target, rng, iters=40):
    """Draw C ~ Exp(mean = m*median(T)); bisection on m to hit target rate."""
    medT = np.median(T)
    lo, hi = 1e-3, 1e3
    for _ in range(iters):
        m = np.sqrt(lo * hi)
        C = rng.exponential(m * medT, size=T.shape[0])
        rate = np.mean(T > C)
        if rate > target:
            lo = m            # too much censoring -> larger mean C
        else:
            hi = m
    return rng.exponential(np.sqrt(lo * hi) * medT, size=T.shape[0])
