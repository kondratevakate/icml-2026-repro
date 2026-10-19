"""Simplified functional censored quantile estimators for the FunCQNet core
claim (tau = 0.5).

Two models are compared, exactly mirroring the paper's contrast:

  * Interaction     -- coefficient functions alpha_d(t, Z) are allowed to
                       depend on the scalar covariates Z (like FunCQNet).
  * No Interaction  -- alpha_d(t) is constant in Z (the reduced model).

Both expand alpha_d(t) in a cubic B-spline basis and are fit by IPCW-weighted
median regression (a legitimate censored quantile estimator: Bang & Tsiatis
2002), solved as a linear program. This is NOT the paper's neural network; it
is a transparent CPU estimator whose sole purpose is to test the *structural*
claim -- that a Z-interaction model recovers Z-modulated effects while a
no-interaction model cannot. Because the paper's true alpha_d is linear in Z,
this linear-in-Z estimator can represent it exactly, so the comparison is fair
rather than a strawman.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# functional scores  u_{d,m} = int X_d(t) B_m(t) dt
# ---------------------------------------------------------------------------
def trap_weights(grid: np.ndarray) -> np.ndarray:
    w = np.full(len(grid), grid[1] - grid[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def scores(X: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(n, T) x (T, K) -> (n, K) functional scores against basis B."""
    return (X * w) @ B


# ---------------------------------------------------------------------------
# IPCW weights via Kaplan-Meier estimate of the censoring survival G
# ---------------------------------------------------------------------------
def ipcw_weights(Y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Bang-Tsiatis IPCW: uncensored obs weighted by 1/G(Y-), censored -> 0."""
    order = np.argsort(Y)
    Ys, ds = Y[order], delta[order]
    n = len(Y)
    G = np.ones(n)             # KM estimate of P(C > t) on the sorted grid
    surv = 1.0
    at_risk = n
    for i in range(n):
        # censoring events are delta == 0
        if ds[i] == 0:
            surv *= (at_risk - 1) / at_risk
        G[i] = surv
        at_risk -= 1
    Gu = np.empty(n)
    Gu[order] = G
    Gu = np.clip(Gu, 1e-3, None)
    w = delta / Gu
    return w


# ---------------------------------------------------------------------------
# weighted median (tau=0.5) regression as a linear program
# ---------------------------------------------------------------------------
def wquantile_lp(D: np.ndarray, y: np.ndarray, w: np.ndarray, tau: float = 0.5,
                 l1: float = 0.0, l1_mask: np.ndarray | None = None):
    """min_b sum_i w_i * rho_tau(y_i - D_i b) + l1 * ||mask*b||_1, via LP.

    The L1 term is encoded as extra pseudo-observations (rows e_j scaled by l1,
    target 0, unit weight) so the problem stays a linear program."""
    if l1 > 0:
        p = D.shape[1]
        mask = np.ones(p) if l1_mask is None else l1_mask
        idx = np.where(mask > 0)[0]
        Dp = np.zeros((len(idx), p)); Dp[np.arange(len(idx)), idx] = l1
        D = np.vstack([D, Dp])
        y = np.concatenate([y, np.zeros(len(idx))])
        w = np.concatenate([w, np.ones(len(idx))])
    n, p = D.shape
    # variables: b (p, free) , u>=0 , v>=0   residual = y - Db = u - v
    # objective: tau * w.u + (1-tau) * w.v
    c = np.concatenate([np.zeros(p), tau * w, (1 - tau) * w])
    # equality: D b + u - v = y
    Aeq = np.hstack([D, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=Aeq, b_eq=y, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError("LP failed: " + res.message)
    return res.x[:p]


# ---------------------------------------------------------------------------
# design matrices
# ---------------------------------------------------------------------------
def design_no_interaction(U1, U2):
    n = U1.shape[0]
    return np.hstack([np.ones((n, 1)), U1, U2])


def design_interaction(U1, U2, Z):
    """Interact each functional score with [1, Z1, Z2, Z3]."""
    n = U1.shape[0]
    Zt = np.hstack([np.ones((n, 1)), Z])            # (n, 4)
    def kron_rows(U):
        # (n, K) x (n, 4) -> (n, K*4)
        return (U[:, :, None] * Zt[:, None, :]).reshape(n, -1)
    return np.hstack([np.ones((n, 1)), kron_rows(U1), kron_rows(U2)])


# ---------------------------------------------------------------------------
# recover alpha_hat_d(t, Z) on a t-grid for a set of Z rows
# ---------------------------------------------------------------------------
def alpha_from_no_interaction(coef, B):
    K = B.shape[1]
    b = coef[0]
    c1 = coef[1:1 + K]
    c2 = coef[1 + K:1 + 2 * K]
    a1 = B @ c1                    # (T,)
    a2 = B @ c2
    return b, a1, a2


def alpha_from_interaction(coef, B, Zrows):
    """Return alpha_hat_d(t, Z) shaped (len(Zrows), T)."""
    K = B.shape[1]
    b = coef[0]
    blk = 4 * K
    A1 = coef[1:1 + blk].reshape(K, 4)      # c1_{m,j}
    A2 = coef[1 + blk:1 + 2 * blk].reshape(K, 4)
    Zt = np.hstack([np.ones((Zrows.shape[0], 1)), Zrows])   # (m, 4)
    c1 = Zt @ A1.T                            # (m, K)
    c2 = Zt @ A2.T
    a1 = c1 @ B.T                             # (m, T)
    a2 = c2 @ B.T
    return b, a1, a2
