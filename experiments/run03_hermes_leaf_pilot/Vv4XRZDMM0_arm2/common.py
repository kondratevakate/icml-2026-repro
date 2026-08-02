"""common.py -- shared DGPs, oracle quantities and conformal machinery for the
reproduction of "Multi-Distribution Robust Conformal Prediction" (arXiv 2601.02998).

All DGP constants are transcribed from Section 5.1 / 5.2 / 5.3 of the paper:
  K = 3 sources, d = 10, alpha = 0.1, Sigma_ij = 0.2 + 0.8*1{i=j},
  |I| = 4 signal coordinates, n_k = 2000, split 37.5 / 12.5 / 50, tau = 2.5,
  classification: C = 6 classes, multinomial f_k(y=c|x) ∝ exp(eta_kc(x)),
      eta_kc(x) = xi_k (b_kc + beta_kc^T x) + 1{c>1} g(x),  g == 0 in "Linear"
      xi_k = 2.5 (1 + 0.25 tau u_k), u_k ~ U[-1,1];  b_kc ~ N(0,(0.4 tau)^2)
      beta_kc = beta_bar_c + tau * Delta_kc, (beta_bar_c)_j ~ N(0,1),
      (Delta_kc)_j ~ N(0,0.15^2) for j in I, 0 otherwise.
  regression: Y = b_k + beta_k^T x + sigma_k eps,  beta_k = beta_bar + 0.2 tau delta_k,
      beta_bar_j ~ N(0,1), (delta_k)_j ~ N(0,1) for j in I;
      b_k = b + tau v_k, b ~ N(0,0.5^2), v_k ~ N(0,0.5^2);
      SNR ~ U[5,10] per run, achieved by adjusting sigma_k^2.

Conformity-score convention (paper Sec. 4.2, eq. (7)):
      s_k(x,y) := -h_lambda(x,y)          (larger score = LESS conforming)
      p^(k)(y) := (1 + #{i in cal_k : s_i >= s(x,y)}) / (n_k + 1)
      C_hat    := { y : max_k p^(k)(y) > alpha }
Equivalently p^(k)(y) = (1 + #{i : h_i <= h(x,y)})/(n_k+1), i.e. higher h = more
conforming, which is what `code_mdcp/model/MDCP.py::aggregated_conformal_set_multi`
implements. We follow the official code's convention exactly.
"""
from __future__ import annotations

import numpy as np

K_SOURCES = 3
D_FEAT = 10
N_SIGNAL = 4
ALPHA = 0.1
TAU = 2.5
N_PER_SOURCE = 2000
N_CLASSES = 6
TRAIN_RATIO, CAL_RATIO = 0.375, 0.125


def sigma_matrix(d: int = D_FEAT) -> np.ndarray:
    """Sigma_ij = 0.2 + 0.8*1{i=j} (Section 5.1)."""
    return 0.2 * np.ones((d, d)) + 0.8 * np.eye(d)


def chol_sigma(d: int = D_FEAT) -> np.ndarray:
    """Cholesky factor, precomputed once (pitfall P1: never call multivariate_normal in a loop)."""
    return np.linalg.cholesky(sigma_matrix(d))


def draw_X(rng: np.random.Generator, n: int, L: np.ndarray) -> np.ndarray:
    return rng.standard_normal((n, L.shape[0])) @ L.T


# --------------------------------------------------------------------------- #
#  Classification DGP (Section 5.2, "Linear": g(x) == 0)
# --------------------------------------------------------------------------- #
class ClassificationDGP:
    def __init__(self, rng: np.random.Generator, tau: float = TAU,
                 K: int = K_SOURCES, C: int = N_CLASSES, d: int = D_FEAT):
        self.K, self.C, self.d, self.tau = K, C, d, tau
        self.I = rng.choice(d, size=N_SIGNAL, replace=False)
        u = rng.uniform(-1.0, 1.0, size=K)
        self.xi = 2.5 * (1.0 + 0.25 * tau * u)                      # (K,)
        self.b = rng.normal(0.0, 0.4 * tau, size=(K, C))            # (K,C)
        beta_bar = np.zeros((C, d))
        beta_bar[:, self.I] = rng.normal(0.0, 1.0, size=(C, N_SIGNAL))
        Delta = np.zeros((K, C, d))
        Delta[:, :, self.I] = rng.normal(0.0, 0.15, size=(K, C, N_SIGNAL))
        self.beta = beta_bar[None, :, :] + tau * Delta              # (K,C,d)

    def cond_prob(self, X: np.ndarray) -> np.ndarray:
        """Oracle f_k(y=c|x). Returns (K, n, C)."""
        eta = self.xi[:, None, None] * (
            self.b[:, None, :] + np.einsum('nd,kcd->knc', X, self.beta))
        eta = eta - eta.max(axis=2, keepdims=True)
        e = np.exp(eta)
        return e / e.sum(axis=2, keepdims=True)

    def sample(self, rng: np.random.Generator, n_per_source: int, L: np.ndarray):
        Xs, Ys = [], []
        for k in range(self.K):
            Xk = draw_X(rng, n_per_source, L)
            pk = self.cond_prob(Xk)[k]                              # (n,C)
            cdf = np.cumsum(pk, axis=1)
            Yk = (rng.random((n_per_source, 1)) > cdf).sum(axis=1)
            Xs.append(Xk)
            Ys.append(np.clip(Yk, 0, self.C - 1))
        return Xs, Ys


# --------------------------------------------------------------------------- #
#  Regression DGP (Section 5.3, "Linear": g(x) == 0)
# --------------------------------------------------------------------------- #
class RegressionDGP:
    def __init__(self, rng: np.random.Generator, tau: float = TAU,
                 K: int = K_SOURCES, d: int = D_FEAT):
        self.K, self.d, self.tau = K, d, tau
        self.I = rng.choice(d, size=N_SIGNAL, replace=False)
        beta_bar = np.zeros(d)
        beta_bar[self.I] = rng.normal(0.0, 1.0, size=N_SIGNAL)
        delta = np.zeros((K, d))
        delta[:, self.I] = rng.normal(0.0, 1.0, size=(K, N_SIGNAL))
        self.beta = beta_bar[None, :] + 0.2 * tau * delta           # (K,d)
        b0 = rng.normal(0.0, 0.5)
        v = rng.normal(0.0, 0.5, size=K)
        self.b = b0 + tau * v                                       # (K,)
        snr = rng.uniform(5.0, 10.0)
        Sig = sigma_matrix(d)
        sig_var = np.einsum('kd,de,ke->k', self.beta, Sig, self.beta) / snr
        self.sigma = np.sqrt(sig_var)                               # (K,)
        self.snr = snr

    def mu(self, X: np.ndarray) -> np.ndarray:
        """Oracle mu_k(x). Returns (K, n)."""
        return self.b[:, None] + self.beta @ X.T

    def cond_pdf(self, X: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        """Oracle f_k(y|x) on a grid. Returns (K, n, m)."""
        mu = self.mu(X)                                             # (K,n)
        z = (y_grid[None, None, :] - mu[:, :, None]) / self.sigma[:, None, None]
        return np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * self.sigma[:, None, None])

    def sample(self, rng: np.random.Generator, n_per_source: int, L: np.ndarray):
        Xs, Ys = [], []
        for k in range(self.K):
            Xk = draw_X(rng, n_per_source, L)
            mu = self.b[k] + Xk @ self.beta[k]
            Ys.append(mu + self.sigma[k] * rng.standard_normal(n_per_source))
            Xs.append(Xk)
        return Xs, Ys


# --------------------------------------------------------------------------- #
#  max-p aggregation (paper eq. (7); mirrors MDCP.py::aggregated_conformal_set_multi)
# --------------------------------------------------------------------------- #
def pvalues_from_scores(cal_h: np.ndarray, test_h: np.ndarray) -> np.ndarray:
    """p(y) = (1 + #{i : cal_h_i <= test_h(y)}) / (n_cal + 1). Higher h = more conforming.

    cal_h : (n_cal,)  h evaluated at the source's calibration pairs
    test_h: (..., m)  h evaluated at (x_test, y) for each candidate y
    """
    order = np.sort(cal_h)
    cnt = np.searchsorted(order, test_h, side='right')
    return (1.0 + cnt) / (order.size + 1.0)


def aggregate(pvals: np.ndarray, alpha: float = ALPHA, how: str = 'max') -> np.ndarray:
    """pvals: (K, ..., m) -> boolean membership mask. `how` selects the mutation arm."""
    if how == 'max':
        agg = pvals.max(axis=0)
    elif how == 'mean':
        agg = pvals.mean(axis=0)
    elif how == 'min':
        agg = pvals.min(axis=0)
    else:
        raise ValueError(how)
    return agg > alpha


def softplus(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    big = z > 30
    out[big] = z[big]
    out[~big] = np.log1p(np.exp(z[~big]))
    return out


def mc_se(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float('nan')
