"""
repro_common.py -- shared helpers for reproducing the anchored claims of
"Momentum Further Constrains Sharpness at the Edge of Stochastic Stability"
(OpenReview mL4i6z7Miy / arXiv 2604.14108) from first principles.

All quantities are computed with numpy only (CPU).  The mathematical model is
the paper's own random-quadratic / "Edge of Stochastic Stability" linearization
(see eoss_core.py for the core machinery).  We deliberately AVOID the buggy
Schur-reduction helpers in eoss_core (second_moment_operator_1d / Tslow_1d /
slow_operator / Tslow_theorem41_1d give numerically wrong slow eigenvalues) and
instead verify Theorem 4.1 with (a) the correct full 4x4 second-moment operator
THB_1d / critical_curvature_1d, and (b) a direct dynamical second-moment
ensemble simulation whose growth rate we compare to the SGD(eta_eff) prediction.

Paper predictions reproduced here:
  * vanilla SGD MSS threshold (single):             2/eta
  * SGDM small-batch (noise-dominated) plateau:    2(1-beta)/eta   (= eta_eff rescaling)
  * SGDM large-batch (deterministic) plateau:     2(1+beta)/eta   (= heavy-ball limit)
where eta_eff = eta / (1 - beta)   (Theorem 4.1 effective learning rate).
"""

import numpy as np
from eoss_core import critical_curvature_1d, RandomQuadratic

# ---- global hyperparameters (paper's standard operating point) ----
ETA = 0.1
BETA = 0.9
SEED = 11
ETA_EFF = ETA / (1.0 - BETA)          # Theorem 4.1 effective learning rate
TWO_OVER_ETA = 2.0 / ETA              # = 20.0  (SGD single threshold)
SMALL_PLATEAU = 2.0 * (1.0 - BETA) / ETA   # = 2.0   (SGDM small-batch)
LARGE_PLATEAU = 2.0 * (1.0 + BETA) / ETA   # = 38.0  (SGDM large-batch)

# batch variance s^2/b uses this sigma scale for the curvature-sampling model
SIGMA = 6.0


# ---------------------------------------------------------------------------
# Correct random-quadratic model builder (make_model_spectrum in eoss_core is
# shape-broken, so we build the (N,d) data matrix directly).
# ---------------------------------------------------------------------------
def build_quadratic(eigs, seed=SEED, N=2000):
    rng = np.random.default_rng(seed)
    d = len(eigs)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Hsq = Q @ np.diag(np.sqrt(eigs)) @ Q.T
    A = rng.normal(size=(N, d)) @ Hsq          # rows ~ N(0, Hbar)
    return RandomQuadratic(A, seed=seed)


# ---------------------------------------------------------------------------
# MSS critical curvature (max stable curvature) via the CORRECT full operator.
# ---------------------------------------------------------------------------
def sgdm_critical_curvature(sigma, b, beta=BETA, eta=ETA, n_mc=1200, seed=SEED):
    return float(critical_curvature_1d(sigma, b, eta, beta, n_mc=n_mc,
                                       rng=np.random.default_rng(seed)))


def sgd_critical_curvature(sigma, b, eta=ETA, n_mc=1200, seed=SEED):
    return float(critical_curvature_1d(sigma, b, eta, 0.0, n_mc=n_mc,
                                       rng=np.random.default_rng(seed)))


# ---------------------------------------------------------------------------
# Dynamical second-moment growth rate (Theorem 4.1 verification).
# In the noise-dominated (small-batch) regime the SGDM recursion's variance
# growth rate must equal that of SGD with step eta_eff = eta/(1-beta).
# ---------------------------------------------------------------------------
def sgdm_mss_growth(a, s2, b, eta=ETA, beta=BETA, n_steps=600, n_ens=4000, seed=0):
    rng = np.random.default_rng(seed)
    ee = eta / (1.0 - beta)
    xs = rng.normal(size=n_ens)
    vs = np.zeros(n_ens)
    logs = np.zeros(n_steps)
    for t in range(n_steps):
        h = rng.normal(a, np.sqrt(s2), size=b).mean()   # batch curvature, var s2/b
        v = beta * vs + h * xs
        xs = xs - eta * v
        vs = v
        logs[t] = np.log(np.mean(xs * xs) + 1e-12)
    return float((logs[-1] - logs[0]) / n_steps)


def sgd_mss_growth(step, a, s2, b, n_steps=600, n_ens=4000, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.normal(size=n_ens)
    logs = np.zeros(n_steps)
    for t in range(n_steps):
        h = rng.normal(a, np.sqrt(s2), size=b).mean()
        xs = xs - step * h * xs
        logs[t] = np.log(np.mean(xs * xs) + 1e-12)
    return float((logs[-1] - logs[0]) / n_steps)


def sgdm_operating_plateau_growth(a, s2, b):
    """Predicted MSS growth rate of SGD(eta_eff) used by Theorem 4.1:
    (1 - eta_eff a)^2 + eta_eff^2 * (s2 / b)."""
    ee = ETA_EFF
    return (1.0 - ee * a) ** 2 + ee * ee * (s2 / b)
