"""Shared conformal-prediction primitives (paper App. A.4, Algorithm 2 + Algorithm 1)."""
import numpy as np


def conformal_quantile(scores, alpha):
    """Q_hat: the ceil((n+1)(1-alpha))-th smallest score (paper's decreasing-order defn).

    Returns +inf when ceil((n+1)(1-alpha)) > n (alpha too small for the calibration size).
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    if k < 1:
        return 0.0
    return float(s[k - 1])


def alpha_prime(alpha, p):
    """Algorithm 1 line 6."""
    return 1.0 - (1.0 - alpha) / p


def vcp_intervals(mu_cal, y_cal, mu_test, alpha):
    """Vanilla split CP with absolute-residual score. Returns (lo, hi, half_width)."""
    q = conformal_quantile(np.abs(y_cal - mu_cal), alpha)
    return mu_test - q, mu_test + q, q


def pt_vcp(mu_cal, y_cal, mu_test, alpha, p, rng, adjust=True):
    """Algorithm 1 on top of VCP.

    adjust=True  -> paper's alpha' = 1 - (1-alpha)/p   (correct mechanism)
    adjust=False -> MUTATION: use alpha unchanged inside the non-null branch.
    Returns (lo, hi, length, is_null) arrays over the test points.
    """
    a_in = alpha_prime(alpha, p) if adjust else alpha
    q = conformal_quantile(np.abs(y_cal - mu_cal), a_in)
    u = rng.uniform(size=mu_test.shape[0])
    null = u > p
    lo = np.where(null, mu_test, mu_test - q)
    hi = np.where(null, mu_test, mu_test + q)
    length = np.where(null, 0.0, 2.0 * q)
    return lo, hi, length, null


def covered(lo, hi, y):
    return ((y >= lo) & (y <= hi)).astype(float)
