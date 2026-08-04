"""Shared utilities for ugjBMARbyt reproduction (arm2)."""
import json, os, numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)


def save(name, obj):
    p = os.path.join(RES, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print("wrote", p)
    return p


def loglog_slope(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    A = np.vstack([np.log(xs), np.ones_like(xs)]).T
    sl, ic = np.linalg.lstsq(A, np.log(ys), rcond=None)[0]
    return float(sl), float(ic)


def beta_t(V_reg, lam, sigma, Cbar, delta, DLam_op=1.0):
    """Eq. (7) width:  sigma*sqrt(log(4 det(DLam + M M*/lam)/delta^2)) + sqrt(lam)*Cbar/||DLam||_op^{1/2}.
    With Lambda = 0.5||.||^2, DLam = Id, and det(Id + V/lam) = det(V^lam)/lam^d.
    """
    d = V_reg.shape[0]
    sign, logdet = np.linalg.slogdet(V_reg / lam)   # log det(Id + V/lam)
    assert sign > 0
    return sigma * np.sqrt(np.log(4.0 / delta ** 2) + logdet) + np.sqrt(lam) * Cbar / np.sqrt(DLam_op)
