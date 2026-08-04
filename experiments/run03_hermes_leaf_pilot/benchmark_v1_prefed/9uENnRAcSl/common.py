"""
Shared helpers for reproducing the 6 anchored claims of
"Understanding Behavior Cloning with Action Quantization" (arXiv:2603.20538).

All math is CPU-only numpy/scipy/sympy. No GPU, no network at runtime.

The paper's central objects (from TASK.md / input_bundle.json):
  - Quantizers: binning (||q(u)-u||<=eps_q) and learning-based (avg error eps_q).
  - P-IISS / P-EIISS: stability of dynamics w.r.t. action mismatch (Def 3).
  - RTVC / TVC: smoothed total-variation continuity of the policy (Def 4).
  - Regret J(pi*)-J(pi) as the figure of merit (H * rate).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Moduli (stability gamma, smoothness kappa)
# ---------------------------------------------------------------------------
def gamma_max(r_seq, C=1.0):
    """Max-type P-IISS modulus used in Theorem 3 (deterministic bound):
    gamma((r_t)) = gamma(max_t r_t).  Returns C * max(r)."""
    r = np.asarray(r_seq, dtype=float)
    return C * float(np.max(r)) if r.size else 0.0


def gamma_peiiss(r_seq, C=1.0, eta=0.5):
    """Probabilistic Exponential-IISS modulus (Def 3):
    gamma((r_k)) = sum_{k} C eta^{t-k} r_k  (contractive, eta in (0,1))."""
    r = np.asarray(r_seq, dtype=float)
    t = len(r)
    if t == 0:
        return 0.0
    weights = np.array([C * eta ** (t - 1 - k) for k in range(t)], dtype=float)
    return float(np.dot(weights, r))


def kappa_rtvc(r, delta0):
    """RTVC (relaxed TV continuity) modulus for a BINNING quantizer
    (Prop 3.2(ii)): kappa(r) = 1{r > delta0}, a thresholded (non-Lipschitz)
    modulus that is 0 below the locality scale delta0."""
    return 1.0 if float(r) > float(delta0) else 0.0


def kappa_lipschitz(r, L):
    """Wasserstein / Lipschitz continuity modulus kappa(r)=L*r.
    The paper's Remark after Def 4 shows this is NOT sufficient: Lipschitz
    policies imitating a Lipschitz expert incur EXPONENTIAL compounding."""
    return float(L) * float(r)


# ---------------------------------------------------------------------------
# Quantizers
# ---------------------------------------------------------------------------
def binning_quantizer(u, eps_q, lo=-1.0, hi=1.0):
    """Uniform binning quantizer on [lo,hi] with bin width = 2*eps_q,
    representative = bin midpoint, so ||q(u)-u|| <= eps_q for all u."""
    u = np.asarray(u, dtype=float)
    half = eps_q
    # number of bins so that width <= 2*eps_q
    nb = max(1, int(np.ceil((hi - lo) / (2.0 * eps_q))))
    edges = np.linspace(lo, hi, nb + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.searchsorted(edges, u) - 1, 0, nb - 1)
    q = mid[idx]
    # clamp outside range
    q = np.where(u < lo, lo, q)
    q = np.where(u > hi, hi, q)
    return q


def kmeans_quantizer_fit(u_train, K, seed=0, n_iter=200):
    """Learning-based (k-means / VQ) quantizer: a 'non-smooth' quantizer fit
    to expert actions.  Returns (codebook, assign)."""
    rng = np.random.default_rng(seed)
    u = np.asarray(u_train, dtype=float).reshape(-1, 1)
    N = u.shape[0]
    # init via random pick
    perm = rng.permutation(N)[:K]
    cb = u[perm].copy().reshape(-1)
    for _ in range(n_iter):
        d2 = (u[:, 0][:, None] - cb[None, :]) ** 2
        a = np.argmin(d2, axis=1)
        new_cb = np.array([u[a == k, 0].mean() if np.any(a == k) else cb[k]
                           for k in range(K)])
        if np.allclose(new_cb, cb):
            break
        cb = new_cb
    return cb


def kmeans_quantizer_apply(u, codebook):
    u = np.asarray(u, dtype=float)
    d2 = (u[:, None] - codebook[None, :]) ** 2
    a = np.argmin(d2, axis=1)
    return codebook[a], a


# ---------------------------------------------------------------------------
# 1-D dynamical-system rollout used by claims 2,3,6
# ---------------------------------------------------------------------------
def rollout(x0, policy_action, A, B, H, noise_sigma=0.0, rng=None):
    """x_{h+1} = A x_h + B u_h + noise.  policy_action(x, h) -> u.
    Returns state trajectory x[0..H] (length H+1) and actions u[0..H-1]."""
    x = np.empty(H + 1, dtype=float)
    u = np.empty(H, dtype=float)
    x[0] = x0
    for h in range(H):
        uh = policy_action(x[h], h)
        u[h] = uh
        nx = A * x[h] + B * uh
        if noise_sigma > 0:
            nx += rng.normal(0.0, noise_sigma)
        x[h + 1] = nx
    return x, u
