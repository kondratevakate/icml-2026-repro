"""
cde_core.py — from-first-principles CPU reproduction kernel for
"Efficient Neural Controlled Differential Equations via Attentive Kernel Smoothing"
(OpenReview e6hVbhHEXh, arXiv:2602.02157v2).

Stack: numpy / scipy / sympy only. No GPU, no autodiff, no torch.

This module provides everything needed to count the **Number of Function Evaluations
(NFE)** of a Neural-CDE integration, exactly as defined in the paper (Sec. 3.2):
    NFE = total number of calls to the vector field f_theta during adaptive
    integration (accepted + rejected steps).

We implement our OWN adaptive embedded Runge-Kutta solvers (Bogacki-Shampine 3(2)
and Dormand-Prince 5(4)) so the solver order p is explicit and the NFE counter is
ours, then we cross-check trajectories against scipy.integrate.solve_ivp.

Control paths (Sec. 3.3 / 3.4):
  * linear_spline    — exact linear interpolation (rough: C^0, derivative jumps)
  * cubic_spline     — natural cubic spline (C^2, moderate roughness)
  * gp_smooth        — Gaussian-Process posterior mean, RBF kernel, lengthscale h
  * kernel_smooth    — Nadaraya-Watson kernel regression, RBF kernel, bandwidth h

Vector field f_theta(z) = tanh(z)  (smooth, state-dependent, O(1) magnitude) so the
roughness seen by the solver is that of dX/dt (the control path), exactly the regime
the paper studies ("regardless of the learned vector field, the solver must traverse
the geometric complexity of the driving path", Sec. 2).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve

GLOBAL_SEED = 20260802

# ----------------------------------------------------------------------------
# Adaptive embedded Runge-Kutta solvers (first principles)
# ----------------------------------------------------------------------------
# Each tableau: c (stages), A (lower-tri), b (high order), bhat (embedded), order p

BS23 = {
    # Bogacki-Shampine 3(2)
    "c": np.array([0.0, 1/2, 3/4, 1.0]),
    "A": np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1/2, 0.0, 0.0, 0.0],
        [0.0, 3/4, 0.0, 0.0],
        [2/9, 1/3, 4/9, 0.0],
    ]),
    "b": np.array([2/9, 1/3, 4/9, 0.0]),
    "bhat": np.array([7/24, 1/4, 1/3, 1/8]),
    "order": 3,
}

DP45 = {
    # Dormand-Prince 5(4)
    "c": np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]),
    "A": np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1/5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3/40, 9/40, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44/45, -56/15, 32/9, 0.0, 0.0, 0.0, 0.0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0.0, 0.0, 0.0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0.0, 0.0],
        [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0],
    ]),
    "b": np.array([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0]),
    "bhat": np.array([5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]),
    "order": 5,
}

TABLEAUX = {"BS23": BS23, "DP45": DP45}


def adaptive_rk(func, t0, T, z0, tableau="DP45", tol=1e-4, h0=None,
                max_steps=200000, safety=0.9, fac_max=5.0, fac_min=0.2):
    """Adaptive embedded-RK integrator.

    func(t, z) -> dz/dt  (vector field evaluation == one NFE).
    Returns dict with final state, nfe (function evaluations), n_accept, n_reject.
    NFE counts every func evaluation, including rejected steps (matches
    torchdiffeq's odeint NFE convention used in the paper).
    """
    tab = TABLEAUX[tableau]
    A, b, bhat, c = tab["A"], tab["b"], tab["bhat"], tab["c"]
    p = tab["order"]
    z = np.asarray(z0, dtype=float).copy()
    t = float(t0)
    if h0 is None:
        h0 = (T - t0) / 16.0
    h = float(h0)
    nfe = 0
    n_accept = 0
    n_reject = 0

    # first stage of the very first step
    k = [None] * len(c)
    k[0] = np.asarray(func(t, z), dtype=float)
    nfe += 1

    while t < T - 1e-15 and (n_accept + n_reject) < max_steps:
        if t + h > T:
            h = T - t
        # evaluate stages
        for i in range(1, len(c)):
            ti = t + c[i] * h
            zi = z + h * sum(A[i, j] * k[j] for j in range(i))
            k[i] = np.asarray(func(ti, zi), dtype=float)
            nfe += 1
        z_new = z + h * sum(b[j] * k[j] for j in range(len(c)))
        z_err = z_new - (z + h * sum(bhat[j] * k[j] for j in range(len(c))))
        # scaled error (inf-norm over states)
        scale = tol + tol * np.maximum(np.abs(z), np.abs(z_new))
        err = np.sqrt(np.mean((z_err / scale) ** 2))  # RMS, robust multi-d
        if err <= 1.0:
            t = t + h
            z = z_new
            n_accept += 1
            # carry k[0] to next step (FSAL-like reuse)
            k[0] = np.asarray(func(t, z), dtype=float)
            nfe += 1
            # step-size growth
            if err == 0.0:
                factor = fac_max
            else:
                factor = min(fac_max, max(fac_min, safety * err ** (-1.0 / (p + 1))))
            h = h * factor
        else:
            n_reject += 1
            factor = max(fac_min, safety * err ** (-1.0 / (p + 1)))
            h = h * factor
            # re-evaluate k[0] for the retry at the same t, z
            k[0] = np.asarray(func(t, z), dtype=float)
            nfe += 1
        # safety against zero step
        if h <= 1e-12:
            h = 1e-12
    return {
        "zT": z,
        "nfe": nfe,
        "n_accept": n_accept,
        "n_reject": n_reject,
        "t_end": t,
        "reached_end": t >= T - 1e-12,
    }


# ----------------------------------------------------------------------------
# Vector field (the "neural" part).
# Two regimes are supported:
#   * "identity" : f_theta(z) = I  (returns the identity map). Then the CDE
#                  reduces to z'(t) = dX/dt  ->  z(t) = X(t) + z0, so the NFE is
#                  *purely* the control-path regularity. This is the regime in
#                  which Theorem 3.1 is tightest and isolates exactly the
#                  mechanism the paper identifies as dominant ("regardless of
#                  the learned vector field, the solver must traverse the
#                  geometric complexity of the driving path", Sec. 2).
#   * "tanh"     : f_theta(z) = tanh(z), a smooth state-dependent field that
#                  mimics a small neural readout; used for a robustness check.
# ----------------------------------------------------------------------------
def f_theta(z, kind="identity"):
    if kind == "identity":
        return z  # identity map; combined with dX/dt gives z' = dX/dt
    return np.tanh(z)


# ----------------------------------------------------------------------------
# Control-path constructors. Each returns X(t) and dX/dt (callable).
# ----------------------------------------------------------------------------
def _rbf(t, tv, h):
    t = np.asarray(t, dtype=float).ravel()
    tv = np.asarray(tv, dtype=float).ravel()
    return np.exp(-((t[:, None] - tv[None, :]) ** 2) / (2.0 * h * h))


def linear_spline(obs_t, obs_x):
    obs_t = np.asarray(obs_t, dtype=float)
    obs_x = np.asarray(obs_x, dtype=float)
    # slope per segment
    dt = np.diff(obs_t)
    slopes = np.diff(obs_x, axis=0) / dt[:, None]

    def X(t):
        t = np.asarray(t, dtype=float)
        idx = np.clip(np.searchsorted(obs_t, t, side="right") - 1, 0, len(obs_t) - 2)
        return obs_x[idx] + (t - obs_t[idx])[:, None] * slopes[idx]

    def Xp(t):
        t = np.asarray(t, dtype=float)
        idx = np.clip(np.searchsorted(obs_t, t, side="right") - 1, 0, len(obs_t) - 2)
        return slopes[idx]

    return X, Xp


def cubic_spline(obs_t, obs_x):
    obs_t = np.asarray(obs_t, dtype=float)
    obs_x = np.asarray(obs_x, dtype=float)
    cs = CubicSpline(obs_t, obs_x, bc_type="natural")
    X = cs
    Xp = cs.derivative(1)
    return X, Xp


def gp_smooth(obs_t, obs_x, h, sigma2=1e-6):
    """GP posterior mean with RBF kernel, lengthscale h, observation noise sigma2."""
    obs_t = np.asarray(obs_t, dtype=float)
    obs_x = np.asarray(obs_x, dtype=float)
    N = len(obs_t)
    K = _rbf(obs_t[:, None], obs_t[None, :], h) + sigma2 * np.eye(N) + 1e-7 * np.eye(N)
    # Robust solve (min-norm) — stable even when the RBF Gram is near-singular
    # at very small lengthscales (path approaches exact interpolation).
    Kinv_X = np.linalg.lstsq(K, obs_x, rcond=None)[0]

    def X(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        k = _rbf(t, obs_t, h)            # (M, N)
        out = k @ Kinv_X
        return out[0] if out.shape[0] == 1 else out

    def Xp(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        dk = -((t[:, None] - obs_t[None, :]) / (h * h)) * _rbf(t, obs_t, h)
        out = dk @ Kinv_X
        return out[0] if out.shape[0] == 1 else out

    return X, Xp


def kernel_smooth(obs_t, obs_x, h):
    """Nadaraya-Watson kernel regression with Gaussian RBF, bandwidth h."""
    obs_t = np.asarray(obs_t, dtype=float)
    obs_x = np.asarray(obs_x, dtype=float)

    def X(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        k = _rbf(t, obs_t, h)            # (M, N)
        w = k / k.sum(axis=1, keepdims=True)
        out = w @ obs_x
        return out[0] if out.shape[0] == 1 else out

    def Xp(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        k = _rbf(t, obs_t, h)
        dk = -((t[:, None] - obs_t[None, :]) / (h * h)) * k
        denom = k.sum(axis=1, keepdims=True)
        # derivative of w = k/sum(k): (dk*sum - k*sum(dk))/sum^2
        ddenom = dk.sum(axis=1, keepdims=True)
        dw = (dk * denom - k * ddenom) / (denom ** 2)
        out = dw @ obs_x
        return out[0] if out.shape[0] == 1 and np.ndim(t) == 1 and t.shape[0] == 1 else out

    return X, Xp


def fixed_rk(func, t0, T, z0, tableau="DP45", n_steps=200):
    """Non-adaptive fixed-step embedded-RK integrator (mutation probe).

    Uses the same stage evaluations as `adaptive_rk` but with a constant step
    size. NFE == n_steps * (stages-1) + 1 and is INDEPENDENT of the tolerance,
    demonstrating that the Theorem-3.1 tolerance scaling requires adaptivity.
    """
    tab = TABLEAUX[tableau]
    A, b, c = tab["A"], tab["b"], tab["c"]
    z = np.asarray(z0, dtype=float).copy()
    t = float(t0)
    h = (T - t0) / n_steps
    nfe = 0
    k = [None] * len(c)
    k[0] = np.asarray(func(t, z), dtype=float)
    nfe += 1
    for step in range(n_steps):
        for i in range(1, len(c)):
            ti = t + c[i] * h
            zi = z + h * sum(A[i, j] * k[j] for j in range(i))
            k[i] = np.asarray(func(ti, zi), dtype=float)
            nfe += 1
        z = z + h * sum(b[j] * k[j] for j in range(len(c)))
        t = t + h
        k[0] = np.asarray(func(t, z), dtype=float)
        nfe += 1
    return {"zT": z, "nfe": nfe, "n_accept": n_steps, "n_reject": 0,
            "t_end": t, "reached_end": True}


# ----------------------------------------------------------------------------
# CDE integration wrapper: count NFE for a given control path.
# ----------------------------------------------------------------------------
def cde_nfe(obs_t, obs_x, method, tableau="DP45", tol=1e-4, h0=None,
            path_kwargs=None, z0=None, vector_field="identity"):
    """Integrate the Neural CDE z'(t) = f_theta(z(t)) dX/dt, count NFE.

    method: "linear" | "cubic" | "gp" | "kernel"
    vector_field: "identity" (f_theta = I, so z' = dX/dt, NFE is purely the
                  control-path regularity -- the tight regime of Theorem 3.1)
                  or "tanh" (smooth state-dependent f_theta).
    Returns dict with nfe etc.
    """
    path_kwargs = path_kwargs or {}
    if method == "linear":
        X, Xp = linear_spline(obs_t, obs_x)
    elif method == "cubic":
        X, Xp = cubic_spline(obs_t, obs_x)
    elif method == "gp":
        X, Xp = gp_smooth(obs_t, obs_x, **path_kwargs)
    elif method == "kernel":
        X, Xp = kernel_smooth(obs_t, obs_x, **path_kwargs)
    else:
        raise ValueError(method)

    d = obs_x.shape[1]
    if z0 is None:
        z0 = np.zeros(d)

    if vector_field == "identity":
        def func(t, z):
            return Xp(t)  # f_theta = I  ->  z' = dX/dt
    elif vector_field == "tanh":
        def func(t, z):
            return f_theta(z, kind="tanh") * Xp(t)
    else:
        raise ValueError(vector_field)

    res = adaptive_rk(func, obs_t[0], obs_t[-1], z0, tableau=tableau,
                      tol=tol, h0=h0)
    res["method"] = method
    res["path_kwargs"] = path_kwargs
    res["vector_field"] = vector_field
    return res


# ----------------------------------------------------------------------------
# Synthetic data generators that mimic the three UEA datasets' characteristics.
# (Exact NFE reproduction requires the trained multi-view model + the paper's
#  solver tolerance, which are not released; these proxies reproduce the
#  MECHANISM — control-path regularity -> NFE — on representative signals.)
# ----------------------------------------------------------------------------
def make_signal(rng, kind, n_points, d, T, base_freq=1.0, noise=0.0):
    t = np.linspace(0.0, T, n_points)
    # smooth underlying signal: mixture of a few sinusoids per channel
    X = np.zeros((n_points, d))
    for j in range(d):
        ph = rng.uniform(0, 2 * np.pi)
        f = base_freq * (1.0 + 0.5 * j / max(d, 1))
        X[:, j] = np.sin(2 * np.pi * f * t + ph)
        if d > 1:
            X[:, j] += 0.5 * np.sin(2 * np.pi * (2.3 * f) * t + 1.7 * ph)
    if noise > 0:
        X = X + rng.normal(0.0, noise, size=X.shape)
    # normalise to comparable scale
    X = X / (np.std(X) + 1e-9)
    return t, X


def seed_rng(seed=GLOBAL_SEED):
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------------
# Dataset proxies that mimic the three UEA benchmarks used in the paper's
# Table 1 (characteristics: dimension d, #points n, sampling, noise level).
# Exact NFE reproduction requires the trained multi-view model + the paper's
# solver tolerance (not released); these proxies reproduce the MECHANISM
# (control-path regularity -> NFE) on representative signals.
# ----------------------------------------------------------------------------
DATASET_SPECS = {
    # CharacterTrajectories: d=3, ~182 points, high-frequency pen dynamics.
    "CharacterTrajectories": dict(d=3, n=182, T=1.0, base_freq=1.5, noise=0.18, h=0.06),
    # SpokenArabicDigits: d=13, ~93 points, speech-feature coefficients.
    "SpokenArabicDigits": dict(d=13, n=93, T=1.0, base_freq=2.0, noise=0.22, h=0.05),
    # UWaveGestureLibrary: d=3, ~315 points, longer movement sequences.
    "UWaveGestureLibrary": dict(d=3, n=315, T=1.0, base_freq=1.0, noise=0.20, h=0.05),
}


def make_dataset(kind, seed, noise_override=None):
    spec = DATASET_SPECS[kind]
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, spec["T"], int(spec["n"]))
    X = np.zeros((int(spec["n"]), int(spec["d"])))
    for j in range(int(spec["d"])):
        ph = rng.uniform(0, 2 * np.pi)
        f = spec["base_freq"] * (1.0 + 0.4 * j / max(int(spec["d"]), 1))
        X[:, j] = (np.sin(2 * np.pi * f * t + ph)
                   + 0.4 * np.sin(2 * np.pi * 2.7 * f * t + 1.3 * ph))
    noise = spec["noise"] if noise_override is None else noise_override
    X = X + rng.normal(0.0, noise, size=X.shape)
    X = X / (np.std(X) + 1e-9)
    return t, X, spec


def nfe_across_methods(kind, seed, tol=1e-4, n_seeds=5):
    """Mean +/- std NFE for the 4 control-path builders over n_seeds instances."""
    methods = ["linear", "cubic", "gp", "kernel"]
    out = {m: [] for m in methods}
    for s in range(n_seeds):
        t, X, spec = make_dataset(kind, seed + s * 1000)
        for m in methods:
            if m == "gp":
                kw = dict(h=spec["h"], sigma2=1e-4)
            elif m == "kernel":
                kw = dict(h=spec["h"])
            else:
                kw = {}
            r = cde_nfe(t, X, m, tableau="DP45", tol=tol,
                        vector_field="identity", path_kwargs=kw)
            out[m].append(r["nfe"])
    return {m: (float(np.mean(out[m])), float(np.std(out[m]))) for m in methods}, spec
