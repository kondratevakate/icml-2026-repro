"""
repro_common.py — shared machinery for reproducing the anchored claims of
"Bridging the Gap Between Average and Discounted TD Learning" (omkG80XURl,
arXiv:2605.02103, ICML 2026).

Implements, from first principles and exactly as written in the paper:
  * MDP construction (ergodic chains with computable stationary dist mu).
  * Exact projected-Bellman fixed point theta* (Eq.14) and relative value W*
    (Eq.5); uniqueness check (Lemma 3.1).
  * Condition numbers eta1, eta2, eta3 (Eqs.1-3,8) with
        ||f||^2_Dir = f^T D (I-P) f   (Dirichlet form, Eq.7).
  * The double-chain algorithm (Eq.15) and single-chain algorithm (Eq.17),
    under i.i.d. or two independent Markov-chain sampling.
  * Helpers to run trajectories, average over fixed seeds (deterministic),
    and fit convergence rates.

All randomness is seed-pinned so results are bit-for-bit reproducible.
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import solve, eigvalsh, eigh, norm
from scipy.linalg import sqrtm


# --------------------------------------------------------------------------
# MDP construction
# --------------------------------------------------------------------------
def build_mdp_mix(n, eps, rng):
    """Uniform-stationary, symmetric chain P=(1-eps)*J + eps*I.
    Eigenvalues: 1 and eps (mult n-1) -> gap = 1-eps, mu = uniform."""
    J = np.ones((n, n)) / n
    I = np.eye(n)
    P = (1.0 - eps) * J + eps * I
    mu = np.full(n, 1.0 / n)
    R = rng.uniform(-1.0, 1.0, size=n)
    return dict(n=n, P=P, mu=mu, R=R, eps=eps, kind="mix")


def build_mdp_birth_death(n, lazy, rng):
    """Symmetric birth-death chain (p_i=q_i=(1-lazy)/2), uniform mu."""
    p = (1.0 - lazy) / 2.0
    P = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            P[i, i - 1] = p
        if i < n - 1:
            P[i, i + 1] = p
        P[i, i] = 1.0 - (P[i, i - 1] if i > 0 else 0.0) - (P[i, i + 1] if i < n - 1 else 0.0)
    mu = np.full(n, 1.0 / n)
    R = rng.uniform(-1.0, 1.0, size=n)
    return dict(n=n, P=P, mu=mu, R=R, lazy=lazy, kind="bd")


def stationary(mdp):
    """Left eigenvector of P for eigenvalue 1 (sanity / cross-check)."""
    vals, vecs = np.linalg.eig(mdp["P"].T)
    i = int(np.argmin(np.abs(vals - 1.0)))
    mu = np.real(vecs[:, i])
    mu = mu / mu.sum()
    return mu


# --------------------------------------------------------------------------
# Exact solutions
# --------------------------------------------------------------------------
def relative_value(P, mu, R):
    """W* solving W* + g e = P W* + R, mu^T W* = 0  (Eq.5)."""
    n = P.shape[0]
    e = np.ones(n)
    g = float(mu @ R)
    A = np.eye(n) - P
    w0 = np.linalg.pinv(A) @ (R - g * e)
    c = -(mu @ w0)
    W = w0 + c * e
    return W, g


def theta_star(Phi, P, mu, R):
    """Unique solution of Phi^T D (I - Pi P) Phi theta = Phi^T D Pi R (Eq.14).
    Returns (theta*, Hessian A) and the condition number of A (uniqueness)."""
    n, d = Phi.shape
    e = np.ones(n)
    D = np.diag(mu)
    Pi = np.eye(n) - np.outer(e, mu)
    A = Phi.T @ D @ (np.eye(n) - Pi @ P) @ Phi
    b = Phi.T @ D @ Pi @ R
    theta = solve(A, b)
    return theta, A, np.linalg.cond(A)


# --------------------------------------------------------------------------
# Condition numbers  (Eqs.1-3,8).  ||f||^2_Dir = f^T D (I-P) f
# --------------------------------------------------------------------------
def eta1(Phi, P, mu):
    """eta1 = min_{||x||=1} x^T Phi^T D(I-P) Phi x + (mu^T Phi x)^2
            = lambda_min( Phi^T D(I-P) Phi + (Phi^T mu)(Phi^T mu)^T )."""
    n, d = Phi.shape
    D = np.diag(mu)
    M = Phi.T @ D @ (np.eye(n) - P) @ Phi
    c = Phi.T @ mu
    S = M + np.outer(c, c)
    return float(eigvalsh(S).min())


def eta2(Phi, P, mu):
    """eta2 = min_{||x||=1, x^T e = 0} x^T Phi^T D(I-P) Phi x."""
    n, d = Phi.shape
    e = np.ones(d)
    D = np.diag(mu)
    M = Phi.T @ D @ (np.eye(n) - P) @ Phi  # symmetric
    w, V = eigh(M)
    # feasible eigenvectors are those orthogonal to e
    feas = [w[i] for i in range(d) if abs(V[:, i] @ e) < 1e-7]
    return float(min(feas)) if feas else float(w.min())


def eta3(Phi, P, mu):
    """eta3 = (min_{||x||=1} x^T Phi^T D Phi x) *
              (min_{<y,e>_D=0, ||y||_D=1} y^T D(I-P) y)."""
    n, d = Phi.shape
    D = np.diag(mu)
    a = float(eigvalsh(Phi.T @ D @ Phi).min())          # first factor
    M = D @ (np.eye(n) - P)
    # Restrict Rayleigh quotient y^T M y / (y^T D y) to y perp_D e.
    # In D-basis: C = D^{-1/2} M D^{-1/2}; its symmetric part Hs has a 0
    # eigenvalue along D^{1/2} e; smallest *positive* eigenvalue = b.
    Din = np.diag(1.0 / np.diag(D))
    Dhalf = sqrtm(Din)                                   # D^{-1/2}
    C = Dhalf @ M @ Dhalf
    Hs = (C + C.T) / 2.0
    w = eigvalsh(Hs)
    w = np.sort(w)
    b = float(w[np.argmax(w > 1e-10)])
    return a * b, a, b


# --------------------------------------------------------------------------
# Feature matrices
# --------------------------------------------------------------------------
def phi_identity(n):
    return np.eye(n)                                     # tabular (paper covers it)


def phi_paper(n, d, mdp, rng):
    """Paper's Appendix-G construction: Phi = [tilde, e, W*] with tilde
    Bernoulli(+/-1); rows normalised so ||phi(s)|| <= 1; full column rank."""
    W, _ = relative_value(mdp["P"], mdp["mu"], mdp["R"])
    for _ in range(50):
        tilde = (rng.random((n, d - 2)) < 0.5).astype(float) * 2.0 - 1.0
        Phi = np.column_stack([tilde, np.ones(n), W])
        rn = np.linalg.norm(Phi, axis=1, keepdims=True)
        Phi = Phi / np.maximum(rn, 1.0)
        if np.linalg.matrix_rank(Phi) == d:
            return Phi
    raise RuntimeError("phi_paper failed full-rank")


def phi_random(n, d, mdp, rng, scale=1.0):
    """Random features with controlled column scale (to vary eta)."""
    Phi = rng.standard_normal((n, d)) * scale
    Phi[:, -1] = 1.0                                     # bias column (like e)
    rn = np.linalg.norm(Phi, axis=1, keepdims=True)
    Phi = Phi / np.maximum(rn, 1.0)
    return Phi


# --------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------
def make_sampler(mdp, kind, rng):
    """kind in {'iid','markov2'}. Returns a function -> (s, shat, sp)."""
    n = mdp["n"]
    P = mdp["P"]
    mu = mdp["mu"]
    if kind == "iid":
        def samp():
            s = rng.choice(n, p=mu)
            sh = rng.choice(n, p=mu)
            sp = rng.choice(n, p=P[s])
            return s, sh, sp
    elif kind == "markov2":
        x = rng.choice(n, p=mu)
        xh = rng.choice(n, p=mu)
        def samp():
            nonlocal x, xh
            xn = rng.choice(n, p=P[x])
            xhn = rng.choice(n, p=P[xh])
            sp = rng.choice(n, p=P[x])
            x, xh = xn, xhn
            return x, xh, sp
    else:
        raise ValueError(kind)
    return samp


# --------------------------------------------------------------------------
# Algorithms (exactly as in the paper)
# --------------------------------------------------------------------------
def double_chain_step(Phi, theta, s, sh, sp, R, alpha, use_indep=True):
    """Eq.15. If use_indep=False, sh is forced equal to s (mutation: breaks
    the double-sampling independence -> correlated / biased update)."""
    sh = s if not use_indep else sh
    phis = Phi[s]
    phish = Phi[sh]
    phisp = Phi[sp]
    f = -phish * (R[s] + phis @ theta)
    g = phis * (R[s] + phisp @ theta - phis @ theta)
    return theta + alpha * (f + g)


def single_chain_step(Phi, theta, w, s, sp, R, alpha, beta, Rtheta, Rw):
    """Eq.17 (single chain, two-timescale). w -> Phi^T mu; auxiliary estimate
    replaces the independent sample phi(shat)."""
    phis = Phi[s]
    phisp = Phi[sp]
    fw = phis - w
    w = w + beta * fw
    nw = norm(w)
    if nw > Rw:
        w = w * (Rw / nw)
    g = phis * (R[s] + phisp @ theta - phis @ theta) - (R[s] + phis @ theta) * w
    theta = theta + alpha * g
    nt = norm(theta)
    if nt > Rtheta:
        theta = theta * (Rtheta / nt)
    return theta, w


# --------------------------------------------------------------------------
# Vectorised sampling helpers (fast, seed-deterministic)
# --------------------------------------------------------------------------
def _cdf_choice(cdf, u):
    """Batch categorical draw. cdf: (n,) or (N,n) cumulative; u: (N,)."""
    cdf = np.atleast_2d(cdf)
    cdf = cdf.copy()
    cdf[:, -1] = 1.0                      # guard against <1.0 float drift
    return np.clip((u[:, None] > cdf).sum(axis=1), 0, cdf.shape[1] - 1)


def _sample_states(mdp, kind, rng, N):
    """Return (s, sh, sp) arrays of shape (N,) for one step.
    kind in {'iid','markov2'}."""
    n = mdp["n"]
    mu = mdp["mu"]
    P = mdp["P"]
    cdf_mu = np.cumsum(mu)
    if kind == "iid":
        u = rng.random((N, 3))
        s = _cdf_choice(cdf_mu, u[:, 0])
        sh = _cdf_choice(cdf_mu, u[:, 1])
        sp = _cdf_choice(P[s], u[:, 2])
        return s, sh, sp
    else:  # markov2: two independent chains evolving by P
        if not hasattr(rng, "_mc_x"):
            rng._mc_x = _cdf_choice(cdf_mu, rng.random(N))
            rng._mc_xh = _cdf_choice(cdf_mu, rng.random(N))
        cdfP = np.cumsum(P, axis=1)
        u = rng.random((N, 3))
        xn = _cdf_choice(cdfP[rng._mc_x], u[:, 0])
        xhn = _cdf_choice(cdfP[rng._mc_xh], u[:, 1])
        sp = _cdf_choice(cdfP[rng._mc_x], u[:, 2])
        rng._mc_x, rng._mc_xh = xn, xhn
        return rng._mc_x.copy(), rng._mc_xh.copy(), sp


# --------------------------------------------------------------------------
# Trajectory runners (vectorised over N trajectories, deterministic given seed)
# --------------------------------------------------------------------------
def run_double_chain(mdp, Phi, theta0, alpha, T, sampler_kind, seed,
                     use_indep=True, N=24, record=200):
    """Average ||theta_t - theta*||^2 over N trajectories.
    Returns (T_grid, mean_err2)."""
    rng = np.random.default_rng(seed)
    if hasattr(rng, "_mc_x"):
        del rng._mc_x
    theta_star_v, _, _ = theta_star(Phi, mdp["P"], mdp["mu"], mdp["R"])
    n, d = Phi.shape
    theta = np.tile(theta0, (N, 1)).astype(float)
    err2 = np.zeros(T)
    cnt = 0
    for t in range(T):
        s, sh, sp = _sample_states(mdp, sampler_kind, rng, N)
        if not use_indep:
            sh = s
        Phis = Phi[s]        # (N,d)
        Phish = Phi[sh]
        Phisp = Phi[sp]
        pred_s = np.sum(Phis * theta, axis=1, keepdims=True)   # (N,1)
        f = -Phish * (mdp["R"][s].reshape(-1, 1) + pred_s)
        g = Phis * (mdp["R"][s].reshape(-1, 1) + np.sum(Phisp * theta, axis=1, keepdims=True) - pred_s)
        a = alpha(t) if callable(alpha) else alpha
        theta = theta + a * (f + g)
        d = theta - theta_star_v[None, :]
        err2[t] = np.mean(np.sum(d * d, axis=1))
    grid = np.linspace(0, T - 1, record).astype(int)
    return grid, err2[grid]


def run_single_chain(mdp, Phi, theta0, w0, alpha, beta, T, sampler_kind, seed,
                     Rtheta, Rw, N=24, record=200):
    rng = np.random.default_rng(seed)
    if hasattr(rng, "_mc_x"):
        del rng._mc_x
    theta_star_v, _, _ = theta_star(Phi, mdp["P"], mdp["mu"], mdp["R"])
    n, d = Phi.shape
    theta = np.tile(theta0, (N, 1)).astype(float)
    w = np.tile(w0, (N, 1)).astype(float)
    muTPhi = (mdp["mu"] @ Phi)   # (d,) = Phi^T mu
    err2 = np.zeros(T)
    for t in range(T):
        s, sh, sp = _sample_states(mdp, sampler_kind, rng, N)
        Phis = Phi[s]
        Phisp = Phi[sp]
        pred_s = np.sum(Phis * theta, axis=1, keepdims=True)
        fw = Phis - w
        w = w + (beta(t) if callable(beta) else beta) * fw
        wn = np.linalg.norm(w, axis=1, keepdims=True)
        w = np.where(wn > Rw, w * (Rw / wn), w)
        g = Phis * (mdp["R"][s].reshape(-1, 1) + np.sum(Phisp * theta, axis=1, keepdims=True) - pred_s) \
            - pred_s * w
        a = alpha(t) if callable(alpha) else alpha
        theta = theta + a * g
        tn = np.linalg.norm(theta, axis=1, keepdims=True)
        theta = np.where(tn > Rtheta, theta * (Rtheta / tn), theta)
        d = theta - theta_star_v[None, :]
        err2[t] = np.mean(np.sum(d * d, axis=1))
    grid = np.linspace(0, T - 1, record).astype(int)
    return grid, err2[grid]


# --------------------------------------------------------------------------
# Rate / scaling analysis
# --------------------------------------------------------------------------
def fit_decay_rate(T_grid, err2):
    """Fit log(err2) ~ slope*T + c over the window where it clearly decays.
    Returns slope (negative) and the achieved decay rate c = -slope."""
    e = err2.copy()
    e[e <= 0] = 1e-30
    le = np.log(e)
    # window: from 20% to 80% of the trajectory, where signal is present
    lo, hi = max(1, len(e) // 5), max(2, 4 * len(e) // 5)
    t = T_grid[lo:hi].astype(float)
    y = le[lo:hi]
    if len(t) < 3:
        return float("nan"), float("nan")
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return -slope, float(np.exp(le[-1]))


def t_to_eps(T_grid, err2, eps):
    """Smallest T such that err2 <= eps (linear interp); inf if never."""
    idx = np.where(err2 <= eps)[0]
    if len(idx) == 0:
        return float("inf")
    return float(T_grid[idx[0]])


def fit_scaling_exponent(etas, ys):
    """Log-log slope of ys vs etas. Returns slope (and r^2)."""
    le = np.log(np.asarray(etas, float))
    ly = np.log(np.asarray(ys, float))
    A = np.vstack([le, np.ones_like(le)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    pred = slope * le + intercept
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(r2)
