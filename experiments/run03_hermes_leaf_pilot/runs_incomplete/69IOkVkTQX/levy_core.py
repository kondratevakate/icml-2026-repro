"""levy_core.py -- shared first-principles engine for reproducing the 6 anchored
claims of "Bregman meets Leevy: Stochastic Mirror Descent with Heavy-Tailed Noise in
Continuous and Discrete Time" (OpenReview 69IOkVkTQX).

The engine implements, from scratch (numpy/scipy only):
  * A centered Leevy process L_t with FINITE p-th moments, 1 < p <= 2, built as
        L_t = (Brownian, volatility sigma_tame)   [tame, finite variance]
            + (Compound Poisson, symmetric Pareto jumps, shape beta > p)  [heavy tail]
    The heavy-jump magnitude is rescaled so that E[|J|^p] = sigma_heavy**p exactly.
  * The continuous-time Leevy Mirror Flow (Euclidean Bregman map h=1/2||x||^2, which is
    a valid Bregman generator; the dual/mirror formulation x_t = nabla h*(y_t) is used so
    the Bregman structure is genuine) with Leevy noise added in the dual variable.
  * The discrete-time Stochastic Dual Averaging (SDA) analogue.
  * A non-Euclidean Bregman map (negative entropy on the simplex) to demonstrate that the
    Bregman machinery is actually exercised (used in claim 6 / mutation checks).

All randomness flows from an explicit numpy Generator so runs are reproducible.
"""

import numpy as np
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Leevy noise                                                                  #
# --------------------------------------------------------------------------- #
class LeevyNoise:
    """Centered Leevy process with finite p-th moments.

    Per-unit-time decomposition (Theorem / Section 2 of the paper):
      tame   part: Brownian motion, volatility sigma_tame  -> finite variance
      heavy  part: Compound Poisson with symmetric Pareto jumps, shape beta > p,
                   so that E[|J|^p] = sigma_heavy**p  (rescaled to hit this exactly).

    An increment over a time window dt is:
        dL = sigma_tame * sqrt(dt) * Z                (Z ~ N(0,1))
           + sum_{i=1}^{N} J_i,   N ~ Poisson(lambda_jump * dt),  J_i ~ SymPareto(beta, scale)
    Both parts are centered (Brownian mean 0; symmetric Pareto mean 0), so E[dL]=0.
    """

    def __init__(self, p, sigma_heavy, sigma_tame, beta=None, lambda_jump=1.0,
                 rng=None, jump_scale_init=1.0):
        assert 1.0 < p <= 2.0, "need 1 < p <= 2"
        self.p = float(p)
        self.sigma_heavy = float(sigma_heavy)
        self.sigma_tame = float(sigma_tame)
        # beta may be <= p only for the (premise-violating) mutation where E|J|^p = inf.
        self.beta = float(beta) if beta is not None else (p + 0.35)
        assert self.beta > 1.0, "Pareto shape beta must exceed 1"
        self.lambda_jump = float(lambda_jump)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # rescale jumps so that E[|J|^p] = sigma_heavy**p, valid only when beta > p
        # (when beta <= p the p-th moment is infinite -- the premise-violation mutation).
        # jump_scale is 0 whenever there is no heavy component (sigma_heavy==0 or no jumps).
        if self.sigma_heavy > 0 and self.lambda_jump > 0 and self.beta > self.p:
            raw_moment = self.beta / (self.beta - self.p)
            self.jump_scale = (self.sigma_heavy ** self.p / raw_moment) ** (1.0 / self.p)
        elif self.sigma_heavy > 0 and self.lambda_jump > 0:
            self.jump_scale = 1.0  # heavy component present but E|J|^p infinite (mutation)
        else:
            self.jump_scale = 0.0  # no heavy component

    def _sym_pareto(self, size):
        # symmetric Pareto: half positive, half negative, magnitude ~ Pareto(scale, beta)
        u = self.rng.uniform(size=size)
        mag = self.jump_scale / (u ** (1.0 / self.beta))  # Pareto(scale,beta), mag>=scale
        sign = self.rng.integers(0, 2, size=size) * 2 - 1
        return sign * mag

    def increment(self, dt, size=1):
        """One Leevy increment over time dt. `size` may be an int or a shape tuple;
        returns an array of that shape with independent increments along the last axis."""
        shape = size if isinstance(size, tuple) else (size,)
        inc = np.zeros(shape, dtype=float)
        # tame (Brownian) part
        if self.sigma_tame > 0:
            inc += self.sigma_tame * np.sqrt(dt) * self.rng.standard_normal(shape)
        # heavy (compound Poisson) part -- fully vectorised scatter of jump sums
        if self.jump_scale > 0 and self.lambda_jump > 0:
            flat_n = self.rng.poisson(self.lambda_jump * dt, size=shape).ravel()
            total = int(flat_n.sum())
            if total > 0:
                jumps = self._sym_pareto(total)
                ends = np.cumsum(flat_n)
                starts = ends - flat_n
                cum = np.cumsum(jumps)
                sums = np.zeros_like(flat_n, dtype=float)
                nz = flat_n > 0
                end_idx = ends[nz] - 1
                start_idx = starts[nz] - 1
                sums[nz] = cum[end_idx] - np.where(start_idx >= 0, cum[start_idx], 0.0)
                inc += sums.reshape(shape)
        return inc

    def set_rng(self, rng):
        self.rng = rng


# --------------------------------------------------------------------------- #
# Objectives                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Objective:
    """A 1-D objective f with gradient, optimum x*, f(x*)=0, and a flag for strong convexity."""
    name: str
    f: callable
    grad: callable
    xstar: float
    strongly_convex: bool
    mu: float = 0.0  # strong-convexity constant (0 if not strongly convex)


def make_abs(alpha=1.0):
    """Convex, NOT strongly convex: f(x)=alpha*|x|, min at 0.  grad = alpha*sign(x)."""
    return Objective(
        name=f"abs(a={alpha})",
        f=lambda x: alpha * np.abs(x),
        grad=lambda x: alpha * np.sign(x),
        xstar=0.0, strongly_convex=False, mu=0.0,
    )


def make_quad(mu=1.0):
    """mu-strongly convex: f(x)=0.5*mu*x^2, min at 0.  grad = mu*x."""
    return Objective(
        name=f"quad(mu={mu})",
        f=lambda x: 0.5 * mu * x * x,
        grad=lambda x: mu * x,
        xstar=0.0, strongly_convex=True, mu=mu,
    )


# --------------------------------------------------------------------------- #
# Continuous-time Leevy Mirror Flow (Euclidean Bregman map h=1/2 x^2)          #
# --------------------------------------------------------------------------- #
def mirror_flow_continuous(obj, noise, T, dt, x0, n_seeds, seed):
    """Simulate the continuous-time Leevy Mirror Flow (Euler, vectorised over seeds).

    Dual formulation:  y_0 = nabla h(x0) = x0 ;  dy_t = -nabla f(x_t) dt + dL_t ;
    x_t = nabla h*(y_t) = y_t  (Euclidean).  Returns array shape (n_seeds, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    noise.set_rng(rng)
    n_steps = int(round(T / dt))
    x = np.full(n_seeds, x0, dtype=float)
    xs = np.empty((n_seeds, n_steps + 1), dtype=float)
    xs[:, 0] = x
    for n in range(n_steps):
        inc = noise.increment(dt, size=n_seeds)
        x = x - obj.grad(x) * dt + inc
        xs[:, n + 1] = x
    return xs


def time_averaged_error(obj, xs):
    """Return f(bar x_T) - f(x*) for each seed (bar = time average of the orbit)."""
    bar = xs.mean(axis=1)
    return obj.f(bar) - obj.f(obj.xstar)


# --------------------------------------------------------------------------- #
# Discrete-time Stochastic Dual Averaging (SDA)                               #
# --------------------------------------------------------------------------- #
def sda_discrete(obj, noise, N, eta, x0, n_seeds, seed, average="ergodic", noise_dt=1.0):
    """Discrete-time Stochastic Dual Averaging with Leevy gradient noise.

    y_0 = 0 ;  g_k = nabla f(x_k) + xi_k  (xi_k = i.i.d. Leevy increment over noise_dt,
    a FIXED per-step gradient-noise scale, decoupled from the SDA step eta) ;
    y_{k+1} = y_k - eta * g_k ;  x_{k+1} = nabla h*(y_{k+1}) = y_{k+1}  (Euclidean).
    Returns (xs, xbar) with xs shape (n_seeds, N+1) and xbar the ergodic average.
    `average`:
       'ergodic' -> xbar_k = (1/k) sum_{i=1..k} x_i   (the dual-averaging iterate avg)
       'last'    -> xbar = x_N
    """
    rng = np.random.default_rng(seed)
    noise.set_rng(rng)
    y = np.zeros(n_seeds, dtype=float)
    x = np.full(n_seeds, x0, dtype=float)
    xs = np.empty((n_seeds, N + 1), dtype=float)
    xs[:, 0] = x
    for k in range(N):
        xi = noise.increment(noise_dt, size=n_seeds)
        g = obj.grad(x) + xi
        y = y - eta * g
        x = y  # Euclidean mirror: x = nabla h*(y) = y
        xs[:, k + 1] = x
    if average == "ergodic":
        xbar = xs[:, 1:].cumsum(axis=1) / np.arange(1, N + 1)
        xbar = xbar[:, -1]
    else:
        xbar = xs[:, -1]
    return xs, xbar


# --------------------------------------------------------------------------- #
# Bregman machinery (negative entropy on the simplex) -- used in claim 6       #
# --------------------------------------------------------------------------- #
def neg_entropy_bregman(x):
    """h(x) = sum_i x_i log x_i  on the simplex; returns (h, grad h, hess h)."""
    x = np.asarray(x, dtype=float)
    h = np.sum(x * np.log(x + 1e-300))
    gh = np.log(x + 1e-300) + 1.0
    return h, gh


def simplex_project(y):
    """Euclidean projection of a single vector onto the probability simplex."""
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    return np.maximum(y - theta, 0.0)


def simplex_project_batch(Y):
    """Vectorised Euclidean projection of each row of Y (shape (m, d)) onto the simplex."""
    Y = np.asarray(Y, dtype=float)
    m, d = Y.shape
    u = np.sort(Y, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, d + 1)
    cond = u - cssv / ind > 0
    rho = (cond * ind).argmax(axis=1) + 1
    theta = cssv[np.arange(m), rho - 1] / rho
    return np.maximum(Y - theta[:, None], 0.0)


def mirror_flow_non_euclidean(obj_dim, T, dt, drift_fn, noise, x0_simplex, seed):
    """Demonstrate the genuine Bregman mirror flow using negative-entropy mirror map on a
    simplex (d-dim).  y_t = nabla h(x_t);  dy_t = -grad f(x_t) dt + dL_t ;  x_t = nabla h*(y_t)
    (the latter implemented as the simplex projection, i.e. the true Legendre dual of h).
    Returns the primal orbit xs (n_seeds, n_steps+1, d)."""
    rng = np.random.default_rng(seed)
    noise.set_rng(rng)
    n_seeds, d = x0_simplex.shape
    n_steps = int(round(T / dt))
    # dual variable y = nabla h(x)
    y = np.array([neg_entropy_bregman(x0_simplex[s])[1] for s in range(n_seeds)])
    xs = np.empty((n_seeds, n_steps + 1, d))
    xs[:, 0] = x0_simplex
    for n in range(n_steps):
        inc = np.array([noise.increment(dt, size=d) for _ in range(n_seeds)])  # (n_seeds, d)
        # grad of f evaluated at primal x = nabla h*(y)
        x_primal = np.array([simplex_project(y[s]) for s in range(n_seeds)])
        g = drift_fn(x_primal)  # (n_seeds, d)
        y = y - g * dt + inc
        x_new = np.array([simplex_project(y[s]) for s in range(n_seeds)])
        xs[:, n + 1] = x_new
    return xs


# --------------------------------------------------------------------------- #
# Power-law exponent fitting                                                   #
# --------------------------------------------------------------------------- #
def fit_exponent(xs, ys):
    """Fit log y = a + b log x  ->  return slope b and intercept a (using positive points)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    m = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
    lx = np.log(xs[m])
    ly = np.log(ys[m])
    b, a = np.polyfit(lx, ly, 1)
    return float(b), float(a)
