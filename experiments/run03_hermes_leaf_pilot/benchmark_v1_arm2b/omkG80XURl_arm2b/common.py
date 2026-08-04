"""Common utilities: average-reward MRP with linear function approximation,
double-chain (Eq. 15/16) and single-chain (Eq. 17/18) TD algorithms.
Source: Tian, Chen, Paschalidis, Olshevsky, "Bridging the Gap Between Average and
Discounted TD Learning" (ICML 2026, omkG80XURl / arXiv 2605.02103).
CPU-only, numpy.
"""
import numpy as np


def make_mrp(n=8, d=3, seed=0, feat_scale=None, gap=1.0, normalize=True):
    """Random irreducible aperiodic chain + random feature matrix.
    gap in (0,1]: mixes P towards uniform-ish; smaller gap -> slower mixing."""
    rng = np.random.default_rng(seed)
    P = rng.random((n, n)) ** 2 + 0.05
    P /= P.sum(1, keepdims=True)
    P = (1 - gap) * np.eye(n) + gap * P          # slow down mixing when gap small
    R = rng.normal(size=n)
    Phi = rng.normal(size=(n, d))
    if feat_scale is not None:
        Phi = Phi * feat_scale                    # column conditioning knob
    if normalize:                                 # assumption max_s ||phi(s)|| <= 1
        Phi = Phi / np.max(np.linalg.norm(Phi, axis=1))
    mu = stationary(P)
    return dict(P=P, R=R, Phi=Phi, mu=mu, n=n, d=d)


def stationary(P):
    w, v = np.linalg.eig(P.T)
    i = np.argmin(np.abs(w - 1.0))
    mu = np.real(v[:, i])
    mu = mu / mu.sum()
    return np.maximum(mu, 1e-15)


def eta1(m):
    """Eq. (3)/(8): min_{||x||=1} ||Phi x||^2_Dir + (mu^T Phi x)^2."""
    P, Phi, mu = m['P'], m['Phi'], m['mu']
    D = np.diag(mu)
    M = Phi.T @ D @ (np.eye(len(mu)) - P) @ Phi
    M = 0.5 * (M + M.T)                            # Dirichlet form is symmetric part
    v = Phi.T @ mu
    return float(np.linalg.eigvalsh(M + np.outer(v, v)).min())


def eta3(m):
    """Eq. (2): (min_{||x||=1} x' Phi' D Phi x) * (min_{<y,e>_D=0,||y||_D=1} y' D(I-P) y)."""
    P, Phi, mu = m['P'], m['Phi'], m['mu']
    D = np.diag(mu)
    sigma = float(np.linalg.eigvalsh(Phi.T @ D @ Phi).min())
    A = 0.5 * (D @ (np.eye(len(mu)) - P) + (D @ (np.eye(len(mu)) - P)).T)
    # change of variables y = D^{-1/2} z : ||y||_D=1 <-> ||z||=1 ; <y,e>_D=0 <-> z' sqrt(mu)=0
    Dh = np.diag(np.sqrt(mu))
    Dhi = np.diag(1 / np.sqrt(mu))
    B = Dhi @ A @ Dhi
    u = np.sqrt(mu)                                 # constraint direction
    Q = null_basis(u)
    lam = float(np.linalg.eigvalsh(Q.T @ B @ Q).min())
    return sigma * lam, sigma, lam


def null_basis(u):
    u = u / np.linalg.norm(u)
    n = len(u)
    Q, _ = np.linalg.qr(np.eye(n) - np.outer(u, u))
    # keep components orthogonal to u
    Q = Q[:, np.abs(Q.T @ u) < 1e-8]
    return Q


def expected_system(m):
    """Expected update  E[u(theta)] = b - M theta  for the double-chain algorithm.
    Equivalent to Eq. (14): Phi' D (I - Pi P) Phi theta* = Phi' D Pi R."""
    P, Phi, mu, R = m['P'], m['Phi'], m['mu'], m['R']
    D = np.diag(mu)
    b = Phi.T @ D @ R - np.outer(Phi.T @ mu, mu) @ R
    M = Phi.T @ D @ Phi - Phi.T @ D @ P @ Phi + np.outer(Phi.T @ mu, mu @ Phi)
    return b, M


def theta_star(m):
    b, M = expected_system(m)
    return np.linalg.solve(M, b)


def theta_star_eq14(m):
    P, Phi, mu, R = m['P'], m['Phi'], m['mu'], m['R']
    n = len(mu)
    D = np.diag(mu)
    Pi = np.eye(n) - np.outer(np.ones(n), mu)
    A = Phi.T @ D @ (np.eye(n) - Pi @ P) @ Phi
    rhs = Phi.T @ D @ Pi @ R
    return np.linalg.solve(A, rhs)


def sample_iid(rng, m, T):
    mu, P = m['mu'], m['P']
    n = m['n']
    s = rng.choice(n, size=T, p=mu)
    sh = rng.choice(n, size=T, p=mu)               # independent second chain
    cum = np.cumsum(P, axis=1)
    u = rng.random(T)
    sp = (cum[s] < u[:, None]).sum(1)
    return s, sp, sh


def sample_markov(rng, m, T):
    P, mu, n = m['P'], m['mu'], m['n']
    s = np.empty(T + 1, dtype=int)
    sh = np.empty(T, dtype=int)
    cum = np.cumsum(P, axis=1)
    u1 = rng.random(T)
    u2 = rng.random(T)
    s[0] = rng.choice(n, p=mu)
    x = rng.choice(n, p=mu)
    for t in range(T):
        s[t + 1] = int(np.searchsorted(cum[s[t]], u1[t]))
        sh[t] = x
        x = int(np.searchsorted(cum[x], u2[t]))    # second, independent chain
    return s[:T], s[1:], sh


def run_double(m, T, alpha, seed=0, markov=False, theta0=None, coupled=False):
    """Eq. (15)-(16). coupled=True is the MUTATION: reuse s_t for the hat-chain."""
    rng = np.random.default_rng(seed)
    Phi, R = m['Phi'], m['R']
    s, sp, sh = (sample_markov if markov else sample_iid)(rng, m, T)
    if coupled:
        sh = s
    th = np.zeros(m['d']) if theta0 is None else theta0.copy()
    ts = theta_star(m)
    err = np.empty(T + 1)
    err[0] = np.sum((th - ts) ** 2)
    acc = np.zeros(m['d']); nacc = 0
    seq = np.full(T, alpha) if np.isscalar(alpha) else np.asarray(alpha)
    for t in range(T):
        ph, php, phh = Phi[s[t]], Phi[sp[t]], Phi[sh[t]]
        r = R[s[t]]
        f = -(r + ph @ th) * phh
        g = (r + php @ th - ph @ th) * ph
        th = th + seq[t] * (f + g)
        err[t + 1] = np.sum((th - ts) ** 2)
        if t >= T // 2:
            acc += th
            nacc += 1
    return th, err, acc / max(nacc, 1)


def run_single(m, T, alpha, beta=None, seed=0, markov=True, Rth=None):
    """Eq. (17)-(18) single-chain GTD-style variant."""
    rng = np.random.default_rng(seed)
    Phi, R = m['Phi'], m['R']
    if beta is None:
        beta = alpha
    s, sp, _ = (sample_markov if markov else sample_iid)(rng, m, T)
    th = np.zeros(m['d'])
    w = np.zeros(m['d'])
    ts = theta_star(m)
    if Rth is None:
        Rth = 10 * (np.linalg.norm(ts) + 1)
    err = np.empty(T + 1)
    err[0] = np.sum((th - ts) ** 2)
    for t in range(T):
        ph, php = Phi[s[t]], Phi[sp[t]]
        r = R[s[t]]
        w = w + beta * (ph - w)
        g = (r + php @ th - ph @ th) * ph - (r + ph @ th) * w
        th = th + alpha * g
        nrm = np.linalg.norm(th)
        if nrm > Rth:
            th *= Rth / nrm
        err[t + 1] = np.sum((th - ts) ** 2)
    return th, err


def hitting_time(err_mean, eps):
    idx = np.nonzero(err_mean <= eps)[0]
    return int(idx[0]) if len(idx) else None


def loglog_slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(coef[0]), float(coef[1]), float(ss)
