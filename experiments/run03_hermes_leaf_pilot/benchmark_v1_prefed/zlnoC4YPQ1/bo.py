"""bo.py — the five BO algorithms on a finite grid (CPU, numpy only).

GP-UCB (Srinivas 2010), Student-t Process UCB (Shah et al. 2014), DiagnosticsGP
(Martinez-Cantin 2018 style outlier discarding), FC-RCGP-UCB (Alg 1), A2-RCGP-UCB (Alg 2).
Conventions follow notes_paper.md / Appendix I.1: outcome standardisation, c = 1,
L = 95%-quantile of |y - median(y)| (robust heuristic) or a fixed L, sigma^R used as the
proxy for sigma_uc, T_c estimated as #observations outside the plateau.
"""
import numpy as np
from rcgp import rbf, pimq, posterior, psi, objective, GRID, XSTAR, FSTAR, beta_prime, Adversary

KPAR = dict(lengthscale=0.15, outputscale=1.0)
SIGMA_NOISE_RAW = 1.0    # sigma_noise^2 = 1 (Sec 5.3)


def _std(y):
    m, s = float(np.mean(y)), float(np.std(y))
    return m, (s if s > 1e-8 else 1.0)


def _L_heuristic(ys, fixed_L=None):
    if fixed_L is not None:
        return fixed_L
    med = np.median(ys)
    return float(np.quantile(np.abs(ys - med), 0.95))


def run_bo(method, seed, n_iter, corrupt, n_init=5, fixed_L=None, tc_mode="est",
           adv_kw=None, standardize=True):
    """Returns dict with instantaneous regrets and query points."""
    rng = np.random.default_rng(seed)
    # 5 initial uncorrupted quasi-random (Sobol-like scrambled) points
    x0 = (np.arange(n_init) + rng.uniform(0, 1, n_init)) / n_init
    X = list(x0)
    Y = [float(objective(x) + rng.normal(0, SIGMA_NOISE_RAW)) for x in x0]
    budget = int(np.ceil(n_iter ** (1.0 / 3.0))) if corrupt else 0
    adv = Adversary(budget, **(adv_kw or {})) if corrupt else None
    queries, regrets = [], []

    for t in range(1, n_iter + 1):
        Xa, Ya = np.array(X), np.array(Y)
        if standardize:
            m, s = _std(Ya)
        else:
            m, s = 0.0, 1.0
        ys = (Ya - m) / s
        sn = SIGMA_NOISE_RAW / s
        bp = beta_prime(t)

        if method == "gp_ucb":
            mu, sd = posterior(Xa, ys, GRID, sn, KPAR)
            acq = mu + np.sqrt(bp) * sd
        elif method == "studentt_ucb":
            nu = 5.0
            mu, sd = posterior(Xa, ys, GRID, sn, KPAR)
            n = len(ys)
            K = rbf(Xa, Xa, **KPAR) + sn ** 2 * np.eye(n) + 1e-10 * np.eye(n)
            b2 = float(ys @ np.linalg.solve(K, ys))
            scale = np.sqrt((nu + b2 - 2.0) / (nu + n - 2.0))
            acq = mu + np.sqrt(bp) * scale * sd
        elif method == "diagnostics_gp":
            med = np.median(ys)
            mad = np.median(np.abs(ys - med)) * 1.4826
            keep = np.abs(ys - med) <= 3.0 * max(mad, 1e-6)
            if keep.sum() < 2:
                keep = np.ones_like(ys, bool)
            mu, sd = posterior(Xa[keep], ys[keep], GRID, sn, KPAR)
            acq = mu + np.sqrt(bp) * sd
        elif method in ("fc_rcgp_ucb", "a2_rcgp_ucb"):
            L = _L_heuristic(ys, fixed_L)
            w, mw = pimq(ys, 0.0, L, 1.0, sn)
            mu_a, sd_a = posterior(Xa, ys, GRID, sn, KPAR, weights=w, mw=mw)
            n_out = int(np.sum(np.abs(ys - 0.0) > L))
            tc = n_out if tc_mode == "est" else 0
            if method == "fc_rcgp_ucb":
                mu, sd = mu_a, sd_a
            else:  # A2: acquisition model centred on the anchor posterior mean
                # anchor posterior mean evaluated at the training inputs = adaptive centering g
                g_at_tr = np.interp(Xa, GRID, mu_a)
                L_r = _L_heuristic(ys - g_at_tr, fixed_L)
                # g enters ONLY through the weights / m_w (Eqs 5-6), never by shifting y
                w2, mw2 = pimq(ys, g_at_tr, L_r, 1.0, sn)
                mu, sd = posterior(Xa, ys, GRID, sn, KPAR, weights=w2, mw=mw2)
                n_out = int(np.sum(np.abs(ys - g_at_tr) > L_r))
                tc = n_out if tc_mode == "est" else 0
            Cw = 1.0
            beta_t = (np.sqrt(bp) + Cw * np.sqrt(tc)) ** 2
            acq = mu + np.sqrt(beta_t) * psi(tc, KPAR["outputscale"], sn ** 2) * sd
        else:
            raise ValueError(method)

        xt = float(GRID[int(np.argmax(acq))])
        yt = float(objective(xt) + rng.normal(0, SIGMA_NOISE_RAW))
        if adv is not None:
            yt, _ = adv(xt, yt)
        X.append(xt); Y.append(yt)
        queries.append(xt)
        regrets.append(float(FSTAR - objective(xt)))
    return {"queries": queries, "regrets": regrets,
            "cum_regret": float(np.sum(regrets)), "n_corrupted": (adv.used if adv else 0)}
