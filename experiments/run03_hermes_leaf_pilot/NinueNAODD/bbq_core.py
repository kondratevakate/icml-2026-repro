"""BBQ / Bayes-BT / Crowd-BT reference implementations.

Source: Aczel, Theis, Wattenhofer, "Efficient Bayesian Inference from Noisy
Pairwise Comparisons", arXiv:2510.09333v2.
  - Eq. (1)  Bradley-Terry
  - Eq. (2)  rater-quality mixture:  q_r * lam_i/(lam_i+lam_j) + (1-q_r)/2
  - Eq. (3)  observed-data log-likelihood
  - Eq. (11) gamma responsibility (E-step)
  - Eq. (12) q_r M-step (Beta prior alpha,beta)
  - Eq. (13) lambda_i M-step (Gamma prior a,b)
  - Eq. (14) ELO = log(skill) * 400
Hyperparameters (Appendix B): a=5, b=0.1, alpha=10, beta=2, converge when no
ELO score changes by more than 1 between iterations.

Everything is plain NumPy, CPU-only. Deterministic given a seed.
"""
import numpy as np

ELO_SCALE = 400.0
PRIOR = dict(a=5.0, b=0.1, alpha=10.0, beta=2.0)


# ---------------------------------------------------------------- data model
class Comparisons:
    """Dense (R,K,K) win-count tensor w[r,i,j] = # times rater r put i above j."""

    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)
        self.R, self.K, _ = self.w.shape

    @classmethod
    def from_triples(cls, triples, R, K):
        w = np.zeros((R, K, K))
        for r, i, j in triples:  # rater r said i beats j
            w[r, i, j] += 1
        return cls(w)

    def n_r(self):
        return self.w.sum(axis=(1, 2))

    def total(self):
        return self.w.sum()


# ------------------------------------------------------------- log-likelihood
def loglik(cmp_, lam, q, eps=1e-300):
    """Observed-data log-likelihood, Eq. (3)."""
    y = lam[:, None] / (lam[:, None] + lam[None, :])   # y[i,j] = P(i beats j)
    p = q[:, None, None] * y[None, :, :] + (1.0 - q)[:, None, None] * 0.5
    return float(np.sum(cmp_.w * np.log(np.maximum(p, eps))))


def logprior(lam, q, a, b, alpha, beta, eps=1e-300):
    lp = np.sum((a - 1) * np.log(np.maximum(lam, eps)) - b * lam)
    qq = np.clip(q, 1e-12, 1 - 1e-12)
    lp += np.sum((alpha - 1) * np.log(qq) + (beta - 1) * np.log(1 - qq))
    return float(lp)


def logpost(cmp_, lam, q, **pr):
    p = dict(PRIOR); p.update(pr)
    return loglik(cmp_, lam, q) + logprior(lam, q, **p)


# ------------------------------------------------------------------ BBQ (EM)
def _gamma_resp(lam, q):
    """Eq. (11). Returns gam[r,i,j]."""
    y = lam[:, None] / (lam[:, None] + lam[None, :])
    num = q[:, None, None] * y[None, :, :]
    return num / (num + (1.0 - q)[:, None, None] * 0.5)


def bbq_em(cmp_, a=5.0, b=0.1, alpha=10.0, beta=2.0, max_iter=500,
           elo_tol=1.0, seed=0, track=False, broken_estep=False,
           fixed_q=None):
    """BBQ EM (Sec 3.3). fixed_q=1.0 recovers Bayes-BT (Caron & Doucet EM).

    broken_estep=True is the MUTATION control: replaces the correct
    responsibility (Eq. 11) with a mis-specified one, which destroys the
    monotonicity guarantee.
    """
    rng = np.random.default_rng(seed)
    R, K = cmp_.R, cmp_.K
    lam = np.exp(rng.normal(0, 0.3, K))            # random positive init
    q = np.full(R, 0.9) if fixed_q is None else np.full(R, float(fixed_q))
    w = cmp_.w
    n_r = cmp_.n_r()
    hist = []
    if track:
        hist.append(logpost(cmp_, lam, q, a=a, b=b, alpha=alpha, beta=beta))

    for it in range(max_iter):
        gam = _gamma_resp(lam, q)
        if broken_estep:                            # deliberate mis-E-step
            gam = np.clip(gam ** 0.35 * 1.4, 0, 1)

        # ---- M-step for q (Eq. 12)
        if fixed_q is None:
            eff = (w * gam).sum(axis=(1, 2))
            q_new = (eff + (alpha - 1)) / (n_r + alpha + beta - 2)
            q_new = np.clip(q_new, 1e-6, 1 - 1e-6)
        else:
            q_new = q

        # ---- M-step for lambda (Eq. 13)
        eff_w = (w * gam).sum(axis=0)               # [i,j] effective wins
        num = eff_w.sum(axis=1) + (a - 1)
        pair = eff_w + eff_w.T                      # symmetric effective count
        denom_mat = pair / (lam[:, None] + lam[None, :])
        np.fill_diagonal(denom_mat, 0.0)
        lam_new = num / (denom_mat.sum(axis=1) + b)

        d_elo = np.max(np.abs(np.log(lam_new) - np.log(lam)) * ELO_SCALE)
        lam, q = lam_new, q_new
        if track:
            hist.append(logpost(cmp_, lam, q, a=a, b=b, alpha=alpha, beta=beta))
        if d_elo < elo_tol:
            break

    out = dict(lam=lam, q=q, iters=it + 1, elo=np.log(lam) * ELO_SCALE)
    if track:
        out["hist"] = np.array(hist)
    return out


def bayes_bt(cmp_, **kw):
    """Bayes-BT: same EM with q fixed to 1 (no rater model)."""
    kw.setdefault("fixed_q", 1.0)
    return bbq_em(cmp_, **kw)


# ------------------------------------------------------------- Crowd-BT (SGD)
def crowd_bt(cmp_, lr=0.02, epochs=60, seed=0, gamma_reg=1.0, eta_init=0.9):
    """Chen et al. (2013) style Crowd-BT: online gradient ascent on
    s_i (item scores, softmax/BT in exp-space) and eta_r (rater accuracy).
    No convergence guarantee -- that is the point of Claim 1.
    """
    rng = np.random.default_rng(seed)
    R, K = cmp_.R, cmp_.K
    s = np.zeros(K)
    eta = np.full(R, eta_init)
    idx = np.argwhere(cmp_.w > 0)
    counts = cmp_.w[cmp_.w > 0]
    order = np.arange(len(idx))
    for ep in range(epochs):
        rng.shuffle(order)
        for t in order:
            r, i, j = idx[t]
            c = counts[t]
            pij = 1.0 / (1.0 + np.exp(-(s[i] - s[j])))
            denom = eta[r] * pij + (1 - eta[r]) * (1 - pij)
            denom = max(denom, 1e-12)
            g = (2 * eta[r] - 1) * pij * (1 - pij) / denom
            s[i] += lr * c * g
            s[j] -= lr * c * g
            ge = (pij - (1 - pij)) / denom
            eta[r] = np.clip(eta[r] + lr * c * ge * 0.1, 1e-3, 1 - 1e-3)
        s -= s.mean()
    return dict(s=s, eta=eta, elo=s * ELO_SCALE / np.log(10) * np.log(10))


# ------------------------------------------------------------------ utilities
def kendall_tau(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x); num = 0.0; nx = 0.0; ny = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            a = np.sign(x[i] - x[j]); b = np.sign(y[i] - y[j])
            num += a * b; nx += a * a; ny += b * b
    return num / np.sqrt(nx * ny) if nx > 0 and ny > 0 else 0.0


def simulate(K=8, R=30, n_per_rater=40, frac_bad=0.3, seed=0,
             lam_true=None):
    """Synthetic data from the generative model of Eq. (2)."""
    rng = np.random.default_rng(seed)
    if lam_true is None:
        lam_true = np.exp(rng.normal(0, 0.8, K))
    q_true = np.where(rng.random(R) < frac_bad, rng.uniform(0.0, 0.3, R),
                      rng.uniform(0.8, 1.0, R))
    triples = []
    for r in range(R):
        for _ in range(n_per_rater):
            i, j = rng.choice(K, 2, replace=False)
            if rng.random() < q_true[r]:
                win_i = rng.random() < lam_true[i] / (lam_true[i] + lam_true[j])
            else:
                win_i = rng.random() < 0.5
            triples.append((r, i, j) if win_i else (r, j, i))
    return Comparisons.from_triples(triples, R, K), lam_true, q_true
