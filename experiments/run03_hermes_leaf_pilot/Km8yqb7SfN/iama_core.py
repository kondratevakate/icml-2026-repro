"""iama_core.py — shared numeric substrate for reproducing arXiv 2602.01603 (IAMA / non-linear GRPO).

Finite response space Y = {1..K} with reward vector r (per criterion).
BoN_N over a discrete distribution: R[p] = sum_k r_k (F_k^N - F_{k-1}^N) with rewards sorted
ascending and F the CDF. Functional derivative (discrete analogue of Prop. 4.2):
    dR/dp_j = - sum_{k>=j, k<K} (r_{k+1} - r_k) * N * F_k^(N-1)
(defined up to an additive constant, cf. Definition 4.1).
CPU only, numpy/scipy only.
"""
import numpy as np


def bon_value(p, r, N):
    """Exact E_{y ~ BoN_N[p]}[r(y)] for a discrete distribution p over rewards r."""
    p = np.asarray(p, float)
    r = np.asarray(r, float)
    o = np.argsort(r, kind="stable")
    rs, ps = r[o], p[o]
    F = np.cumsum(ps)
    F = np.clip(F, 0.0, 1.0)
    Fprev = np.concatenate(([0.0], F[:-1]))
    return float(np.sum(rs * (F ** N - Fprev ** N)))


def bon_grad(p, r, N):
    """dR/dp (vector, in the original ordering of r) for R[p] = E_{BoN_N[p]}[r]."""
    p = np.asarray(p, float)
    r = np.asarray(r, float)
    o = np.argsort(r, kind="stable")
    rs, ps = r[o], p[o]
    K = len(rs)
    F = np.clip(np.cumsum(ps), 0.0, 1.0)
    gaps = np.diff(rs)                       # r_{k+1} - r_k, length K-1
    terms = gaps * N * F[:K - 1] ** (N - 1)  # contribution of index k = 0..K-2
    # d_j = - sum_{k >= j, k <= K-2} terms[k]
    tail = np.concatenate((np.cumsum(terms[::-1])[::-1], [0.0]))
    d_sorted = -tail
    d = np.empty(K)
    d[o] = d_sorted
    return d


def linear_value(p, r):
    """Un-transformed expected reward E_p[r] (linear in p) — the 'no inference-time transform' case."""
    return float(np.dot(np.asarray(p, float), np.asarray(r, float)))


def agg_value(p, rewards, Ns, w):
    """R[p] = sum_i w_i R_i[p], R_i = BoN_{N_i} objective for reward r_i (weighted-sum g)."""
    return float(sum(wi * bon_value(p, ri, Ni) for wi, ri, Ni in zip(w, rewards, Ns)))


def agg_grad(p, rewards, Ns, w):
    """dR/dp for the weighted-sum aggregation (chain rule of Section 4)."""
    g = np.zeros(len(p))
    for wi, ri, Ni in zip(w, rewards, Ns):
        g += wi * bon_grad(p, ri, Ni)
    return g


def kl(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))


def loss(p, rewards, Ns, w, beta, p_ref):
    """L[p] = -R[p] + beta KL[p | p_ref]  (Section 5)."""
    return -agg_value(p, rewards, Ns, w) + beta * kl(p, p_ref)


def exact_step(p, d, beta, eta, p_ref):
    """Exact proximal / mirror-descent update of Algorithm 1:
    p_{t+1} = argmin -<d, p> + beta KL[p|p_ref] + (1/eta) KL[p|p_t]
            propto exp( (d + beta log p_ref + (1/eta) log p_t) / (beta + 1/eta) ).
    """
    p = np.asarray(p, float)
    c = beta + 1.0 / eta
    logit = (d + beta * np.log(p_ref) + (1.0 / eta) * np.log(p)) / c
    logit -= logit.max()
    q = np.exp(logit)
    return q / q.sum()


def span(f):
    """Span seminorm ||f||_sp = (sup f - inf f)/2."""
    f = np.asarray(f, float)
    return float((f.max() - f.min()) / 2.0)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def solve_optimum(rewards, Ns, w, beta, p_ref, seed=0, restarts=4, iters=4000):
    """INDEPENDENT solver for pi*: L-BFGS-B on a softmax parametrization (not mirror descent)."""
    from scipy.optimize import minimize
    K = len(p_ref)
    rng = np.random.default_rng(seed)
    best = None
    for s in range(restarts):
        z0 = np.zeros(K) if s == 0 else rng.normal(0, 1.0, K)

        def fun(z):
            p = softmax(z)
            val = loss(p, rewards, Ns, w, beta, p_ref)
            d = agg_grad(p, rewards, Ns, w)
            # dL/dp = -d + beta(log p - log p_ref + 1); chain through softmax
            dLdp = -d + beta * (np.log(p) - np.log(p_ref) + 1.0)
            gz = p * (dLdp - np.dot(p, dLdp))
            return val, gz

        res = minimize(fun, z0, jac=True, method="L-BFGS-B",
                       options={"maxiter": iters, "ftol": 1e-18, "gtol": 1e-14})
        p = softmax(res.x)
        v = loss(p, rewards, Ns, w, beta, p_ref)
        if best is None or v < best[0]:
            best = (v, p)
    return best[1], best[0]


def estimate_L(rewards, Ns, w, p_ref, seed=0, n_pairs=4000, conc=(0.2, 1.0, 5.0)):
    """Smallest L consistent with relative smoothness (Assumption 5.1) on sampled pairs:
        L >= ( R[pi] + <dR[pi], pi'-pi> - R[pi'] ) / KL[pi'|pi].
    Returns (L_hat, max_concavity_violation). A margin should be applied by the caller.
    """
    rng = np.random.default_rng(seed)
    K = len(p_ref)
    Lmax = 0.0
    conc_viol = 0.0
    for _ in range(n_pairs):
        a = rng.choice(conc)
        b = rng.choice(conc)
        p = rng.dirichlet(np.full(K, a))
        q = rng.dirichlet(np.full(K, b))
        p = np.clip(p, 1e-12, None); p /= p.sum()
        q = np.clip(q, 1e-12, None); q /= q.sum()
        Rp = agg_value(p, rewards, Ns, w)
        Rq = agg_value(q, rewards, Ns, w)
        d = agg_grad(p, rewards, Ns, w)
        lin = Rp + np.dot(d, q - p)
        conc_viol = max(conc_viol, Rq - lin)          # concavity: R[q] <= lin
        kq = kl(q, p)
        if kq > 1e-9:
            Lmax = max(Lmax, (lin - Rq) / kq)
    return float(Lmax), float(conc_viol)
