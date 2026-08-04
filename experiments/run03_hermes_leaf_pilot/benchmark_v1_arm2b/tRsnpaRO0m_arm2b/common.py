"""Reusable primitives for STRUCTURAL / functional-analysis theory-paper repro.

Copy into the repro workdir as `common.py`. Companion to
`references/structural_theory_repro.md`. Validated on tRsnpaRO0m (6/6 verified).

Deps: numpy, scipy.  CPU only.  Pin a seed (convention: 20260802).
"""
import numpy as np
from scipy.optimize import linprog

SEED = 20260802


# --------------------------------------------------------------------------
# Graph families
# --------------------------------------------------------------------------
def regular_graph(n, d, rng):
    """Random d-regular graph, O(n*d). `d` MUST be even.

    PITFALL THIS EXISTS TO AVOID: the naive rejection loop
        while A[i].sum() < d: j = rng.integers(n); ...
    can fail to terminate (blew a 600s timeout in the pilot). Build a
    circulant ring, then randomise with degree-preserving double-edge swaps.
    Need odd degree? Use d+1 and say so, or add a perfect matching.
    """
    assert d % 2 == 0, "circulant construction needs even d"
    A = np.zeros((n, n))
    for k in range(1, d // 2 + 1):
        for i in range(n):
            j = (i + k) % n
            A[i, j] = A[j, i] = 1
    edges = np.array(np.triu(A, 1).nonzero()).T
    for _ in range(2 * len(edges)):
        a, b = rng.integers(len(edges), size=2)
        (i, j), (k, l) = edges[a], edges[b]
        if len({i, j, k, l}) < 4 or A[i, l] or A[k, j]:
            continue
        A[i, j] = A[j, i] = A[k, l] = A[l, k] = 0
        A[i, l] = A[l, i] = A[k, j] = A[j, k] = 1
        edges[a] = (i, l); edges[b] = (k, j)
    return A


def bounded_degree_graph(n, r, rng, tries_factor=3):
    """Max-degree <= r graph (degrees not exact). Bounded attempts => always halts."""
    A = np.zeros((n, n))
    for _ in range(n * r * tries_factor):
        i, j = rng.integers(n, size=2)
        if i != j and A[i, j] == 0 and A[i].sum() < r and A[j].sum() < r:
            A[i, j] = A[j, i] = 1
    return A


def er_graph(n, p, rng):
    U = np.triu(rng.random((n, n)), 1)
    A = ((U > 0) & (U < p)).astype(float)
    return A + A.T


def powerlaw_graph(n, rng, alpha=1.2, m_factor=3):
    """Heavy-tailed degrees -> NOT a bounded-fiber family. Mutation baseline."""
    A = np.zeros((n, n))
    w = rng.pareto(alpha, size=n) + 1
    w /= w.sum()
    for _ in range(m_factor * n):
        i, j = rng.choice(n, size=2, p=w)
        if i != j:
            A[i, j] = A[j, i] = 1
    return A


# --------------------------------------------------------------------------
# Operator audit (self-adjointness / positivity on n uniform atoms)
# --------------------------------------------------------------------------
def audit_operator(W, n, rng, trials=200):
    """(Af)(x)=int W f dmu  ->  (W@f)/n on n uniform atoms."""
    A = lambda f: (W @ f) / n
    inner = lambda f, g: float(f @ g / n)
    sa = pos = 0.0
    for _ in range(trials):
        f, g = rng.normal(size=n), rng.normal(size=n)
        sa = max(sa, abs(inner(A(f), g) - inner(f, A(g))))
        h = rng.random(n)                      # h >= 0
        pos = max(pos, float(max(0.0, -np.min(A(h)))))
    return dict(selfadjoint_err=sa, positivity_violation=pos,
                op_norm_inf_to_1=float(np.abs(A(np.ones(n))).sum() / n))


# --------------------------------------------------------------------------
# DIDMs + exact mover's distance + eps-net (compactness surrogate)
# --------------------------------------------------------------------------
def didm(A, rmax):
    """Joint (deg, neighbour-deg) histogram on [0,rmax]^2, flattened & normalised."""
    deg = A.sum(1).astype(int)
    H = np.zeros((rmax + 1, rmax + 1))
    for i in range(A.shape[0]):
        nb = np.nonzero(A[i])[0]
        di = min(deg[i], rmax)
        if len(nb) == 0:
            H[di, 0] += 1.0
        else:
            for j in nb:
                H[di, min(deg[j], rmax)] += 1.0 / len(nb)
    return (H / H.sum()).ravel()


def movers_lp(bins):
    """Build an EXACT W1 solver on a bins x bins grid (l1 ground metric).

    Returns f(p, q) -> W1. Constraint matrix is built once; reuse it.
    """
    gx, gy = np.meshgrid(np.arange(bins), np.arange(bins), indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
    C = np.abs(pts[:, None, :] - pts[None, :, :]).sum(-1)
    K = C.shape[0]
    rows = []
    for i in range(K):
        r = np.zeros((K, K)); r[i, :] = 1; rows.append(r.ravel())
    for j in range(K):
        r = np.zeros((K, K)); r[:, j] = 1; rows.append(r.ravel())
    AEQ, CVEC = np.array(rows), C.ravel()

    def w1(p, q):
        res = linprog(CVEC, A_eq=AEQ, b_eq=np.concatenate([p, q]),
                      bounds=(0, None), method="highs")
        return float(res.fun)
    return w1


def w1_1d(p, q):
    """Exact 1-D W1 via CDF difference. For unbounded/ragged supports."""
    m = max(len(p), len(q))
    p = np.pad(p, (0, m - len(p))); q = np.pad(q, (0, m - len(q)))
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


def greedy_net(items, eps, dist):
    """Greedy eps-net. Compactness surrogate: sweep len(items) and check the
    net SATURATES for the bounded family and does NOT for the unbounded one."""
    net = []
    for x in items:
        if all(dist(x, c) > eps for c in net):
            net.append(x)
    return net


# --------------------------------------------------------------------------
# Random-feature MPNN (capacity sweeps for UAT / generalization)
# --------------------------------------------------------------------------
def mpnn_embed(A, width, seed, scale, depth=3):
    """Fixed random hidden weights + mean readout. Pair with a ridge head."""
    g = np.random.default_rng(seed)
    n = A.shape[0]
    h = np.concatenate([A.sum(1, keepdims=True) / scale, np.ones((n, 1))], 1)
    for _ in range(depth):
        din = h.shape[1]
        W1 = g.normal(size=(din, width)) / np.sqrt(din)
        W2 = g.normal(size=(din, width)) / np.sqrt(din * scale)
        h = np.tanh(h @ W1 + (A @ h) @ W2)
    return h.mean(0)


def ridge_fit(X, y, lam=1e-6):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)


def ridge_predict(X, w):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ w


# NOTE: when measuring a GENERALIZATION GAP, add label noise (sigma ~0.05).
# With a noiseless target the head fits exactly, the gap sits at the float64
# noise floor (~1e-7) and the fitted rate is meaningless, not merely bad.


# --------------------------------------------------------------------------
# Fits
# --------------------------------------------------------------------------
def loglog_slope(xs, ys):
    """Growth/decay exponent. ~0 => bounded; ~1 => linear; ~-0.5 => sqrt-rate."""
    return float(np.polyfit(np.log(np.asarray(xs, float)),
                            np.log(np.maximum(np.asarray(ys, float), 1e-16)), 1)[0])
