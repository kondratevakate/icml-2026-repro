"""Shared first-principles harness for reproducing anchored claims of
arXiv:2509.24757 "Accelerating Regression Tasks with Quantum Algorithms"
(Liu & Ji, ICML-2026 submission TBSyYj4VV6).  CPU-only, numpy/scipy.

Two independently testable halves of every claim:

  (A) CORRECTNESS half  -- the algorithm is "sparsify then solve":
      importance (Lewis/leverage) sampling of s = O~(n/eps^2) rows must give a
      (1 +- eps) approximation of the GLM objective F(x)=sum_i f_i(<a_i,x>),
      and the solution of the sparsified problem must be near-optimal for the
      full problem.  This is fully executable on CPU.

  (B) COST half -- the claimed O~(r*sqrt(mn)/eps + poly(n)) quantum runtime.
      No quantum hardware / no full state-vector simulation is feasible here,
      so we *count queries* of an explicitly implemented quantum-rejection-
      sampling routine using the standard exact amplitude-amplification
      iteration count ceil((pi/4)/sqrt(p_acc)), while actually drawing the
      samples classically and checking the output distribution is the intended
      one (TV distance).  This measures the *scaling exponent in m*, which is
      what "quadratic speedup in the sample count m" means.  Verdicts for this
      half are therefore at best `toy` (simulated cost model), never
      `verified` on hardware.
"""
from __future__ import annotations
import json, math, os, time
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


# ---------------------------------------------------------------- data
def gen_sparse_design(m, n, r, seed=0, cond=10.0, spikes=0):
    """m x n design matrix with exactly r nonzeros per row (sparsity r<=n).

    `spikes` rows are "outlier" rows supported on a single coordinate with a
    large norm; they carry leverage score ~1, which puts the instance in the
    worst-case regime the paper's O~(r sqrt(mn)/eps) bound is stated for
    (max_i tau_i = Theta(1) while sum_i tau_i = rank <= n).
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((m, n))
    for i in range(m):
        cols = rng.choice(n, size=r, replace=False)
        A[i, cols] = rng.standard_normal(r)
    for k in range(min(spikes, m)):
        A[k] = 0.0
        A[k, k % n] = 50.0 * (1.0 + k)
    # give the matrix a nontrivial (but bounded) condition number
    scale = np.linspace(1.0, cond, n)
    A = A * scale[None, :]
    return A


# --------------------------------------------------- importance scores
def leverage_scores(A, reg=1e-12):
    """tau_i = a_i^T (A^T A)^+ a_i.  sum_i tau_i = rank(A) <= n."""
    G = A.T @ A
    Gi = np.linalg.pinv(G + reg * np.eye(A.shape[1]))
    return np.einsum("ij,jk,ik->i", A, Gi, A)


def lewis_weights(A, p, iters=30, reg=1e-12):
    """Cohen-Peng fixed point for l_p Lewis weights.  p=2 -> leverage scores."""
    m = A.shape[0]
    w = np.ones(m)
    for _ in range(iters):
        Aw = A * (w ** (0.5 - 1.0 / p))[:, None]
        tau = leverage_scores(Aw, reg)
        w_new = np.clip(tau, 1e-15, None) ** (p / 2.0)
        w = 0.5 * w + 0.5 * w_new  # damped for stability
    return w


# ------------------------------------------------------- sparsification
def importance_sample(scores, s, m, rng):
    """Sample s rows i.i.d. prop. to scores; return (idx, multiplier)."""
    p = scores / scores.sum()
    idx = rng.choice(m, size=s, replace=True, p=p)
    mult = 1.0 / (s * p[idx])
    return idx, mult


def sparsify(A, s, seed=0, p=2, uniform=False):
    """Return (idx, weights) of an eps-sparsifier candidate of A."""
    m = A.shape[0]
    rng = np.random.default_rng(seed)
    scores = np.ones(m) if uniform else np.clip(lewis_weights(A, p), 1e-15, None)
    idx, mult = importance_sample(scores, s, m, rng)
    return idx, mult


# ------------------------------------------------------------- losses
def loss_apply(name, z, delta=1.0, p=2.0):
    if name == "l2":
        return z ** 2
    if name == "lp":
        return np.abs(z) ** p
    if name == "huber":
        a = np.abs(z)
        return np.where(a <= delta, 0.5 * z ** 2, delta * (a - 0.5 * delta))
    if name == "gamma_p":  # gamma_p loss: quadratic small / |z|^p large
        a = np.abs(z)
        return np.where(a <= 1.0, 0.5 * z ** 2, (a ** p) / p + 0.5 - 1.0 / p)
    raise ValueError(name)


def F_full(A, b, x, name, **kw):
    return float(np.sum(loss_apply(name, A @ x - b, **kw)))


def F_sparse(A, b, x, idx, w, name, **kw):
    return float(np.sum(w * loss_apply(name, A[idx] @ x - b[idx], **kw)))


def relative_errors(A, b, name, idx, w, n_probe=200, seed=1, **kw):
    """max over probe directions x of |F_S(x)/F(x) - 1|.

    Probes = random directions + the n canonical directions (the latter expose
    a sparsifier that misses high-leverage / outlier rows).
    """
    rng = np.random.default_rng(seed)
    n = A.shape[1]
    probes = [rng.standard_normal(n) / np.sqrt(n) * rng.uniform(0.2, 5.0)
              for _ in range(n_probe)]
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        probes.append(e)
    errs = []
    for x in probes:
        f = F_full(A, b, x, name, **kw)
        if f <= 0:
            continue
        fs = F_sparse(A, b, x, idx, w, name, **kw)
        errs.append(abs(fs / f - 1.0))
    return float(np.max(errs)), float(np.mean(errs))


# ------------------------------------------------ simulated quantum cost
class QueryCounter:
    def __init__(self):
        self.q = 0

    def add(self, k):
        self.q += int(math.ceil(k))


def quantum_rejection_sample(scores, s, rng, counter):
    """Quantum rejection sampling (Grover / amplitude amplification) of s rows
    from the distribution prop. to `scores`.

    Cost model (standard, exact): preparing one accepted sample costs
    ceil((pi/4)/sqrt(p_acc)) queries to the row oracle, where
    p_acc = mean(scores)/max(scores).  We ACTUALLY draw the samples with a
    classical rejection loop so the output distribution can be validated.
    """
    m = len(scores)
    smax = scores.max()
    p_acc = scores.mean() / smax
    per_sample = (math.pi / 4.0) / math.sqrt(p_acc)
    out = np.empty(s, dtype=int)
    for j in range(s):
        while True:  # classical realisation of the same distribution
            i = rng.integers(m)
            if rng.random() < scores[i] / smax:
                out[j] = i
                break
        counter.add(per_sample)
    return out


def classical_sample_cost(m, s, r, counter):
    """Classical importance sampling must at minimum touch every row once to
    build the score vector: Theta(m*r) work (Jambulapati et al. STOC'24)."""
    counter.add(m * r)
    return None


def fit_exponent(xs, ys):
    """slope of log y vs log x."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(sol[0])


def save(name, payload):
    payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", path)
    return path
