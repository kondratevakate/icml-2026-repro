"""Shared substrate for the l35QweVxgn reproduction (arm2).

Independent reimplementation from the paper's equations (arXiv 2510.05573):
  Eq.1/2  XOR-cluster data with Gaussian noise, K tasks, orthogonal means
  App. C  two-layer net, quadratic activation, only first layer trained, hinge loss
  Eq.17   general train-time forgetting characterisation
No code from the authors' repo was used.
"""
import json
import os
import numpy as np

SEED = 20260803
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def rng_for(claim, extra=0):
    return np.random.default_rng(SEED + 1000 * claim + extra)


# ---------------------------------------------------------------- data (Eq. 2)
def task_means(d, K, rng, mu_norm, orthogonal=True):
    """Return list of (mu_plus, mu_minus) per task, all 2K vectors mutually orthogonal."""
    need = 2 * K
    if orthogonal:
        assert need <= d, "need 2K <= d for mutually orthogonal means"
        Q, _ = np.linalg.qr(rng.standard_normal((d, need)))
        V = Q[:, :need].T
    else:  # mutation: shared/correlated directions across tasks
        V = rng.standard_normal((need, d))
        base = rng.standard_normal(d)
        V = 0.2 * V / np.linalg.norm(V, axis=1, keepdims=True) + 0.8 * base / np.linalg.norm(base)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
    V = mu_norm * V / np.linalg.norm(V, axis=1, keepdims=True)
    return [(V[2 * k], V[2 * k + 1]) for k in range(K)]


def sample_task(n, d, mup, mum, sigma, rng):
    y = rng.choice([-1.0, 1.0], size=n)
    s = rng.choice([-1.0, 1.0], size=n)
    centers = np.where(y[:, None] > 0, mup[None, :], mum[None, :]) * s[:, None]
    X = centers + sigma * rng.standard_normal((n, d))
    return X, y


def make_stream(d, K, n, sigma, rng, mu_norm=1.0, orthogonal=True, n_test=0):
    means = task_means(d, K, rng, mu_norm, orthogonal)
    tasks = []
    for k in range(K):
        X, y = sample_task(n, d, means[k][0], means[k][1], sigma, rng)
        te = sample_task(n_test, d, means[k][0], means[k][1], sigma, rng) if n_test else None
        tasks.append({"X": X, "y": y, "test": te, "mu": means[k]})
    return tasks


def sigma_of(d, c=1.0):
    """sigma = Theta(1/(log^c(d) sqrt(d)))  (Sec 2.1.2)."""
    return 1.0 / (np.log(d) ** c * np.sqrt(d))


# ------------------------------------------------- Eq.17 forgetting functional
def A_matrix(X, y):
    """A_j = (1/n) sum_v y_v x_v x_v^T."""
    n = X.shape[0]
    return (X * y[:, None]).T @ X / n


def forgetting_eq17(tasks, k, K, eta_T):
    """|(1/n) sum_{x_k} eta*T * x_k^T (sum_{j>k} A_j) x_k|  (paper Eq.17 leading term).

    k is 1-indexed task position; tasks[0..K-1].
    """
    S = np.zeros((tasks[0]["X"].shape[1],) * 2)
    for j in range(k, K):  # tasks k+1..K in 1-indexed == python k..K-1
        S += A_matrix(tasks[j]["X"], tasks[j]["y"])
    Xk = tasks[k - 1]["X"]
    quad = np.einsum("ij,jl,il->i", Xk, S, Xk)
    return abs(eta_T * quad.mean())


# ------------------------------------------------------- network (App. C) + GD
class QuadNet:
    """Phi(w,x) = (1/sqrt(m)) sum_i a_i <w_i,x>^2 ; first layer trained, hinge loss."""

    def __init__(self, d, m, rng):
        self.m = m
        self.a = rng.choice([-1.0, 1.0], size=m)
        self.W = rng.standard_normal((m, d))
        self.W0 = self.W.copy()

    def out(self, X, W=None):
        W = self.W if W is None else W
        Z = X @ W.T                      # n x m
        return (Z ** 2) @ self.a / np.sqrt(self.m)

    def hinge(self, X, y, W=None):
        u = y * self.out(X, W)
        return np.maximum(1.0 - u, 0.0).mean()

    def err(self, X, y, W=None):
        return float((np.sign(self.out(X, W)) != y).mean())

    def grad(self, X, y):
        n = X.shape[0]
        Z = X @ self.W.T
        u = y * ((Z ** 2) @ self.a / np.sqrt(self.m))
        act = (u < 1.0).astype(float)            # hinge subgradient mask
        coef = -(act * y)[:, None] * (2.0 * Z) * self.a[None, :] / np.sqrt(self.m)
        return coef.T @ X / n                     # m x d

    def gd(self, X, y, eta, T, record=None):
        for t in range(T):
            if record is not None:
                record.append(self.hinge(X, y))
            self.W -= eta * self.grad(X, y)
        return self


def save(n, payload):
    payload.setdefault("seed", SEED)
    payload.setdefault("orid", "l35QweVxgn")
    p = os.path.join(RESULTS, f"claim{n}.json")
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=float)
    print(f"[saved] {p}")
    print(json.dumps(payload, indent=2, sort_keys=True, default=float)[:2500])


def loglog_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])
