"""
nn_eoss.py -- a tiny numpy MLP + SGD/SGDM/SGDN with exact Batch Sharpness
(Definition 3.1 of the paper) measured via a finite-difference Hessian-vector
product.  Used to empirically reproduce the Edge-of-Stochastic-Stability
phenomena (batch-size-dependent Batch Sharpness plateaus and catapults) on CPU,
fully from first principles (no torch, no GPU).

Batch Sharpness at theta (paper Eq. 7):
    BS(theta) = E_{B~P_b}[ g_B(theta)^T H_B(theta) g_B(theta) / ||g_B(theta)||^2 ]
with g_B the mini-batch gradient and H_B the mini-batch Hessian.  We estimate
g_B^T H_B g_B by a finite-difference HVP:
    H_B g_B ~= ( grad_{theta}(L_B)(theta + eps*g_B) - grad(theta - eps*g_B) ) / (2 eps).
"""

import numpy as np


def _tanh(x):
    return np.tanh(x)


def _tanh_prime(x):
    return 1.0 - np.tanh(x) ** 2


class MLP:
    """2-hidden-layer MLP, scalar output, MSE loss.  Weights flattened for optimizers."""

    def __init__(self, din=2, dh=24, dout=1, seed=0, init_scale=0.3):
        rng = np.random.default_rng(seed)
        self.din, self.dh, self.dout = din, dh, dout
        self.W1 = rng.normal(scale=init_scale, size=(dh, din))
        self.b1 = rng.normal(scale=init_scale, size=(dh,))
        self.W2 = rng.normal(scale=init_scale, size=(dout, dh))
        self.b2 = rng.normal(scale=init_scale, size=(dout,))

    def flat(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def unflat(self, p):
        i = 0
        self.W1 = p[i:i + self.dh * self.din].reshape(self.dh, self.din); i += self.dh * self.din
        self.b1 = p[i:i + self.dh]; i += self.dh
        self.W2 = p[i:i + self.dout * self.dh].reshape(self.dout, self.dh); i += self.dout * self.dh
        self.b2 = p[i:i + self.dout]; i += self.dout

    def forward(self, X):
        # X: (n, din) -> (n, dout)
        z1 = X @ self.W1.T + self.b1           # (n, dh)
        a1 = _tanh(z1)
        z2 = a1 @ self.W2.T + self.b2         # (n, dout)
        return z2

    def loss(self, X, Y):
        out = self.forward(X)
        return np.mean((out - Y) ** 2)

    def grad(self, X, Y):
        """Flat gradient of mean MSE w.r.t. parameters (manual backprop)."""
        n = X.shape[0]
        z1 = X @ self.W1.T + self.b1
        a1 = _tanh(z1)
        z2 = a1 @ self.W2.T + self.b2
        dL_dz2 = 2.0 * (z2 - Y) / n            # (n, dout)
        dW2 = dL_dz2.T @ a1                    # (dout, dh)
        db2 = dL_dz2.sum(0)                    # (dout,)
        da1 = dL_dz2 @ self.W2                 # (n, dh)
        dz1 = da1 * _tanh_prime(z1)            # (n, dh)
        dW1 = dz1.T @ X                        # (dh, din)
        db1 = dz1.sum(0)                       # (dh,)
        return np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])


def batch_sharpness(net, X, Y, bs, n_mc=30, eps=1e-3, rng=None):
    """Definition 3.1 Monte-Carlo Batch Sharpness at current params (full gradient on batch)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = X.shape[0]
    vals = []
    for _ in range(n_mc):
        idx = rng.choice(n, size=bs, replace=False)
        Xb, Yb = X[idx], Y[idx]
        theta = net.flat()
        g = net.grad(Xb, Yb)
        gn = np.dot(g, g)
        if gn < 1e-12:
            continue
        gp = net.flat().copy()
        net.unflat(theta + eps * g)
        gp = net.grad(Xb, Yb)
        net.unflat(theta - eps * g)
        gm = net.grad(Xb, Yb)
        net.unflat(theta)
        Hg = (gp - gm) / (2.0 * eps)          # H_B g (HVP by finite difference)
        gHg = float(np.dot(g, Hg))
        vals.append(gHg / gn)
    return float(np.mean(vals)) if vals else float('nan')


# ---- optimizers operating on the flat parameter vector ----
def make_optimizer(kind, eta, beta=0.9):
    def sgd(net, X, Y, bs, rng, state):
        idx = rng.choice(X.shape[0], size=bs, replace=False)
        g = net.grad(X[idx], Y[idx])
        net.unflat(net.flat() - eta * g)
        return g

    def sgdm(net, X, Y, bs, rng, state):
        idx = rng.choice(X.shape[0], size=bs, replace=False)
        g = net.grad(X[idx], Y[idx])
        v = beta * state['v'] + g
        state['v'] = v
        net.unflat(net.flat() - eta * v)
        return g

    def sgdn(net, X, Y, bs, rng, state):
        look = net.flat() - beta * eta * state['v']
        net.unflat(look)
        idx = rng.choice(X.shape[0], size=bs, replace=False)
        g = net.grad(X[idx], Y[idx])
        v = beta * state['v'] + g
        state['v'] = v
        net.unflat(net.flat() - eta * v)
        return g

    if kind == 'sgd':
        return sgd
    if kind == 'sgdm':
        return sgdm
    if kind == 'sgdn':
        return sgdn
    raise ValueError(kind)


def synthetic_data(n=800, din=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, din))
    # smooth target + small label noise so the landscape supports EoSS
    Y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.05 * rng.normal(size=n)
    return X.astype(np.float64), Y.astype(np.float64).reshape(-1, 1)


def train(net, X, Y, kind, eta, beta, bs, steps, measure_every=200, n_mc=20, seed=0):
    rng = np.random.default_rng(seed)
    opt = make_optimizer(kind, eta, beta)
    state = {'v': np.zeros(net.flat().shape)}
    bs_hist, loss_hist, step_hist = [], [], []
    for t in range(steps):
        g = opt(net, X, Y, bs, rng, state)
        if t % measure_every == 0:
            bs_val = batch_sharpness(net, X, Y, bs, n_mc=n_mc, rng=rng)
            loss = net.loss(X, Y)
            bs_hist.append(bs_val)
            loss_hist.append(loss)
            step_hist.append(t)
    return {'step': step_hist, 'batch_sharpness': bs_hist, 'loss': loss_hist}
