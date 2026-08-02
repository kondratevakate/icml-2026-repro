"""verify_claim4.py — Section 4.1 / Table 2: 73.33% MSE reduction of CAffNet-TF
vs the soft-constrained NN, with zero constraint violations.

Setup reimplemented from Appendix D.1 (see notes_paper.md):
  target f: [-2,2] -> R, n_in = n_out = 1
  4 piecewise bounds -> A(x) = [1,1,-1,-1]^T (m=4), b(x) = [g1u, g2u, -g1l, -g2l]
  => min(m, n_out) = 1, so Gamma = {(0,),(1,),(2,),(3,)} and the null-space term
     vanishes for every gamma (rank(A_gamma) = 1 = n_out).
  50 uniform-random training samples, 400 linspace test samples in [-2,2],
  MSE loss, Adam lr 1e-4, soft-penalty 100*ReLU(A y - b) for NN and HardNet,
  FF = 3 hidden layers x 200 ReLU; TF = 3 attention heads of size 40, hidden 120.
  5 seeds (paper: 5 seeds).

Methods compared: NN (soft), HardNet-Aff (+soft), CAffNet-FF, CAffNet-TF.
Metrics: test MSE vs the true target on the 400 test points, max/mean violation
r = ReLU(A(x)y - b(x)), and the MSE reduction of CAffNet-TF relative to NN.

MUTATION: 'CAffNet-TF-nolayer' — identical transformer trained with the CAffine
layer removed (plain soft penalty). Prediction: constraint violations reappear
(> 0) and the zero-violation part of the claim is attributable to the layer.

Epoch count is configurable (--epochs); the paper uses 50000. Whatever is used is
recorded in results/claim4.json.

Run: .venv/bin/python verify_claim4.py --epochs 50000
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)


# ---------------- target and constraints (Appendix D.1) ----------------
def _pw(x, fns):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    m1 = x <= -1
    m2 = (x > -1) & (x <= 0)
    m3 = (x > 0) & (x <= 1)
    m4 = x > 1
    for m, f in zip((m1, m2, m3, m4), fns):
        if m.any():
            out[m] = f(x[m])
    return out


S = lambda x: np.sin(np.pi / 2 * (x + 1))

f_target = lambda x: _pw(x, [lambda z: -5 * S(z) - 2,
                             lambda z: np.full_like(z, -2.0),
                             lambda z: 2 - 9 * (z - 2 / 3) ** 2,
                             lambda z: 3 / z ** 2 - 2])
g1u = lambda x: _pw(x, [lambda z: -3 * S(z) + 0.2,
                        lambda z: np.full_like(z, -2.0),
                        lambda z: 3 - 4 * (z - 0.5) ** 2,
                        lambda z: np.full_like(z, 2.0)])
g2u = lambda x: _pw(x, [lambda z: -3 * S(z) ** 3 + 1,
                        lambda z: np.full_like(z, 2.0),
                        lambda z: 3 - 4 * (z - 0.8) ** 2,
                        lambda z: np.full_like(z, 2.5)])
g1l = lambda x: _pw(x, [lambda z: 5 * S(z) ** 2 - 3,
                        lambda z: np.full_like(z, -2.0),
                        lambda z: (4 - 9 * (z - 2 / 3) ** 2) * z - 2.5,
                        lambda z: 3 / z ** 3 - 2.5])
g2l = lambda x: _pw(x, [lambda z: 5 * S(z) ** 8 - 2,
                        lambda z: np.full_like(z, -3.0),
                        lambda z: (5 - 4 * (z - 1 / 6) ** 2) * z - 2.5,
                        lambda z: 3 / (2 * z ** 3) - 16 / 9])


def make_b(x):
    """b(x) in R^{N x 4} with A = [1,1,-1,-1]^T."""
    return np.stack([g1u(x), g2u(x), -g1l(x), -g2l(x)], axis=1)


A_ROWS = np.array([1.0, 1.0, -1.0, -1.0])          # A(x) is constant here (m=4, n_out=1)


# ---------------- models ----------------
class FF(nn.Module):
    def __init__(self, n_in=1, n_out=1, h=200):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, n_out))

    def forward(self, x):
        return self.net(x)


class TF(nn.Module):
    """Transformer variant: 3 attention heads of size 40, hidden layer size 120."""

    def __init__(self, n_in=1, n_out=1, heads=3, dhead=40, hidden=120):
        super().__init__()
        d = heads * dhead                     # 120
        self.inp = nn.Linear(n_in, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.ln2 = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_out)

    def forward(self, x):
        z = self.inp(x).unsqueeze(1)          # (B,1,d) — sequence length 1
        a, _ = self.attn(z, z, z)
        z = self.ln1(z + a)
        z = self.ln2(z + self.ff(z))
        return self.out(z.squeeze(1))


def caffine_layer(f, b):
    """CAffine layer for A = [1,1,-1,-1]^T, n_out = 1 (Gamma = 4 singletons).

    For a scalar output and a_j in {+1,-1}: P_j = b_j / a_j  (the null-space term
    vanishes because rank(A_gamma) = 1 = n_out). Eq. (12): keep f if feasible, else
    the feasible candidate closest to f.  Differentiable in b and f.
    """
    a = torch.tensor(A_ROWS, dtype=f.dtype, device=f.device)      # (4,)
    Af = f * a                                                    # (B,4)
    feasible = (Af <= b).all(dim=1, keepdim=True)
    cand = b / a                                                  # (B,4) each P_gamma
    # feasibility of each candidate under ALL constraints
    ok = (cand.unsqueeze(2) * a.view(1, 1, 4) <= b.unsqueeze(1)).all(dim=2)  # (B,4)
    dist = (cand - f).abs()
    big = torch.full_like(dist, 1e9)
    dist_masked = torch.where(ok, dist, big)
    idx = dist_masked.argmin(dim=1, keepdim=True)
    proj = cand.gather(1, idx)
    return torch.where(feasible, f, proj)


def hardnet_layer(f, b):
    """HardNet-Aff for A(x)=[1,1]^T two-sided: y = clamp(f, max lower, min upper).

    With A = [1,1]^T (m=2 > n_out=1) A is NOT full row rank, so HardNet's assumption
    is violated; the standard pseudo-inverse correction is used, which is what the
    paper reports as producing violations.
    """
    a = torch.tensor(A_ROWS, dtype=f.dtype, device=f.device)
    v = torch.clamp(f * a - b, min=0.0)         # ReLU(A f - b), (B,4)
    # A^dagger = A^T / (A^T A) = a / 4
    return f - (v * a).sum(dim=1, keepdim=True) / 4.0


def violation(y, b):
    a = torch.tensor(A_ROWS, dtype=y.dtype)
    return torch.clamp(y * a - b, min=0.0)


def train_one(method, seed, epochs, lr=1e-4, batch=500, log=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    xtr = rng.uniform(-2, 2, size=50)
    ytr = f_target(xtr)
    btr = make_b(xtr)
    xte = np.linspace(-2, 2, 400)
    yte = f_target(xte)
    bte = make_b(xte)

    Xtr = torch.tensor(xtr, dtype=torch.float32).view(-1, 1)
    Ytr = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    Btr = torch.tensor(btr, dtype=torch.float32)
    Xte = torch.tensor(xte, dtype=torch.float32).view(-1, 1)
    Yte = torch.tensor(yte, dtype=torch.float32).view(-1, 1)
    Bte = torch.tensor(bte, dtype=torch.float32)

    net = TF() if method in ("CAffNet-TF", "NN-TF", "CAffNet-TF-nolayer") else FF()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    soft = method in ("NN", "HardNet", "NN-TF", "CAffNet-TF-nolayer")

    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        f = net(Xtr)                      # batch 500 >= 50 samples -> full batch
        if method == "CAffNet-FF" or method == "CAffNet-TF":
            y = caffine_layer(f, Btr)
        elif method == "HardNet":
            y = hardnet_layer(f, Btr)
        else:
            y = f
        loss = ((y - Ytr) ** 2).mean()
        if soft:
            loss = loss + 100 * violation(y, Btr).mean()
        loss.backward()
        opt.step()
    ttrain = (time.time() - t0) / epochs * 1000.0

    with torch.no_grad():
        t1 = time.time()
        f = net(Xte)
        if method in ("CAffNet-FF", "CAffNet-TF"):
            y = caffine_layer(f, Bte)
        elif method == "HardNet":
            y = hardnet_layer(f, Bte)
        else:
            y = f
        ttest = (time.time() - t1) * 1000.0
        mse = float(((y - Yte) ** 2).mean())
        r = violation(y, Bte)
        vmax = float(r.max())
        vmean = float(r.mean())
        vpct = float((r > 1e-9).any(dim=1).float().mean() * 100)
    return dict(method=method, seed=seed, mse=mse, viol_max=vmax, viol_mean=vmean,
                viol_pct_samples=vpct, T_train_ms=ttrain, T_test_ms=ttest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50000)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    methods = ["NN", "HardNet", "CAffNet-FF", "CAffNet-TF", "CAffNet-TF-nolayer"]
    recs = []
    for meth in methods:
        for s in range(a.seeds):
            r = train_one(meth, s, a.epochs)
            recs.append(r)
            print(f"{meth} seed{s}: mse={r['mse']:.6f} vmax={r['viol_max']:.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    agg = {}
    for meth in methods:
        rs = [r for r in recs if r["method"] == meth]
        agg[meth] = {k: float(np.mean([r[k] for r in rs]))
                     for k in ("mse", "viol_max", "viol_mean", "viol_pct_samples",
                               "T_train_ms", "T_test_ms")}
        agg[meth]["mse_std"] = float(np.std([r["mse"] for r in rs]))
    red = 100.0 * (1 - agg["CAffNet-TF"]["mse"] / agg["NN"]["mse"])
    out = {"claim": 4,
           "claim_text": "CAffNet-TF achieves 73.33% MSE reduction vs the soft-constrained "
                         "NN with zero constraint violations (Sec 4.1, Table 2)",
           "source": "Section 4.1, Table 2; setup Appendix D.1",
           "command": f".venv/bin/python verify_claim4.py --epochs {a.epochs} --seeds {a.seeds}",
           "epochs": a.epochs, "seeds": a.seeds,
           "paper_table2": {"NN": 0.0045, "HardNet": 0.0037, "CAffNet-FF": 0.0020,
                            "CAffNet-TF": 0.0012},
           "paper_reduction_pct": 73.33,
           "paper_reduction_pct_recomputed": 100.0 * (1 - 0.0012 / 0.0045),
           "aggregate": agg, "records": recs,
           "reproduced_reduction_pct_TF_vs_NN": red,
           "reproduced_reduction_pct_FF_vs_NN":
               100.0 * (1 - agg["CAffNet-FF"]["mse"] / agg["NN"]["mse"]),
           "mutation_TF_nolayer_viol_max": agg["CAffNet-TF-nolayer"]["viol_max"],
           "elapsed_s": time.time() - t0}
    json.dump(out, open("results/claim4.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2))
