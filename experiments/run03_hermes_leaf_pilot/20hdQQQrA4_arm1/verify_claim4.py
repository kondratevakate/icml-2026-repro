"""verify_claim4.py — Sec. 4.1 piecewise-constraint benchmark.

Claim: CAffNet-TF reduces MSE by 73.33% vs the soft-constrained NN, with zero violations.
Paper Table 2: NN 0.0045, CAffNet-TF 0.0012 -> (0.0045-0.0012)/0.0045 = 73.33%.

Setup reproduced from notes_paper.md / App. D.1: n_in = n_out = 1, 50 random training
samples on [-2,2], 400 linearly spaced test points, MSE loss, Adam lr 1e-4, 50000 epochs,
batch 500 (= full batch here), 5 seeds. Soft penalty 100*ReLU(A y - b).
FF: 3 hidden x 200 ReLU. TF: 3 heads of size 40, feed-forward hidden 120.

MUTATION: CAffNet-TF with the CAffine layer removed (same transformer, soft penalty only)
-> the zero-violation property must disappear.
"""
import json, math, time, sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)   # fastest for these tiny batches (benchmarked)
def _argv_epochs():
    for a in sys.argv[1:]:
        if a.isdigit():
            return int(a)
    return 50000


EPOCHS = _argv_epochs()
SEEDS = [0, 1, 2, 3, 4]
OUT = "results/claim4.json"


# ---------------- target function and piecewise bounds (App. D.1) ----------------
def _pw(x, fns):
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    m1 = x <= -1; m2 = (x > -1) & (x <= 0); m3 = (x > 0) & (x <= 1); m4 = x > 1
    for m, f in zip((m1, m2, m3, m4), fns):
        if m.any():
            y[m] = f(x[m])
    return y


S = lambda x: np.sin(np.pi / 2 * (x + 1))
f_target = lambda x: _pw(x, [lambda t: -5 * S(t) - 2, lambda t: -2 * np.ones_like(t),
                             lambda t: 2 - 9 * (t - 2 / 3) ** 2, lambda t: 3 / t ** 2 - 2])
g1u = lambda x: _pw(x, [lambda t: -3 * S(t) + 0.2, lambda t: -2 * np.ones_like(t),
                        lambda t: 3 - 4 * (t - 0.5) ** 2, lambda t: 2 * np.ones_like(t)])
g2u = lambda x: _pw(x, [lambda t: -3 * S(t) ** 3 + 1, lambda t: 2 * np.ones_like(t),
                        lambda t: 3 - 4 * (t - 0.8) ** 2, lambda t: 2.5 * np.ones_like(t)])
g1l = lambda x: _pw(x, [lambda t: 5 * S(t) ** 2 - 3, lambda t: -2 * np.ones_like(t),
                        lambda t: (4 - 9 * (t - 2 / 3) ** 2) * t - 2.5,
                        lambda t: 3 / t ** 3 - 2.5])
g2l = lambda x: _pw(x, [lambda t: 5 * S(t) ** 8 - 2, lambda t: -3 * np.ones_like(t),
                        lambda t: (5 - 4 * (t - 1 / 6) ** 2) * t - 2.5,
                        lambda t: 3 / (2 * t ** 3) - 16 / 9])


def constraints(x):
    """A(x) = [1,1,-1,-1]^T, b(x) = [g1u, g2u, -g1l, -g2l]  (m = 4, n_out = 1)."""
    b = np.stack([g1u(x), g2u(x), -g1l(x), -g2l(x)], axis=1)
    A = np.tile(np.array([1.0, 1.0, -1.0, -1.0]), (len(x), 1))
    return A, b


# ---------------------------------- models ----------------------------------
class FF(nn.Module):
    def __init__(self, nout=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 200), nn.ReLU(), nn.Linear(200, 200), nn.ReLU(),
                                 nn.Linear(200, 200), nn.ReLU(), nn.Linear(200, nout))

    def forward(self, x):
        return self.net(x)


class TF(nn.Module):
    """3 attention heads of size 40 (d_model = 120), feed-forward hidden 120."""

    def __init__(self, nout=1):
        super().__init__()
        self.inp = nn.Linear(1, 120)
        self.enc = nn.TransformerEncoderLayer(d_model=120, nhead=3, dim_feedforward=120,
                                              dropout=0.0, batch_first=True)
        self.out = nn.Linear(120, nout)

    def forward(self, x):
        h = self.inp(x).unsqueeze(1)
        h = self.enc(h).squeeze(1)
        return self.out(h)


def caffine_layer(f, w, A, b):
    """Eqs. 8/9/12 in torch, vectorised. n_out = 1 -> Gamma = 4 singletons, k = 1,
    I - A_g^+ A_g = 0 so the null-space term is inert here (paper Sec. 3.2)."""
    N, m = A.shape
    viol = (A * f - b)                              # (N, m)
    feasible = (viol <= 0).all(dim=1, keepdim=True)
    cands = b / A                                   # P_gamma for each single row (n_out=1)
    ok = (A.unsqueeze(1) * cands.unsqueeze(2) - b.unsqueeze(1) <= 1e-9).all(dim=2)  # (N,m)
    dist = (cands - f).abs()
    dist = torch.where(ok, dist, torch.full_like(dist, 1e9))
    j = dist.argmin(dim=1, keepdim=True)
    proj = cands.gather(1, j)
    return torch.where(feasible, f, proj)


def train(kind, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtr = rng.uniform(-2, 2, size=50)
    ytr = f_target(xtr)
    xte = np.linspace(-2, 2, 400)
    yte = f_target(xte)
    Atr, btr = constraints(xtr)
    Ate, bte = constraints(xte)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    Xtr, Ytr = T(xtr).unsqueeze(1), T(ytr).unsqueeze(1)
    Xte = T(xte).unsqueeze(1)
    At, bt = T(Atr), T(btr)
    Ae, be = T(Ate), T(bte)

    if kind == "nn_soft":
        net, wnet, use_caff = FF(), None, False
    elif kind == "caffnet_tf":
        net, wnet, use_caff = TF(), TF(), True
    elif kind == "tf_soft_mutation":
        net, wnet, use_caff = TF(), None, False
    else:
        raise ValueError(kind)
    params = list(net.parameters()) + (list(wnet.parameters()) if wnet else [])
    opt = torch.optim.Adam(params, lr=1e-4)

    for ep in range(epochs):
        opt.zero_grad()
        f = net(Xtr)
        if use_caff:
            y = caffine_layer(f, wnet(Xtr), At, bt)
            loss = ((y - Ytr) ** 2).mean()
        else:
            y = f
            loss = ((y - Ytr) ** 2).mean() + 100 * torch.relu(At * y - bt).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        f = net(Xte)
        y = caffine_layer(f, wnet(Xte), Ae, be) if use_caff else f
        mse = float(((y.squeeze(1).numpy() - yte) ** 2).mean())
        r = np.maximum(Ate * y.numpy() - bte, 0.0)
        return dict(seed=seed, mse=mse, viol_max=float(r.max()), viol_mean=float(r.mean()),
                    viol_pct=float((r > 0).mean() * 100))


if __name__ == "__main__":
    t0 = time.time()
    res = {"claim": 4, "epochs": EPOCHS, "seeds": SEEDS,
           "paper_table2": {"NN_mse": 0.0045, "CAffNet_TF_mse": 0.0012,
                            "claimed_reduction_pct": 73.33}}
    for kind in ("nn_soft", "caffnet_tf", "tf_soft_mutation"):
        runs = [train(kind, s) for s in SEEDS]
        res[kind] = dict(runs=runs,
                         mse_mean=float(np.mean([r["mse"] for r in runs])),
                         mse_std=float(np.std([r["mse"] for r in runs])),
                         viol_max=float(max(r["viol_max"] for r in runs)),
                         viol_mean=float(np.mean([r["viol_mean"] for r in runs])),
                         viol_pct_mean=float(np.mean([r["viol_pct"] for r in runs])))
        print(kind, res[kind]["mse_mean"], res[kind]["viol_max"], flush=True)
    a, b = res["nn_soft"]["mse_mean"], res["caffnet_tf"]["mse_mean"]
    res["measured_reduction_pct"] = float((a - b) / a * 100)
    res["command"] = f"python verify_claim4.py {EPOCHS}"
    res["torch"] = torch.__version__
    res["seconds"] = round(time.time() - t0, 2)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "paper_table2"}, indent=2))
