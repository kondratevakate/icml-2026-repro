"""Claim 4 (Section 4.1, Table 2): on the piecewise-constraint benchmark, CAffNet-TF reduces
test MSE by 73.33% relative to the soft-constrained NN baseline, with zero constraint
violations.

Paper protocol (Appendix D.1, notes_paper.md): target f on [-2,2], n_in = n_out = 1,
m = 4 constraints A(x) = [1,1,-1,-1]^T, b(x) = [g1u, g2u, -g1l, -g2l];
50 random training points, 400 linspace test points, MSE loss, Adam lr 1e-4,
50000 epochs, batch 500 (= full batch), 5 seeds, soft penalty 100*ReLU(A y - b).
Architectures: FF 3x200 ReLU; TF 3 heads of size 40, hidden 120.

Because n_out = 1, min(m, n_out) = 1, so Gamma = {(1,),(2,),(3,),(4,)} and
I - A_g^+ A_g = 0: the null-space term is inactive here (structurally), and Eq (12)
reduces to "keep f_theta if feasible, else take the nearest active bound".

Run: ./.venv/bin/python verify_claim4.py [--epochs N]
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "claim4.json")


# ---------------- target and constraints (Appendix D.1) ----------------
def target(x):
    y = np.empty_like(x)
    m1 = x <= -1; m2 = (x > -1) & (x <= 0); m3 = (x > 0) & (x <= 1); m4 = x > 1
    y[m1] = -5 * np.sin(np.pi / 2 * (x[m1] + 1)) - 2
    y[m2] = -2
    y[m3] = 2 - 9 * (x[m3] - 2 / 3) ** 2
    y[m4] = 3 / x[m4] ** 2 - 2
    return y


def bounds(x):
    s = np.sin(np.pi / 2 * (x + 1))
    m1 = x <= -1; m2 = (x > -1) & (x <= 0); m3 = (x > 0) & (x <= 1); m4 = x > 1
    g1u = np.empty_like(x); g2u = np.empty_like(x); g1l = np.empty_like(x); g2l = np.empty_like(x)
    g1u[m1] = -3 * s[m1] + 0.2;      g1u[m2] = -2; g1u[m3] = 3 - 4 * (x[m3] - 0.5) ** 2; g1u[m4] = 2
    g2u[m1] = -3 * s[m1] ** 3 + 1;   g2u[m2] = 2;  g2u[m3] = 3 - 4 * (x[m3] - 0.8) ** 2; g2u[m4] = 2.5
    g1l[m1] = 5 * s[m1] ** 2 - 3;    g1l[m2] = -2
    g1l[m3] = (4 - 9 * (x[m3] - 2 / 3) ** 2) * x[m3] - 2.5
    g1l[m4] = 3 / x[m4] ** 3 - 2.5
    g2l[m1] = 5 * s[m1] ** 8 - 2;    g2l[m2] = -3
    g2l[m3] = (5 - 4 * (x[m3] - 1 / 6) ** 2) * x[m3] - 2.5
    g2l[m4] = 3 / (2 * x[m4] ** 3) - 16 / 9
    return g1u, g2u, g1l, g2l


def make_b(x):
    """b(x) in R^{N x 4} for A = [1,1,-1,-1]^T."""
    g1u, g2u, g1l, g2l = bounds(x)
    return np.stack([g1u, g2u, -g1l, -g2l], axis=1)


A_ROWS = np.array([1.0, 1.0, -1.0, -1.0])


# ---------------- models ----------------
class FF(nn.Module):
    def __init__(self, out=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 200), nn.ReLU(), nn.Linear(200, 200), nn.ReLU(),
                                 nn.Linear(200, 200), nn.ReLU(), nn.Linear(200, out))

    def forward(self, x):
        return self.net(x)


class TF(nn.Module):
    """3 attention heads of size 40 (d_model = 120), feedforward hidden 120."""
    def __init__(self, out=1):
        super().__init__()
        self.emb = nn.Linear(1, 120)
        self.enc = nn.TransformerEncoderLayer(d_model=120, nhead=3, dim_feedforward=120,
                                              batch_first=True, dropout=0.0)
        self.head = nn.Linear(120, out)

    def forward(self, x):
        h = self.emb(x).unsqueeze(1)
        h = self.enc(h).squeeze(1)
        return self.head(h)


def caffine(f, b):
    """Eq (12) for n_out = 1, m = 4 (candidates = the four active bounds)."""
    cand = b / torch.tensor(A_ROWS, dtype=b.dtype)          # b_j / a_j
    viol = (torch.tensor(A_ROWS, dtype=b.dtype) * f - b)    # N x 4
    feasible_f = (viol <= 0).all(dim=1, keepdim=True)
    cviol = (torch.tensor(A_ROWS, dtype=b.dtype) * cand.unsqueeze(-1)).squeeze(-1)
    # feasibility of each candidate: A * c_j <= b for all rows
    Ac = cand.unsqueeze(2) * torch.tensor(A_ROWS, dtype=b.dtype).view(1, 1, 4)  # N x 4cand x 4row
    ok = (Ac <= b.unsqueeze(1) + 1e-9).all(dim=2)                                # N x 4cand
    dist = (cand - f).abs().masked_fill(~ok, float("inf"))
    idx = dist.argmin(dim=1, keepdim=True)
    proj = cand.gather(1, idx)
    return torch.where(feasible_f, f, proj)


def evaluate(y, xt, bt):
    yt = torch.tensor(target(xt), dtype=torch.float64).view(-1, 1)
    mse = float(((y - yt) ** 2).mean())
    v = (torch.tensor(A_ROWS, dtype=y.dtype) * y - bt).clamp(min=0)
    return mse, float(v.max()), float(v.mean())


def train_one(method, seed, epochs, lr=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    xtr = rng.uniform(-2, 2, size=50)
    xte = np.linspace(-2, 2, 400)
    Xtr = torch.tensor(xtr, dtype=torch.float64).view(-1, 1)
    Ytr = torch.tensor(target(xtr), dtype=torch.float64).view(-1, 1)
    Btr = torch.tensor(make_b(xtr), dtype=torch.float64)
    Xte = torch.tensor(xte, dtype=torch.float64).view(-1, 1)
    Bte = torch.tensor(make_b(xte), dtype=torch.float64)
    net = (TF() if method.endswith("TF") else FF()).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        f = net(Xtr)
        if method.startswith("CAffNet"):
            y = caffine(f, Btr)
            loss = ((y - Ytr) ** 2).mean()
        else:                                    # soft-constrained NN
            y = f
            pen = 100 * (torch.tensor(A_ROWS, dtype=y.dtype) * y - Btr).clamp(min=0)
            loss = ((y - Ytr) ** 2).mean() + pen.mean()
        loss.backward(); opt.step()
    ttrain = time.time() - t0
    with torch.no_grad():
        f = net(Xte)
        y = caffine(f, Bte) if method.startswith("CAffNet") else f
        mse, vmax, vmean = evaluate(y, xte, Bte)
    return {"method": method, "seed": seed, "test_mse": mse, "viol_max": vmax,
            "viol_mean": vmean, "train_s": ttrain}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50000)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    runs = []
    for method in ["NN", "CAffNet-FF", "CAffNet-TF"]:
        for seed in range(a.seeds):
            r = train_one(method, seed, a.epochs)
            runs.append(r)
            print(r, flush=True)
    agg = {}
    for method in ["NN", "CAffNet-FF", "CAffNet-TF"]:
        rs = [r for r in runs if r["method"] == method]
        agg[method] = {
            "mse_mean": float(np.mean([r["test_mse"] for r in rs])),
            "mse_std": float(np.std([r["test_mse"] for r in rs], ddof=1)),
            "viol_max_mean": float(np.mean([r["viol_max"] for r in rs])),
            "viol_mean_mean": float(np.mean([r["viol_mean"] for r in rs])),
            "n_seeds_with_any_violation": int(sum(r["viol_max"] > 1e-9 for r in rs)),
        }
    red = 100 * (agg["NN"]["mse_mean"] - agg["CAffNet-TF"]["mse_mean"]) / agg["NN"]["mse_mean"]
    red_ff = 100 * (agg["NN"]["mse_mean"] - agg["CAffNet-FF"]["mse_mean"]) / agg["NN"]["mse_mean"]
    res = {
        "claim": 4,
        "source": "Section 4.1 / Table 2 (paper: NN 0.0045, CAffNet-FF 0.0020, CAffNet-TF 0.0012; 73.33% reduction)",
        "command": f"./.venv/bin/python verify_claim4.py --epochs {a.epochs} --seeds {a.seeds}",
        "epochs": a.epochs, "seeds": a.seeds,
        "paper_table2": {"NN": 0.0045, "HardNet": 0.0037, "CAffNet-FF": 0.0020, "CAffNet-TF": 0.0012,
                         "reduction_TF_vs_NN_pct": 73.33},
        "runs": runs, "aggregate": agg,
        "reproduced_reduction_TF_vs_NN_pct": red,
        "reproduced_reduction_FF_vs_NN_pct": red_ff,
        "caffnet_zero_violations": bool(agg["CAffNet-FF"]["viol_max_mean"] == 0.0
                                        and agg["CAffNet-TF"]["viol_max_mean"] == 0.0),
        "nn_has_violations": bool(agg["NN"]["viol_max_mean"] > 0),
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
