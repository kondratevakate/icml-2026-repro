"""Claim 3: Table 2 -- on real regression benchmarks PT-VCP shortens the interval in 9 of 10
datasets while keeping ~90% marginal coverage (alpha=0.1, p=0.95, 5 seeds, MLP of App. D.2.1).

Datasets attempted here: the freely downloadable UCI ones (bike, bio/CASP, concrete,
blog-data, facebook-1, facebook-2).  MEPS-19/20/21 need AHRQ MEPS PUF access + the CQR
repo's preprocessing, and STAR is Harvard-Dataverse gated -> NOT attempted (see logbook).

Run: .venv/bin/python verify_claim3.py
"""
import json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cp_common import vcp_intervals, pt_vcp, covered

torch.set_num_threads(max(1, os.cpu_count() // 2))
D = "data"
ALPHA, P, SEEDS = 0.10, 0.95, [0, 1, 2, 3, 4]
MAXN = 8000          # subsample cap (CPU budget); recorded in the json
BIAS = {"bike": 10.0, "bio": 10.0, "concrete": 5.0, "blog-data": 20.0,
        "facebook-1": 10.0, "facebook-2": 10.0}


def load(name):
    if name == "bike":
        df = pd.read_csv(f"{D}/bike/hour.csv")
        y = df["cnt"].values.astype(float)
        X = df.drop(columns=["instant", "dteday", "casual", "registered", "cnt"]).values.astype(float)
    elif name == "bio":
        df = pd.read_csv(f"{D}/bio/CASP.csv")
        y = df["RMSD"].values.astype(float); X = df.drop(columns=["RMSD"]).values.astype(float)
    elif name == "concrete":
        df = pd.read_excel(f"{D}/concrete/Concrete_Data.xls")
        y = df.iloc[:, -1].values.astype(float); X = df.iloc[:, :-1].values.astype(float)
    elif name == "blog-data":
        df = pd.read_csv(f"{D}/blog/blogData_train.csv", header=None)
        y = df.iloc[:, -1].values.astype(float); X = df.iloc[:, :-1].values.astype(float)
    elif name.startswith("facebook"):
        k = name[-1]
        df = pd.read_csv(f"{D}/facebook/Dataset/Training/Features_Variant_{k}.csv", header=None)
        y = df.iloc[:, -1].values.astype(float); X = df.iloc[:, :-1].values.astype(float)
    else:
        raise KeyError(name)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    return X[ok], y[ok]


class Net(nn.Module):
    """App. D.2.1: 3 fully connected layers, ReLU, 64 hidden units, dropout 0.1."""
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.1),
                               nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1),
                               nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def train(Xtr, ytr, seed, epochs=60, patience=8):
    torch.manual_seed(seed)
    n = Xtr.shape[0]; nv = max(1, int(0.1 * n))
    idx = np.random.default_rng(seed).permutation(n)
    tr, va = idx[nv:], idx[:nv]
    xt = torch.tensor(Xtr[tr], dtype=torch.float32); yt = torch.tensor(ytr[tr], dtype=torch.float32)
    xv = torch.tensor(Xtr[va], dtype=torch.float32); yv = torch.tensor(ytr[va], dtype=torch.float32)
    net = Net(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-6)
    lossf = nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(xt.shape[0])
        for i in range(0, xt.shape[0], 64):
            b = perm[i:i + 64]
            opt.zero_grad(); loss = lossf(net(xt[b]), yt[b]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(xv), yv))
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    return net


def predict(net, X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).numpy().astype(float)


def run_dataset(name):
    X, y = load(name)
    bias = BIAS[name]
    out = {"dataset": name, "bias": bias, "n_raw": int(X.shape[0]), "d": int(X.shape[1]),
           "per_seed": []}
    for s in SEEDS:
        rng = np.random.default_rng(s)
        idx = rng.permutation(X.shape[0])[:MAXN]
        Xs, ys = X[idx], y[idx]
        n = Xs.shape[0]
        ntr, nca = int(0.4 * n), int(0.2 * n)
        Xtr, ytr = Xs[:ntr], ys[:ntr]
        Xca, yca = Xs[ntr:ntr + nca], ys[ntr:ntr + nca]
        Xte, yte = Xs[ntr + nca:], ys[ntr + nca:]
        mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8
        my = np.mean(np.abs(ytr)) + 1e-8            # CQR-style response scaling
        f = lambda A: (A - mx) / sx
        Xtr, Xca, Xte = f(Xtr), f(Xca), f(Xte)
        ytr, yca, yte = ytr / my, yca / my, yte / my
        net = train(Xtr, ytr, s)
        # model misspecification: constant bias added to the network output (App. D.2.3)
        mca, mte = predict(net, Xca) + bias, predict(net, Xte) + bias
        lo, hi, q = vcp_intervals(mca, yca, mte, ALPHA)
        vc, vl = float(covered(lo, hi, yte).mean()), float(2 * q)
        plo, phi, plen, _ = pt_vcp(mca, yca, mte, ALPHA, P, rng)
        pc, pl = float(covered(plo, phi, yte).mean()), float(plen.mean())
        # interval stability (Definition 1), 200 repeated runs on the same test points
        runs = 200
        Ls = np.empty((runs, mte.size))
        for r in range(runs):
            _, _, l, _ = pt_vcp(mca, yca, mte, ALPHA, P, np.random.default_rng(5000 * s + r))
            Ls[r] = l
        is_pt = float(np.mean(np.var(Ls, axis=0)))
        out["per_seed"].append({"seed": s, "vcp_cov": vc, "vcp_len": vl,
                                "pt_cov": pc, "pt_len": pl, "IS_pt": is_pt, "IS_vcp": 0.0})
    agg = lambda k: (float(np.mean([r[k] for r in out["per_seed"]])),
                     float(np.std([r[k] for r in out["per_seed"]], ddof=1) / np.sqrt(len(SEEDS))))
    for k in ["vcp_cov", "vcp_len", "pt_cov", "pt_len", "IS_pt"]:
        out[k] = agg(k)
    out["pt_shorter"] = out["pt_len"][0] < out["vcp_len"][0]
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = {"command": ".venv/bin/python verify_claim3.py", "alpha": ALPHA, "p": P,
           "seeds": SEEDS, "subsample_cap": MAXN,
           "datasets_attempted": list(BIAS), "datasets_not_attempted": ["meps-19", "meps-20", "meps-21", "star"],
           "paper_table2": {"bike": [20.46, 19.59], "bio": [21.13, 20.44], "concrete": [10.32, 9.87],
                            "blog-data": [41.67, 41.13], "facebook-1": [20.81, 20.80],
                            "facebook-2": [20.97, 21.01]},
           "results": []}
    for name in BIAS:
        r = run_dataset(name)
        res["results"].append(r)
        print(f"{name}: VCP len={r['vcp_len'][0]:.2f}+-{r['vcp_len'][1]:.2f} cov={r['vcp_cov'][0]:.3f} | "
              f"PT len={r['pt_len'][0]:.2f}+-{r['pt_len'][1]:.2f} cov={r['pt_cov'][0]:.3f} | "
              f"IS_pt={r['IS_pt'][0]:.2f} | shorter={r['pt_shorter']}", flush=True)
    res["n_datasets_attempted"] = len(res["results"])
    res["n_pt_shorter"] = sum(r["pt_shorter"] for r in res["results"])
    res["all_coverage_within_0.02_of_nominal"] = all(
        abs(r["pt_cov"][0] - 0.90) <= 0.02 and abs(r["vcp_cov"][0] - 0.90) <= 0.02 for r in res["results"])
    res["min_IS_pt"] = min(r["IS_pt"][0] for r in res["results"])
    res["runtime_sec"] = time.time() - t0
    json.dump(res, open("results/claim3.json", "w"), indent=2)
    print(json.dumps({k: res[k] for k in ["n_datasets_attempted", "n_pt_shorter",
                                          "all_coverage_within_0.02_of_nominal", "min_IS_pt",
                                          "runtime_sec"]}, indent=2))
