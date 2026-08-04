"""verify_claim5.py — Claim 5 (Table 4, Section 4.3.2 of arXiv v1).

Claim: classification experiments report divergent KL+-ERT and KL--ERT values across conformal
prediction methods, demonstrating the over/under-coverage decomposition in practice.

CPU-feasible subset attempted here: MNIST and FashionMNIST (the paper also reports CIFAR10 with
the same small CNN and CIFAR100 with a ResNet; the latter is out of the CPU budget).
Protocol follows the paper: a small CNN (2 conv + maxpool, 2 FC, dropout, ReLU) trained with
cross-entropy; two conformal score strategies
  * "likelihood":  S(x,y) = 1 - f_y(x)
  * "cumulative":  S(x,y) = sum of sorted softmax probabilities down to the true label (APS-style)
split-conformal calibrated at 1-alpha = 0.9; ERT metrics estimated with Algorithm 1 (5-fold CV,
LightGBM on the CNN's penultimate features, mirroring the paper's re-use of the trained trunk).

Reported: L1-ERT, KL-ERT, KL+-ERT, KL--ERT, over 3 seeds.

Run: .venv/bin/python verify_claim5.py
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from ertlib import ert_terms, GBClassifier

torch.set_num_threads(2)
ALPHA = 0.1
T = 1 - ALPHA
SEEDS = [0, 1, 2]
DATASETS = ["mnist", "fashion"]
N_TRAIN, N_CAL, N_TEST = 12000, 4000, 8000
EPOCHS = 3


def load(name):
    from sklearn.datasets import fetch_openml
    key = "mnist_784" if name == "mnist" else "Fashion-MNIST"
    d = fetch_openml(key, version=1, as_frame=False, data_home="./data", parser="auto")
    X = np.asarray(d.data, dtype=np.float32) / 255.0
    y = np.asarray(d.target, dtype=int)
    return X.reshape(-1, 1, 28, 28), y


class CNN(nn.Module):
    def __init__(self, k=10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Dropout(0.25))
        self.head = nn.Linear(64, k)

    def forward(self, x, feats=False):
        f = self.body(x)
        return (self.head(f), f) if feats else self.head(f)


def train(model, X, y, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    n = len(y)
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def probs_feats(model, X):
    model.eval()
    ps, fs = [], []
    Xt = torch.tensor(X)
    for i in range(0, len(X), 512):
        lo, f = model(Xt[i:i + 512], feats=True)
        ps.append(torch.softmax(lo, 1).numpy())
        fs.append(f.numpy())
    return np.concatenate(ps), np.concatenate(fs)


def scores(P, y, kind):
    if kind == "likelihood":
        return 1.0 - P[np.arange(len(y)), y]
    order = np.argsort(-P, axis=1)
    srt = np.take_along_axis(P, order, 1).cumsum(1)
    rank = np.argmax(order == y[:, None], axis=1)
    return srt[np.arange(len(y)), rank]


def ert_all(F, Z, seed):
    """Algorithm 1 with the over/under-coverage split (Sec 3.3)."""
    kf = KFold(5, shuffle=True, random_state=seed)
    m = len(Z)
    acc = {"L1": 0.0, "KL": 0.0, "KL_plus": 0.0, "KL_minus": 0.0}
    for j, (tr, va) in enumerate(kf.split(F)):
        clf = GBClassifier(seed=seed + j).fit(F[tr], Z[tr])
        h = np.clip(clf.predict_proba1(F[va]), 1e-6, 1 - 1e-6)
        w = len(va) / m
        acc["L1"] += w * float(np.mean(ert_terms("L1", h, Z[va], T)))
        acc["KL"] += w * float(np.mean(ert_terms("KL", h, Z[va], T)))
        acc["KL_plus"] += w * float(np.mean(ert_terms("KL", np.maximum(h, T), Z[va], T)))
        acc["KL_minus"] += w * float(np.mean(ert_terms("KL", np.minimum(h, T), Z[va], T)))
    return acc


records, failed = [], {}
for ds in DATASETS:
    try:
        X, y = load(ds)
    except Exception as e:
        failed[ds] = repr(e)[:200]
        continue
    for seed in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        tr = idx[:N_TRAIN]
        cal = idx[N_TRAIN:N_TRAIN + N_CAL]
        te = idx[N_TRAIN + N_CAL:N_TRAIN + N_CAL + N_TEST]
        model = train(CNN(), X[tr], y[tr], seed)
        Pc, _ = probs_feats(model, X[cal])
        Pt, Ft = probs_feats(model, X[te])
        acc_test = float((Pt.argmax(1) == y[te]).mean())
        for kind in ["cumulative", "likelihood"]:
            s_cal = scores(Pc, y[cal], kind)
            k = int(np.ceil((len(cal) + 1) * T))
            q = float(np.sort(s_cal)[k - 1])
            s_te = scores(Pt, y[te], kind)
            Z = (s_te <= q).astype(float)
            e = ert_all(Ft, Z, seed)
            rec = {"dataset": ds, "seed": seed, "method": kind, "q": q,
                   "test_accuracy": acc_test, "marginal_coverage": float(Z.mean()), **e,
                   "secs": time.time() - t0}
            records.append(rec)
            print(rec, flush=True)

summary = {}
for ds in {r["dataset"] for r in records}:
    for kind in ["cumulative", "likelihood"]:
        g = [r for r in records if r["dataset"] == ds and r["method"] == kind]
        if not g:
            continue
        summary[f"{ds}_{kind}"] = {
            k: {"mean": float(np.mean([r[k] for r in g])), "std": float(np.std([r[k] for r in g]))}
            for k in ["L1", "KL", "KL_plus", "KL_minus", "marginal_coverage", "test_accuracy"]}
        summary[f"{ds}_{kind}"]["additivity_err_KL_vs_parts"] = float(np.mean(
            [abs(r["KL"] - (r["KL_plus"] + r["KL_minus"])) for r in g]))

PAPER_TABLE4 = {
    "mnist_cumulative": {"L1": 0.150, "KL": -0.216, "KL_plus": -0.159, "KL_minus": -0.057},
    "mnist_likelihood": {"L1": 0.145, "KL": -0.187, "KL_plus": -0.128, "KL_minus": -0.059},
    "fashion_cumulative": {"L1": 0.165, "KL": -0.260, "KL_plus": -0.185, "KL_minus": -0.075},
    "fashion_likelihood": {"L1": 0.098, "KL": -0.068, "KL_plus": -0.042, "KL_minus": -0.026},
    "cifar10_cumulative": {"L1": 0.072, "KL": -0.017, "KL_plus": -0.030, "KL_minus": 0.012},
    "cifar10_likelihood": {"L1": 0.016, "KL": 0.028, "KL_plus": 0.007, "KL_minus": 0.022},
    "cifar100_cumulative": {"L1": 0.041, "KL": 0.191, "KL_plus": 0.016, "KL_minus": 0.175},
    "cifar100_likelihood": {"L1": 0.007, "KL": 0.409, "KL_plus": 0.085, "KL_minus": 0.323},
}
checks = {
    "decomposition_additivity_holds_in_practice": all(
        v["additivity_err_KL_vs_parts"] < 1e-9 for v in summary.values()),
    "KL_plus_and_KL_minus_differ_within_each_method": {
        k: abs(v["KL_plus"]["mean"] - v["KL_minus"]["mean"]) for k, v in summary.items()},
    "KL_plus_varies_across_methods": {
        ds: abs(summary.get(f"{ds}_cumulative", {}).get("KL_plus", {}).get("mean", float("nan"))
                - summary.get(f"{ds}_likelihood", {}).get("KL_plus", {}).get("mean", float("nan")))
        for ds in {r["dataset"] for r in records}},
    "paper_values_for_comparison": PAPER_TABLE4,
    "not_attempted": ["cifar10 (CNN, CPU-budget)", "cifar100 (ResNet, GPU-scale)"],
}
out = {"alpha": ALPHA, "seeds": SEEDS, "n_train": N_TRAIN, "n_cal": N_CAL, "n_test": N_TEST,
       "epochs": EPOCHS, "summary": summary, "checks": checks, "records": records,
       "datasets_failed": failed, "command": ".venv/bin/python verify_claim5.py"}
with open("results/claim5.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2))
