"""verify_claim2.py — Claim 2 (Table 2, Section 4.1).

Claim: "Using LightGBM as the underlying classifier for L1-ERT achieves 68.4% relative
statistical power (relative to maximum), compared to only 38.3% for the PartitionWise
classifier underlying CovGap."

What is reproducible on CPU: the *protocol* of Table 2 (percentage of the maximum recovered
L1-ERT, averaged over test sizes and datasets) on real regression datasets, with the CPU
classifiers of Table 2 (LightGBM, RandomForest, ExtraTrees, PartitionWise).
What is NOT reproducible here: the exact 68.4 / 38.3 numbers, because the denominator (the
maximum recovered ERT over ALL methods) in the paper is set largely by the GPU tabular
foundation models TabICLv1.1 / RealTabPFN-2.5, and because the paper uses the eight largest
TabArena regression datasets. Both are recorded in the evidence boundary; no toy substitute
is used for the headline numbers.

Protocol per dataset (paper Sec 4.1): 40% train f (MSE), 10% calibration with
S(X,Y)=|Y-f(X)| at 1-alpha=0.9, 50% test subsampled to several sizes; L1-ERT by Algorithm 1
with 5-fold CV; repeated over seeds.

Run: .venv/bin/python verify_claim2.py
"""
import json
import time
import numpy as np
from ertlib import algorithm1

ALPHA = 0.1
T = 1 - ALPHA
SIZES = [1000, 2000, 5000]
SEEDS = [0, 1, 2]
METHODS = ["lightgbm", "randomforest", "extratrees", "partitionwise"]

DATASETS = ["diamonds", "superconduct", "physiochemical_protein", "house_sales"]


def load(name):
    from sklearn.datasets import fetch_openml
    mapping = {"diamonds": ("diamonds", 1),
               "superconduct": ("superconduct", 1),
               "physiochemical_protein": ("physicochemical-protein", 1),
               "house_sales": ("house_sales", 1)}
    did, ver = mapping[name]
    d = fetch_openml(did, version=ver, as_frame=True, data_home="./data", parser="auto")
    X = d.data.copy()
    y = np.asarray(d.target, dtype=float).ravel()
    for c in X.columns:
        if str(X[c].dtype) in ("category", "object"):
            X[c] = X[c].astype("category").cat.codes
    X = X.astype(float).fillna(0.0).to_numpy()
    return X, y


def fit_predictor(Xtr, ytr, seed):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=seed,
                             verbose=-1).fit(Xtr, ytr)


def ert_for_method(X, Z, method, seed):
    if method == "lightgbm":
        return algorithm1(X, Z, T, losses=("L1",), k=5, clf_kind="gb", seed=seed)["L1"]
    if method == "partitionwise":
        return algorithm1(X, Z, T, losses=("L1",), k=5, clf_kind="partitionwise",
                          seed=seed, n_groups=10)["L1"]
    # RF / ET via a small shim reusing Algorithm 1's cross-fitting
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.model_selection import KFold
    from ertlib import ert_terms
    Cls = RandomForestClassifier if method == "randomforest" else ExtraTreesClassifier
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    tot, m = 0.0, len(Z)
    for j, (tr, va) in enumerate(kf.split(X)):
        if len(np.unique(Z[tr])) < 2:
            pred = np.full(len(va), Z[tr].mean())
        else:
            clf = Cls(n_estimators=200, n_jobs=-1, random_state=seed + j).fit(X[tr], Z[tr])
            pred = clf.predict_proba(X[va])[:, 1]
        tot += (len(va) / m) * float(np.mean(ert_terms("L1", pred, Z[va], T)))
    return tot


records = []
used, failed = [], {}
for ds in DATASETS:
    try:
        X, y = load(ds)
    except Exception as e:  # dataset unavailable -> recorded, not faked
        failed[ds] = repr(e)[:200]
        continue
    used.append({"dataset": ds, "n": int(X.shape[0]), "d": int(X.shape[1])})
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n = len(y)
        i1, i2 = int(0.4 * n), int(0.5 * n)
        tr, cal, te = idx[:i1], idx[i1:i2], idx[i2:]
        f = fit_predictor(X[tr], y[tr], seed)
        s_cal = np.abs(y[cal] - f.predict(X[cal]))
        k = int(np.ceil((len(cal) + 1) * T))
        q = float(np.sort(s_cal)[min(k, len(cal)) - 1])
        s_te = np.abs(y[te] - f.predict(X[te]))
        Z_all = (s_te <= q).astype(float)
        X_te = X[te]
        for size in SIZES:
            if size > len(te):
                continue
            sub = rng.choice(len(te), size=size, replace=False)
            Xs, Zs = X_te[sub], Z_all[sub]
            for meth in METHODS:
                t0 = time.time()
                v = ert_for_method(Xs, Zs, meth, seed)
                records.append({"dataset": ds, "seed": seed, "size": size, "method": meth,
                                "L1_ERT": float(v), "secs": time.time() - t0,
                                "marginal_coverage": float(Zs.mean())})
            print(f"done {ds} seed{seed} n{size}", flush=True)

# percentage of the maximum recovered ERT (max over methods, per dataset x size x seed)
pcts = {m: [] for m in METHODS}
for ds in {r["dataset"] for r in records}:
    for seed in SEEDS:
        for size in SIZES:
            grp = [r for r in records if r["dataset"] == ds and r["seed"] == seed
                   and r["size"] == size]
            if not grp:
                continue
            mx = max(r["L1_ERT"] for r in grp)
            if mx <= 0:
                continue
            for r in grp:
                pcts[r["method"]].append(100.0 * r["L1_ERT"] / mx)

table = {m: {"avg_pct_of_max_L1_ERT": float(np.mean(v)) if v else None,
             "std": float(np.std(v)) if v else None, "n_cells": len(v)}
         for m, v in pcts.items()}
times = {m: float(np.mean([r["secs"] for r in records if r["method"] == m]))
         for m in METHODS if any(r["method"] == m for r in records)}

lg = table["lightgbm"]["avg_pct_of_max_L1_ERT"]
pw = table["partitionwise"]["avg_pct_of_max_L1_ERT"]
checks = {
    "paper_lightgbm_pct": 68.4, "paper_partitionwise_pct": 38.3,
    "cpu_reproduction_lightgbm_pct": lg, "cpu_reproduction_partitionwise_pct": pw,
    "ordering_lightgbm_much_greater_than_partitionwise": bool(lg is not None and pw is not None
                                                              and lg > pw + 15),
    "note": ("The denominator here is the max over CPU methods only (no GPU TabICL/TabPFN), so "
             "LightGBM is mechanically inflated relative to the paper's 68.4; the qualitative "
             "gap LightGBM >> PartitionWise is the reproducible part."),
}
out = {"alpha": ALPHA, "sizes": SIZES, "seeds": SEEDS, "datasets_used": used,
       "datasets_failed": failed, "table": table, "avg_seconds_per_estimate": times,
       "checks": checks, "records": records,
       "command": ".venv/bin/python verify_claim2.py"}
with open("results/claim2.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2))
