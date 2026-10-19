"""C00429 (mechanism): random masking of tabular attributes as augmentation during contrastive pretraining makes the
learned representation robust to missing tabular values at test time.

usage: python check_C00429.py A                 # structural checks, full repository attribute lists
       python check_C00429.py B AUG|NOAUG [dry] # pre-train one arm on synthetic tiny16 data and evaluate retrieval
       python check_C00429.py S                 # summary / outcome from the two arm result files
Plan: checks/plan.json (entry C00429)."""
import json
import math
import os
import random
import sys

import rovtl_common as rc

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)  # optional overrides: data=..., epochs=..., lr=..., tag=...
sys.argv = [a for a in sys.argv if "=" not in a]
D = KV.get("data", "/work/work/data/tiny34x8")
TAG = KV.get("tag", "")
RES = "/work/work/results"
os.makedirs(RES, exist_ok=True)
os.makedirs("/work/work/ckpt", exist_ok=True)
mode = sys.argv[1]


def nan_eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


if mode == "A":
    rc.init(reduced=False)
    import numpy as np
    import torch
    from rovtl_common import (MRContrastiveDatasetH5, DataCollator, NUMERICAL_ATTRIBUTES, CATEGORICAL_ATTRIBUTES,
                              LABEL_MAPPING, DISEASE_MAPPING, make_pretrain_prepper, log)
    random.seed(0)
    ds = MRContrastiveDatasetH5(D + "/train.h5", augmentation_rate=-1, simulate_missing=True)
    ds_full = MRContrastiveDatasetH5(D + "/train.h5", augmentation_rate=-1, simulate_missing=False)
    cols = list(ds.metadata_columns)
    n_cat, n_num = len(CATEGORICAL_ATTRIBUTES), len(NUMERICAL_ATTRIBUTES)
    log(f"full repository attribute lists: {n_cat} categorical + {n_num} numerical = {n_cat + n_num} attributes")

    # A1 + A2 ------------------------------------------------------------------------------------------------
    n_draws = 2000
    mismatches, kept_entries = 0, 0
    totals, k_cat_l, k_num_l = [], [], []
    for d in range(n_draws):
        i = random.randrange(len(ds))
        item = ds[i]
        row = ds.metadata[i]
        for col in CATEGORICAL_ATTRIBUTES:
            new = DISEASE_MAPPING.get(col, col)
            if new in item["categorical"]:
                v = row[cols.index(col)]
                exp = LABEL_MAPPING[col][v] if not np.isnan(v) else float("nan")
                kept_entries += 1
                mismatches += 0 if nan_eq(item["categorical"][new], exp) else 1
        for col in NUMERICAL_ATTRIBUTES:
            if col in item["continuous"]:
                v = row[cols.index(col)]
                kept_entries += 1
                mismatches += 0 if nan_eq(item["continuous"][col], float(v)) else 1
        k_cat, k_num = len(item["categorical"]), len(item["continuous"])
        k_cat_l.append(k_cat); k_num_l.append(k_num); totals.append(k_cat + k_num)
        # extra keys that are not attributes? (e.g. injected noise/corrupted columns)
        extra = set(item["categorical"]) - {DISEASE_MAPPING.get(c, c) for c in CATEGORICAL_ATTRIBUTES}
        extra |= set(item["continuous"]) - set(NUMERICAL_ATTRIBUTES)
        mismatches += len(extra)
    totals = np.array(totals)
    log(f"A1 value integrity: {kept_entries} kept entries over {n_draws} draws, mismatches vs ground truth = {mismatches}")
    log(f"A2 kept attributes per draw: min {totals.min()} max {totals.max()} mean {totals.mean():.1f} std {totals.std():.1f} "
        f"distinct sizes {len(set(totals.tolist()))}; cat min/max {min(k_cat_l)}/{max(k_cat_l)} (of {n_cat}), "
        f"num min/max {min(k_num_l)}/{max(k_num_l)} (of {n_num})")
    full_item = ds_full[0]
    log(f"A2 simulate_missing=False keeps all: {len(full_item['categorical'])} cat, {len(full_item['continuous'])} num "
        f"(expected {n_cat}, {n_num})")
    paper_mean = (1 + n_cat + n_num) / 2
    log(f"A2 paper text: subset size in [1, N_full]=[1,{n_cat + n_num}] (uniform mean {paper_mean:.1f}); "
        f"code: cat and num groups sampled separately, k_cat~U[1,{n_cat}], k_num~U[1,{n_num}] -> total in [2,{n_cat + n_num}], "
        f"mean {(1 + n_cat) / 2 + (1 + n_num) / 2:.1f}")
    a1 = mismatches == 0 and kept_entries > 0
    a2 = (totals.min() >= 2 and totals.max() >= 0.9 * (n_cat + n_num) and totals.min() <= 0.1 * (n_cat + n_num)
          and len(set(totals.tolist())) > 20 and len(full_item["categorical"]) == n_cat and len(full_item["continuous"]) == n_num)

    # A3 -----------------------------------------------------------------------------------------------------
    prep = make_pretrain_prepper(D)
    batch = [ds[random.randrange(len(ds))] for _ in range(64)]
    data = DataCollator(prep)(batch)
    mask = data["mask"]
    unmasked = (~mask).sum(1).numpy()
    expected = np.array([1 + sum(0 if (isinstance(v, float) and math.isnan(v)) else 1
                                 for v in list(b["categorical"].values()) + list(b["continuous"].values())) for b in batch])
    a3 = bool((unmasked == expected).all())
    log(f"A3 TARTE token mask: tokens tensor {tuple(data['x'].shape)}, unmasked tokens == 1 + kept non-NaN attributes for "
        f"{int((unmasked == expected).sum())}/64 rows (min unmasked {unmasked.min()}, max {unmasked.max()})")
    ok = a1 and a2 and a3
    log(f"PART A RESULT: A1(value integrity)={a1} A2(size variability / full when off)={a2} A3(masked tokens)={a3} -> "
        f"{'CONSISTENT' if ok else 'INCONSISTENT'}")
    json.dump({"A1": bool(a1), "A2": bool(a2), "A3": bool(a3), "partA_consistent": bool(ok),
               "kept_per_draw_min": int(totals.min()), "kept_per_draw_max": int(totals.max()),
               "kept_per_draw_mean": float(totals.mean()), "mismatches": int(mismatches), "n_draws": n_draws},
              open(f"{RES}/C00429_partA.json", "w"))

elif mode == "B":
    arm = sys.argv[2]
    dry = len(sys.argv) > 3 and sys.argv[3] == "dry"
    rc.init(reduced=True)
    from rovtl_common import pretrain, make_pretrain_prepper, encode_test, retrieval_metrics, log
    import numpy as np
    import torch
    P = dict(batch_size=16, epochs=int(KV.get('epochs', 8)), lr=float(KV.get('lr', 3e-4)), seed=5)
    fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    prep = make_pretrain_prepper(D)
    log(f"ARM {arm}: attributes in table = {prep.max_pad_size_}; params {P}; dry={dry}")
    model, hist = pretrain(D, arm == "AUG", 1 if dry else P["epochs"], P["batch_size"], P["lr"], P["seed"],
                           f"/work/work/ckpt/{arm}{TAG}{'_dry' if dry else ''}.pth", prep=prep, max_batches=1 if dry else None, freeze_image_trunk=True)
    img, tab = encode_test(model, D, prep, [1.0] if dry else fracs, seed=1234)
    n = img[0].shape[0]
    out = {"arm": arm, "loss_per_epoch": hist, "n_test": n, "chance_top5": 5 / n, "metrics": {}}
    for f in tab:
        m = retrieval_metrics(img, tab[f], tab[1.0])
        out["metrics"][str(f)] = m
        log(f"ARM {arm} f={f}: " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
    json.dump(out, open(f"{RES}/C00429_partB_{arm}{TAG}{'_dry' if dry else ''}.json", "w"), indent=1)

elif mode == "S":
    import numpy as np
    A = json.load(open(f"{RES}/C00429_partA.json"))
    R = {a: json.load(open(f"{RES}/C00429_partB_{a}{TAG}.json")) for a in ("AUG", "NOAUG")}
    chance5 = R["AUG"]["chance_top5"]
    low = ["0.1", "0.2", "0.3"]

    def mean_low(a, key):
        return float(np.mean([R[a]["metrics"][f][key] for f in low]))

    for key in ("i2t_top5", "t2i_top5", "i2t_top1", "cos_to_full_table"):
        print(f"low-availability (f=0.1,0.2,0.3) mean {key}: AUG {mean_low('AUG', key):.3f}  NOAUG {mean_low('NOAUG', key):.3f}")
    full = {a: R[a]["metrics"]["1.0"]["i2t_top5"] for a in R}
    print(f"full-table i2t top-5: AUG {full['AUG']:.3f} NOAUG {full['NOAUG']:.3f}; informative threshold 3x chance = {3 * chance5:.3f}")
    drop = {a: (R[a]["metrics"]["1.0"]["i2t_top5"] - R[a]["metrics"]["0.1"]["i2t_top5"]) / max(R[a]["metrics"]["1.0"]["i2t_top5"], 1e-9) for a in R}
    print(f"relative drop of top-5 from f=1.0 to f=0.1: AUG {drop['AUG']:.3f} NOAUG {drop['NOAUG']:.3f}")
    informative = all(full[a] >= 3 * chance5 for a in R)
    better_top5 = mean_low("AUG", "i2t_top5") > mean_low("NOAUG", "i2t_top5")
    better_cos = mean_low("AUG", "cos_to_full_table") > mean_low("NOAUG", "cos_to_full_table")
    less_drop = drop["AUG"] < drop["NOAUG"]
    print(f"informative regime: {informative}; AUG>NOAUG on low-availability top-5: {better_top5}; on cosine-to-full: {better_cos}; "
          f"AUG loses less (rel. drop): {less_drop}")
    if not informative:
        outcome = "blocked"
    elif better_top5 and better_cos and less_drop and A["partA_consistent"]:
        outcome = "consistent"
    elif not (better_top5 or better_cos):
        outcome = "inconsistent"
    else:
        outcome = "inconsistent (mixed)"
    if not A["partA_consistent"]:
        outcome = "inconsistent (part A failed)"
    print("C00429 OUTCOME:", outcome)
    json.dump({"outcome": outcome, "informative": bool(informative), "better_top5": bool(better_top5),
               "better_cos": bool(better_cos), "less_drop": bool(less_drop)}, open(f"{RES}/C00429_summary{TAG}.json", "w"))
