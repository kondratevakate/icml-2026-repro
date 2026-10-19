"""C00657 (mechanism): RoVTL is designed to remain robust across the full range of tabular availability, 0% to 100%.

usage: python check_C00657.py A                     # structural / by-construction checks (random-weight repo model)
       python check_C00657.py B ROVTL|FULL [dry]    # fine-tune one arm from the C00429 checkpoints, AUC vs availability
       python check_C00657.py S                     # outcome from Part A + both arm files
Plan: checks/plan.json (entry C00657)."""
import json
import os
import random
import sys

import rovtl_common as rc

KV = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)  # optional overrides: data=..., epochs=..., lr=..., tag=...
sys.argv = [a for a in sys.argv if "=" not in a]
D = KV.get("data", "/work/work/data/easy34x8")
TAG = KV.get("tag", "")
RES = "/work/work/results"
os.makedirs(RES, exist_ok=True)
mode = sys.argv[1]
LEVELS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EVAL_LEVELS = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

if mode == "A":
    import numpy as np
    import torch
    import torch.nn as nn
    A = rc.init(reduced=False)
    from rovtl_common import (MutlimodalFusionClassifier, MRDownstreamDatasetH5, DownstreamCollator,
                              FirstHalfCollator, make_downstream_prepper, log)
    from torch.utils.data import DataLoader
    ok = {}
    # A1: forward at every availability level incl. 0 ---------------------------------------------------------
    torch.manual_seed(0)
    prep = make_downstream_prepper(D)
    clf = MutlimodalFusionClassifier(num_classes=4, dim=3).eval()
    log(f"model built; tabular attributes in table = {prep.max_pad_size_}")
    finite_all, shapes_ok = True, True
    batches = {}
    for f in LEVELS:
        random.seed(int(100 * f) + 7)
        ds = MRDownstreamDatasetH5(h5_path=D + "/test.h5", filter_csv=D + "/test.csv", simulate_missing=False,
                                   augmentation_rate=-1, num_present=f)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=FirstHalfCollator(DownstreamCollator(prep)))
        data = next(iter(loader))
        batches[f] = data
        n_unmasked_attr = int((~data["mask"][:, 1:]).sum(1).float().mean())
        with torch.no_grad():
            logits, _, _ = clf(data["x"], data["edge_attr"], data["mask"].clone(), data["image"].float())
        fin = bool(torch.isfinite(logits).all())
        finite_all &= fin
        shapes_ok &= tuple(logits.shape) == (8, 4)
        log(f"A1 availability {f:.1f}: mean #attributes present {n_unmasked_attr}/{prep.max_pad_size_}, logits {tuple(logits.shape)} finite={fin}")
    ok["A1_forward_all_levels"] = finite_all and shapes_ok

    # A2: f=0 -> output independent of tabular tensors; f=1 -> depends on them -------------------------------
    def run(d, x=None, e=None):
        with torch.no_grad():
            return clf(d["x"] if x is None else x, d["edge_attr"] if e is None else e, d["mask"].clone(), d["image"].float())[0]
    d = batches[0]
    base = run(d)
    x2, e2 = d["x"].clone(), d["edge_attr"].clone()
    m = d["mask"]
    x2[m] = torch.randn_like(x2[m]) * 10
    e2[m] = torch.randn_like(e2[m]) * 10
    diff0 = float((run(d, x2, e2) - base).abs().max())
    # also perturb the always-present read-out token row: fusion output must still not move at f=0
    x3 = d["x"].clone(); x3[:, 0, :] += 5.0
    diff0b = float((run(d, x3) - base).abs().max())
    log(f"A2 f=0: n fully-masked rows = {int((d['mask'][:, 1:].all(1)).sum())}/8; max |logit change| when masked tensors are replaced by noise = {diff0:.2e}; "
        f"when read-out row shifted = {diff0b:.2e}")
    d = batches[1.0]
    base1 = run(d)
    x4 = d["x"].clone(); keep = ~d["mask"]
    x4[keep] = x4[keep] + torch.randn_like(x4[keep])
    diff1 = float((run(d, x4) - base1).abs().max())
    log(f"A2 f=1.0: max |logit change| when present tabular values are perturbed = {diff1:.2e}")
    ok["A2_fallback_independent_at_0"] = diff0 < 1e-5
    ok["A2_depends_on_tabular_at_1"] = diff1 > 1e-4

    # A4 (additional diagnostic, not in the decision rule): gradients finite with fully-masked rows in a batch ---------
    clf.train()
    dm = {k: torch.cat([batches[0][k][:4], batches[1.0][k][:4]]) if torch.is_tensor(batches[0][k]) else None for k in ("x", "edge_attr", "mask", "image", "label")}
    logits, il, tl = clf(dm["x"], dm["edge_attr"], dm["mask"].clone(), dm["image"].float())
    loss = nn.BCEWithLogitsLoss()(logits, dm["label"]) + nn.BCEWithLogitsLoss()(il.squeeze(), dm["label"]) + nn.BCEWithLogitsLoss()(tl, dm["label"])
    loss.backward()
    bad = [n for n, p in clf.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    log(f"A4 backward with 4 fully-masked + 4 full rows: loss {float(loss):.4f}, parameters with non-finite grad: {len(bad)}")
    ok["A4_grad_finite_diagnostic"] = len(bad) == 0

    decisive = ok["A1_forward_all_levels"] and ok["A2_fallback_independent_at_0"] and ok["A2_depends_on_tabular_at_1"]
    log(f"PART A RESULT: {ok} -> {'CONSISTENT' if decisive else 'INCONSISTENT'} (A4 is a diagnostic only)")
    json.dump({"checks": {k: bool(v) for k, v in ok.items()}, "partA_consistent": bool(decisive),
               "diff_f0": diff0, "diff_f0_readout": diff0b, "diff_f1": diff1},
              open(f"{RES}/C00657_partA.json", "w"))

elif mode == "B":
    arm = sys.argv[2]
    dry = len(sys.argv) > 3 and sys.argv[3] == "dry"
    rc.init(reduced=True)
    from rovtl_common import finetune, eval_auc, make_downstream_prepper, log
    import numpy as np
    P = dict(batch_size=16, epochs=int(KV.get('epochs', 4)), lr=float(KV.get('lr', 3e-4)), seed=5)
    cfg = {"ROVTL": dict(mode="rovtl", lam=1.0), "FULL": dict(mode="full", lam=0.0)}[arm]
    import torch
    from rovtl_common import ImageTabularPretraining, seed_all
    seed_all(5)
    init_model = ImageTabularPretraining(dim=3)  # TARTE pretrained table encoder + random 3D ResNet-50, untrained, identical for both arms
    cfg["ckpt"] = "/work/work/ckpt/init_seed5.pth"
    torch.save(init_model.state_dict(), cfg["ckpt"])
    del init_model
    prep = make_downstream_prepper(D)
    log(f"ARM {arm}: {cfg}; downstream attributes = {prep.max_pad_size_}; {P}; dry={dry}")
    clf, hist = finetune(D, cfg["ckpt"], cfg["mode"], 1 if dry else P["epochs"], P["batch_size"], P["lr"], P["seed"], prep,
                         cfg["lam"], frozen=True, max_batches=1 if dry else None)
    res = {"arm": arm, "cfg": cfg, "loss_per_epoch": hist, "auc": {}}
    levels = [0, 0.5, 1.0] if dry else EVAL_LEVELS
    for li, f in enumerate(LEVELS):
        if f not in levels:
            continue
        draws = 1 if f in (0, 1.0) else (1 if dry else 2)
        a = [eval_auc(clf, D, prep, f, 1000 * li + k) for k in range(draws)]
        res["auc"][str(f)] = a
        log(f"ARM {arm} availability {f:.1f}: AUC draws {[round(x, 4) for x in a]} mean {np.mean(a):.4f}")
    json.dump(res, open(f"{RES}/C00657_partB_{arm}{TAG}{'_dry' if dry else ''}.json", "w"), indent=1)

elif mode == "S":
    import numpy as np
    A = json.load(open(f"{RES}/C00657_partA.json"))
    R = {a: json.load(open(f"{RES}/C00657_partB_{a}{TAG}.json")) for a in ("ROVTL", "FULL")}
    mean = {a: {float(f): float(np.mean(v)) for f, v in R[a]["auc"].items()} for a in R}
    for a in R:
        print(a, "AUC by availability:", {f: round(v, 3) for f, v in sorted(mean[a].items())})
    rv, fl = mean["ROVTL"], mean["FULL"]
    c1 = rv[0.0] >= 0.60
    c2 = min(rv.values()) >= rv[0.0] - 0.03
    c3 = min(rv.values()) > min(fl.values())
    usable = rv[1.0] >= 0.65
    img_learnable = max(rv[0.0], fl[0.0]) >= 0.58
    print(f"usable (ROVTL AUC at 100% >= 0.65): {usable}; image path learnable (max AUC(0%) over arms >= 0.58): {img_learnable}; (i) AUC(0%)>=0.60: {c1} ({rv[0.0]:.3f}); "
          f"(ii) min AUC >= AUC(0%)-0.03: {c2} (min {min(rv.values()):.3f}); (iii) worst-level ROVTL > worst-level FULL: {c3} "
          f"({min(rv.values()):.3f} vs {min(fl.values()):.3f})")
    if not A["partA_consistent"]:
        outcome = "inconsistent (part A failed)"
    elif not usable or not img_learnable:
        outcome = "blocked"
    elif c1 and c2 and c3:
        outcome = "consistent"
    else:
        outcome = "inconsistent"
    print("C00657 OUTCOME:", outcome)
    json.dump({"outcome": outcome, "usable": bool(usable), "c1": bool(c1), "c2": bool(c2), "c3": bool(c3),
               "mean_auc": {a: {str(f): v for f, v in mean[a].items()} for a in mean}}, open(f"{RES}/C00657_summary{TAG}.json", "w"))
