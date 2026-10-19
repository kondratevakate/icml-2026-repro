"""Reproduction driver for SALaD (Explicitly Modeling Censoring, ICML 2026, 0xJt4PqPJv).

Runs a controlled A/B/C comparison for one survival layer family:
  baseline  (plain layer)         vs
  -2B       (two separate branches for event/censor) vs
  -salad    (shared + event- + censor-specific latent decomposition)

on a chosen dataset, over ``n_exp`` random splits, and reports
C-index / IBS / MAE-Hinge (mean +/- std) plus the salad-vs-baseline delta.

Only deviation from the upstream ``run.py`` pipeline: predicted survival
curves are clipped to [0, 1] before evaluation. This is a numerical guard
against float32 rounding (observed max = 1.0000004, ~4e-7 over 1.0), which
the strict SurvivalEVAL 0.8.0 validator rejects. It does not alter the
science -- only removes a version-mismatch artifact (the repo does not pin
SurvivalEVAL). All model code is imported verbatim from ``repo_ref``.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Import the upstream pipeline verbatim from the cloned repo.
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo_ref")
sys.path.insert(0, REPO)
os.chdir(REPO)  # data paths inside repo are relative ("data/support2.csv")

from run import (  # noqa: E402
    _prepare_data,
    _split_and_transform,
    _run_model,
    _evaluate_prediction,
    QUANTILE_MODELS,
)
from utils import df2np, set_seed  # noqa: E402

FAMILIES = {
    "N-MTLR": ("N-MTLR", "N-MTLR-2B", "N-MTLR-salad"),
    "DeepSurv": ("DeepSurv", "DeepSurv-2B", "DeepSurv-salad"),
    "AFTNN-Weibull": ("AFTNN-Weibull", "AFTNN-Weibull-2B", "AFTNN-Weibull-salad"),
    "AFTNN-LogLogistic": ("AFTNN-LogLogistic", "AFTNN-LogLogistic-2B", "AFTNN-LogLogistic-salad"),
}


def _base_args(dataset, model):
    return argparse.Namespace(
        data=dataset, model=model, n_exp=1, seed=0,
        neurons=[64, 64], norm=True, dropout=0.6, activation="ReLU",
        optimizer="AdamW", n_epochs=10000, early_stop=True, batch_size=256,
        lr=0.001, weight_decay=0.01, verbose=False,
        e_dims=[16], c_dims=[16], alpha=10.0, beta=0.01, ipm="mmd-rbf",
        n_quantiles=9, mono_method="bootstrap",
        timestamp=time.strftime("%Y%m%d_%H%M%S"),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        n_features=None,
    )


def _clip_prediction(model_name, prediction):
    if model_name in QUANTILE_MODELS:
        return prediction
    survival, time_coordinates = prediction
    survival = np.clip(np.asarray(survival), 0.0, 1.0)
    return survival, time_coordinates


def run_variant(dataset, model, n_exp, seed0):
    """Return list of per-split metric dicts for one model on one dataset."""
    args = _base_args(dataset, model)
    device = torch.device(args.device)
    data, features, cols_stdz = _prepare_data(args)
    args.n_features = len(features) - 2

    out = []
    for i in range(n_exp):
        seed = seed0 + i
        set_seed(seed, device)
        dtr, dva, dte, dtv = _split_and_transform(args, data, features, cols_stdz, seed)
        x_train, t_train, e_train = df2np(dtr)
        x_val, t_val, e_val = df2np(dva) if not dva.empty else (None, None, None)
        x_test, t_test, e_test = df2np(dte)
        x_tv, t_tv, e_tv = df2np(dtv)

        t0 = time.time()
        pred = _run_model(args, dtr, dva, x_train, t_train, e_train,
                          x_val, t_val, e_val, x_test, x_tv, t_tv, e_tv, seed, device)
        pred = _clip_prediction(model, pred)
        scores = _evaluate_prediction(model, pred, t_test, e_test, t_tv, e_tv)
        c_index, ibs, hinge, po, kmc, dcal_p, xcal = scores
        out.append(dict(seed=seed, cindex=float(c_index), ibs=float(ibs),
                        mae_hinge=float(hinge), mae_po=float(po),
                        km_cal=float(kmc), dcal_p=float(dcal_p),
                        secs=round(time.time() - t0, 1)))
        print(f"  [{model}] split {i} seed={seed}: "
              f"C={c_index:.4f} IBS={ibs:.4f} MAEh={hinge:.3f} ({out[-1]['secs']}s)",
              flush=True)
    return out


def summarize(rows):
    a = {k: np.array([r[k] for r in rows]) for k in ("cindex", "ibs", "mae_hinge", "mae_po")}
    return {k: (float(v.mean()), float(v.std())) for k, v in a.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="SUPPORT")
    p.add_argument("--family", default="N-MTLR", choices=list(FAMILIES))
    p.add_argument("--n-exp", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="path for JSON result (relative to salad_censoring)")
    a = p.parse_args()

    baseline, twobranch, salad = FAMILIES[a.family]
    results = {}
    for model in (baseline, twobranch, salad):
        print(f"\n=== {model} on {a.data} ({a.n_exp} splits) ===", flush=True)
        results[model] = run_variant(a.data, model, a.n_exp, a.seed)

    summary = {m: summarize(rows) for m, rows in results.items()}
    payload = dict(dataset=a.data, family=a.family, n_exp=a.n_exp, seed0=a.seed,
                   raw=results, summary=summary)

    print("\n================ SUMMARY ================")
    hdr = f"{'model':<22} {'C-index':>16} {'IBS':>16} {'MAE-Hinge':>16}"
    print(hdr)
    for m in (baseline, twobranch, salad):
        s = summary[m]
        print(f"{m:<22} "
              f"{s['cindex'][0]:.4f}+-{s['cindex'][1]:.4f}   "
              f"{s['ibs'][0]:.4f}+-{s['ibs'][1]:.4f}   "
              f"{s['mae_hinge'][0]:.3f}+-{s['mae_hinge'][1]:.3f}")
    dC = summary[salad]["cindex"][0] - summary[baseline]["cindex"][0]
    dI = summary[salad]["ibs"][0] - summary[baseline]["ibs"][0]
    print(f"\nSALaD vs baseline: dC-index={dC:+.4f} (higher better), "
          f"dIBS={dI:+.4f} (lower better)")

    if a.out:
        outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
