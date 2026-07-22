"""Five-fold cross-subject training for LERD Cohort A (claim 4 / Table 4 No-prior).

Protocol (Appendix F.6): Adam lr 5e-4, weight decay 1e-4, batch 1024, 30 epochs,
grad-norm clip 1.0. Subject-level prediction by majority vote over window logits.
Metrics: accuracy and macro-F1, mean +/- std across the 5 folds.

Usage:
    python -m lerd_eeg.train --smoke     # 1 fold, 3 epochs, quick validation
    python -m lerd_eeg.train             # full 5-fold, 30 epochs
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from . import data as D
from .model import LERDNoPrior


def load_fold(train_ids, test_ids):
    def stack(ids):
        xs, ys = [], []
        lab = D.subject_labels()
        for s in ids:
            w = D.load_subject(s)                     # (n_win, 19, 1000)
            xs.append(w)
            ys.append(np.full(w.shape[0], D.LABELS[lab[s]], dtype=np.int64))
        return np.concatenate(xs), np.concatenate(ys)
    xtr, ytr = stack(train_ids)
    return xtr, ytr


def macro_f1(y_true, y_pred, n=3):
    f1s = []
    for c in range(n):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def train_one_fold(train_ids, test_ids, epochs, batch, device, log):
    torch.manual_seed(0)
    xtr, ytr = load_fold(train_ids, test_ids)
    xtr_t = torch.from_numpy(xtr).unsqueeze(1)        # (N,1,19,1000)
    ytr_t = torch.from_numpy(ytr)
    model = LERDNoPrior().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    n = xtr_t.shape[0]
    for ep in range(epochs):
        te0 = time.time()
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        log(f"    epoch {ep+1}/{epochs} loss {tot/n:.4f} ({time.time()-te0:.0f}s)")
    # subject-level evaluation by majority vote
    model.eval()
    lab = D.subject_labels()
    yt, yp = [], []
    with torch.no_grad():
        for s in test_ids:
            w = torch.from_numpy(D.load_subject(s)).unsqueeze(1).to(device)
            preds = []
            for i in range(0, w.shape[0], batch):
                preds.append(model(w[i:i + batch]).argmax(1).cpu().numpy())
            wp = np.concatenate(preds)
            yp.append(Counter(wp).most_common(1)[0][0])
            yt.append(D.LABELS[lab[s]])
    yt, yp = np.array(yt), np.array(yp)
    return float(np.mean(yt == yp)), macro_f1(yt, yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cpu"
    epochs = 3 if args.smoke else 30
    batch = 1024
    folds = D.five_folds()

    def log(m):
        print(m, flush=True)

    fold_range = range(1) if args.smoke else range(5)
    accs, f1s = [], []
    t0 = time.time()
    for k in fold_range:
        test_ids = folds[k]
        train_ids = [s for j, f in enumerate(folds) if j != k for s in f]
        log(f"fold {k}: train {len(train_ids)} test {len(test_ids)}")
        acc, f1 = train_one_fold(train_ids, test_ids, epochs, batch, device, log)
        log(f"fold {k}: acc {acc*100:.2f}% f1 {f1*100:.2f}%  ({time.time()-t0:.0f}s)")
        accs.append(acc); f1s.append(f1)
    a = np.array(accs) * 100; f = np.array(f1s) * 100
    log(f"\nNo-prior 5-fold: acc {a.mean():.2f} +/- {a.std():.2f} %  "
        f"f1 {f.mean():.2f} +/- {f.std():.2f} %")
    log("paper Table 4 No-prior (Cohort A): acc 70.52 +/- 11.83, f1 65.46 +/- 13.10")


if __name__ == "__main__":
    main()
