"""Train ConfSleepNet with the paper's protocol (Appendix A.9) and evaluate both fusions.

Protocol, verbatim from Appendix A.9:
    sequences of 20 sleep epochs, batch size 16, Adam lr 1e-3, 100 training epochs,
    subject-wise k-fold CV with k=20 on SleepEDF-20 (= leave-one-subject-out);
    per fold: one subject group tests, one validates, the rest trains.
Metrics: overall accuracy and macro-F1 (Eq. 16-17), pooled over all folds.

Note on the ablation: Eq. 13 sums only the four per-view losses, so CMAM carries no
parameters and receives no gradient. ConfSleepNet and ConfSleepNet- (average fusion)
therefore share identical weights and differ only at inference. We train once per fold
and score both fusions on the same predictions.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from model import FINE_TO_COARSE, K, KC, ConfSleepNet

SEQ_LEN = 20


# --------------------------------------------------------------------------- data


def load_subjects(data_dir: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    """Group recordings by subject id (SC4<subject><night>)."""
    subjects: dict[str, list[pathlib.Path]] = {}
    for f in sorted(data_dir.glob("SC4*.npz")):
        subjects.setdefault(f.name[3:5], []).append(f)
    return subjects


def make_sequences(files: list[pathlib.Path]):
    """Per-recording z-score, then cut into non-overlapping 20-epoch sequences.

    Per-recording normalization, not global: cassette amplifier gain varies between
    subjects, so a global scale would leak subject identity across the LOSO split.
    """
    xs, ys = [], []
    for f in files:
        d = np.load(f)
        x, y = d["x"].astype(np.float32), d["y"].astype(np.int64)
        mu = x.mean(axis=(0, 2), keepdims=True)
        sd = x.std(axis=(0, 2), keepdims=True) + 1e-8
        x = (x - mu) / sd
        n = len(y) // SEQ_LEN
        if n == 0:
            continue
        xs.append(x[: n * SEQ_LEN].reshape(n, SEQ_LEN, 2, 3000))
        ys.append(y[: n * SEQ_LEN].reshape(n, SEQ_LEN))
    return np.concatenate(xs), np.concatenate(ys)


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x, self.y = torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# --------------------------------------------------------------------------- loss


def edl_loss(evidence, target, n_classes, lam):
    """Eq. 11 (digamma form) plus the annealed KL regularizer of Eq. 13.

    Eq. 11:  L_acc = sum_i y_i [psi(S) - psi(e_i + 1)]
    The KL term is the standard EDL penalty on misleading evidence (Sensoy 2018): the
    evidence of the true class is stripped out and the remainder is pulled towards the
    uniform Dirichlet. The paper describes it as "the weight of KL divergence ... avoid
    premature convergence to the uniform distribution" but does not print its formula.
    GAP(5).
    """
    e = evidence.reshape(-1, n_classes)
    y = F.one_hot(target.reshape(-1), n_classes).float()
    alpha = e + 1
    S = alpha.sum(-1, keepdim=True)
    acc = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(-1).mean()

    alpha_t = y + (1 - y) * alpha  # remove evidence of the true class
    St = alpha_t.sum(-1, keepdim=True)
    kl = (
        torch.lgamma(St).squeeze(-1)
        - torch.lgamma(torch.tensor(float(n_classes), device=e.device))
        - torch.lgamma(alpha_t).sum(-1)
        + ((alpha_t - 1) * (torch.digamma(alpha_t) - torch.digamma(St))).sum(-1)
    ).mean()
    return acc + lam * kl


# --------------------------------------------------------------------------- eval


def predict(model, loader, device):
    """Return (y_true, pred_cmam, pred_average) for a loader."""
    model.eval()
    yt, pc, pa = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            eeg, eog = x[:, :, :1], x[:, :, 1:]
            for fusion, bucket in (("cmam", pc), ("average", pa)):
                model.fusion = fusion
                _, b, _ = model(eeg, eog)
                bucket.append(b.argmax(-1).cpu().reshape(-1).numpy())
            yt.append(y.reshape(-1).numpy())
    return np.concatenate(yt), np.concatenate(pc), np.concatenate(pa)


def scores(y, p):
    return {
        "acc": round(100 * accuracy_score(y, p), 2),
        "mf1": round(100 * f1_score(y, p, average="macro"), 2),
        "f1_per_class": [round(v, 1) for v in 100 * f1_score(y, p, average=None, labels=range(K))],
    }


# --------------------------------------------------------------------------- fold


def run_fold(subjects, test_id, val_id, args, device):
    train_files = [f for s, fs in subjects.items() if s not in (test_id, val_id) for f in fs]
    xtr, ytr = make_sequences(train_files)
    xva, yva = make_sequences(subjects[val_id])
    xte, yte = make_sequences(subjects[test_id])

    dl = lambda x, y, sh: torch.utils.data.DataLoader(
        SeqDataset(x, y), batch_size=args.batch_size, shuffle=sh
    )
    tr_loader, va_loader, te_loader = dl(xtr, ytr, True), dl(xva, yva, False), dl(xte, yte, False)

    model = ConfSleepNet("cmam").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    f2c = FINE_TO_COARSE.to(device)

    best_va, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        lam = min(1.0, (ep + 1) / 10)  # Eq. 13 annealing coefficient
        t0 = time.time()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            yc = f2c[y]
            eeg, eog = x[:, :, :1], x[:, :, 1:]
            e, _, _ = model(eeg, eog)
            # f1, f2 supervise the 5-class structure; f3, f4 the 3-class one.
            loss = (
                edl_loss(e[0], y, K, lam)
                + edl_loss(e[1], y, K, lam)
                + edl_loss(e[2], yc, KC, lam)
                + edl_loss(e[3], yc, KC, lam)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        yv, pv, _ = predict(model, va_loader, device)
        va_acc = accuracy_score(yv, pv)
        if va_acc > best_va:
            best_va, best_state = va_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep + 1:3d}/{args.epochs} loss {loss.item():.3f} "
              f"val_acc {100 * va_acc:.2f} best {100 * best_va:.2f} ({time.time() - t0:.0f}s)",
              flush=True)

    model.load_state_dict(best_state)
    return predict(model, te_loader, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data/edf20"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/edf20"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=0, help="0 = all subjects (LOSO)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    subjects = load_subjects(args.data)
    ids = sorted(subjects)
    n_folds = args.folds or len(ids)
    print(f"{len(ids)} subjects, {n_folds} folds, device={device}", flush=True)

    yt_all, pc_all, pa_all, done = [], [], [], []
    for i in range(n_folds):
        test_id = ids[i]
        val_id = ids[(i + 1) % len(ids)]  # next subject validates, the rest trains
        print(f"fold {i + 1}/{n_folds}  test={test_id} val={val_id}", flush=True)
        t0 = time.time()
        yt, pc, pa = run_fold(subjects, test_id, val_id, args, device)
        yt_all.append(yt), pc_all.append(pc), pa_all.append(pa), done.append(test_id)
        print(f"  fold done in {(time.time() - t0) / 60:.1f} min  "
              f"cmam acc {100 * accuracy_score(yt, pc):.2f}  "
              f"avg acc {100 * accuracy_score(yt, pa):.2f}", flush=True)

        # Persist after every fold so an overnight interruption never loses completed work.
        y = np.concatenate(yt_all)
        result = {
            "folds_completed": done, "n_folds_planned": n_folds, "epochs": args.epochs,
            "seed": args.seed, "n_test_epochs": int(len(y)),
            "ConfSleepNet (CMAM)": scores(y, np.concatenate(pc_all)),
            "ConfSleepNet- (average)": scores(y, np.concatenate(pa_all)),
            "paper_SleepEDF20": {"ConfSleepNet": {"acc": 87.2, "mf1": 81.8},
                                 "ConfSleepNet-": {"acc": 86.3, "mf1": 81.3}},
        }
        (args.out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
