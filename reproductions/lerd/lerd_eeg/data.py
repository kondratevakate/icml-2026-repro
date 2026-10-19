"""Data pipeline for LERD Cohort A (ds004504): windowing, caching, folds.

Follows the paper's protocol (Appendix F.1):
  * 2 s non-overlapping windows from 500 Hz recordings -> T=1000 samples.
  * C=19 electrodes (10-20 layout).
  * channel-wise z-score per window (Algorithm 4, line 4).
  * subject-level 3-class labels: A=Alzheimer, F=FTD, C=HC.
  * five-fold CROSS-SUBJECT split (no subject appears in train and test).
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np
import mne

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "ds004504")
DERIV = os.path.join(DATA, "derivatives")
CACHE = os.path.join(HERE, "..", "data", "_cache_win")

SFREQ = 500
WIN = 1000              # 2 s @ 500 Hz
LABELS = {"A": 0, "F": 1, "C": 2}   # AD, FTD, HC
LABEL_NAMES = ["AD", "FTD", "HC"]


def subject_labels() -> dict[str, str]:
    path = os.path.join(DATA, "participants.tsv")
    with open(path) as fh:
        return {r["participant_id"]: r["Group"] for r in csv.DictReader(fh, delimiter="\t")}


def _window_subject(set_path: str) -> np.ndarray:
    """Return non-overlapping 2 s windows, shape (n_win, 19, 1000), float32."""
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    x = raw.get_data().astype(np.float32)         # (19, N)
    n_win = x.shape[1] // WIN
    x = x[:, : n_win * WIN].reshape(x.shape[0], n_win, WIN)  # (19, n_win, 1000)
    return np.transpose(x, (1, 0, 2))             # (n_win, 19, 1000)


def build_cache():
    """Window every subject once and cache to .npy (idempotent)."""
    os.makedirs(CACHE, exist_ok=True)
    labels = subject_labels()
    sets = sorted(glob.glob(os.path.join(DERIV, "sub-*", "eeg", "*_eeg.set")))
    for sp in sets:
        sid = os.path.basename(sp).split("_")[0]
        if labels.get(sid) not in LABELS:
            continue
        out = os.path.join(CACHE, f"{sid}.npy")
        if os.path.exists(out):
            continue
        w = _window_subject(sp)
        np.save(out, w)
        print(f"{sid}: {w.shape[0]} windows", flush=True)
    print("cache complete:", len(glob.glob(os.path.join(CACHE, "*.npy"))), "subjects")


def zscore_per_window(w: np.ndarray) -> np.ndarray:
    """Channel-wise z-score within each window (Algorithm 4, line 4)."""
    m = w.mean(axis=2, keepdims=True)
    s = w.std(axis=2, keepdims=True) + 1e-6
    return (w - m) / s


def load_subject(sid: str) -> np.ndarray:
    return zscore_per_window(np.load(os.path.join(CACHE, f"{sid}.npy")))


def five_folds(seed: int = 0) -> list[list[str]]:
    """Stratified-by-group cross-subject folds (list of subject-id lists)."""
    labels = subject_labels()
    sids = [s for s in labels if labels[s] in LABELS
            and os.path.exists(os.path.join(CACHE, f"{s}.npy"))]
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(5)]
    for g in LABELS:
        members = sorted([s for s in sids if labels[s] == g])
        rng.shuffle(members)
        for i, s in enumerate(members):
            folds[i % 5].append(s)   # round-robin keeps folds balanced per group
    return folds


if __name__ == "__main__":
    build_cache()
    fl = five_folds()
    lab = subject_labels()
    for i, f in enumerate(fl):
        from collections import Counter
        c = Counter(lab[s] for s in f)
        print(f"fold {i}: n={len(f)} {dict(c)}")
