"""Preprocess Sleep-EDF Expanded (sleep-cassette) into per-recording .npz arrays.

Target protocol, taken verbatim from the ConfSleepNet paper (Appendix A.7, Table 5):
    SleepEDF-20 : 20 subjects, 39 recordings, EEG Fpz-Cz, EOG ROC-LOC, R&K scoring, 100 Hz
    SleepEDF-78 : 78 subjects, 153 recordings, same channels/rate

R&K scoring means stages 3 and 4 are merged into a single N3, giving the 5 classes
{W, N1, N2, N3, REM} the paper classifies into. Movement/unscored epochs are dropped.

Output per recording: x float32 (n_epochs, 2, 3000)  -- [EEG, EOG] at 100 Hz x 30 s
                      y int8    (n_epochs,)          -- 0=W 1=N1 2=N2 3=N3 4=REM
"""

from __future__ import annotations

import argparse
import pathlib

import mne
import numpy as np

mne.set_log_level("ERROR")

EEG_CH = "EEG Fpz-Cz"
EOG_CH = "EOG horizontal"
SFREQ = 100
EPOCH_SEC = 30
EPOCH_LEN = SFREQ * EPOCH_SEC  # 3000

# R&K -> 5-class AASM-style labels. Stage 4 folds into N3.
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
DROP = -1  # unscored ("Sleep stage ?") and movement time


def read_recording(psg_path: pathlib.Path, hyp_path: pathlib.Path):
    """Return (x, y) for a full recording, with no epoch selection applied yet.

    x: float32 (n_epochs, 2, 3000), y: int8 (n_epochs,) with DROP for unusable epochs.
    """
    raw = mne.io.read_raw_edf(psg_path, preload=True, include=[EEG_CH, EOG_CH])
    assert raw.info["sfreq"] == SFREQ, f"unexpected sfreq {raw.info['sfreq']}"
    sig = raw.get_data(picks=[EEG_CH, EOG_CH]).astype(np.float32)  # (2, n_samples)

    n_epochs = sig.shape[1] // EPOCH_LEN
    x = sig[:, : n_epochs * EPOCH_LEN].reshape(2, n_epochs, EPOCH_LEN).transpose(1, 0, 2)

    y = np.full(n_epochs, DROP, dtype=np.int8)
    for onset, duration, desc in zip(*_annotations(hyp_path)):
        label = STAGE_MAP.get(desc, DROP)
        start = int(round(onset / EPOCH_SEC))
        stop = min(start + int(round(duration / EPOCH_SEC)), n_epochs)
        if start < n_epochs:
            y[start:stop] = label
    return np.ascontiguousarray(x), y


def _annotations(hyp_path: pathlib.Path):
    ann = mne.read_annotations(hyp_path)
    return ann.onset, ann.duration, ann.description


def select_epochs(y: np.ndarray, wake_margin_epochs: int = 60) -> np.ndarray:
    """Return a boolean mask of the epochs that enter the dataset.

    WHY THIS IS YOUR CALL, NOT MINE
    -------------------------------
    Sleep-EDF cassette recordings run ~22 h but contain ~8 h of sleep: SC4001 opens with
    30630 s (8.5 h) of a single "Sleep stage W" block before lights-out. Keeping all of it
    makes W the majority class by a wide margin and inflates overall accuracy for free.
    The ConfSleepNet paper never states what it did here (Appendix A.7 is silent), yet the
    choice moves accuracy by several points -- more than the entire 0.9-point margin the
    paper claims for CMAM over average fusion. So this single function decides whether our
    87.2% verdict means anything.

    The options, with their trade-offs:
      (a) 30-min margin (wake_margin_epochs=60): keep only the wake epochs within 30 min of
          the first and last non-wake epoch. This is what DeepSleepNet/TinySleepNet did, and
          every baseline in the paper's Table 1 is quoted from papers using it -- so it is
          the only choice under which our number is comparable to their number. Downside: it
          is an assumption about the paper, not a fact from it.
      (b) In-bed only (wake_margin_epochs=0): drop all leading/trailing wake. Cleanest
          physiologically, hardest task, accuracy lands lowest.
      (c) No trimming: keep everything. Faithful to "we used the 39 recordings" read
          literally, but W dominates and the comparison to Table 1 becomes meaningless.

    Whatever you pick, epochs with y == DROP must always be excluded.

    Chosen policy: (a), wake_margin_epochs=60. Recorded in the logbook as an explicit
    assumption about the paper, since the paper itself does not state it.
    """
    scored = y != DROP
    sleep = np.flatnonzero(scored & (y != STAGE_MAP["Sleep stage W"]))
    if sleep.size == 0:
        return np.zeros_like(scored)
    lo = max(0, sleep[0] - wake_margin_epochs)
    hi = min(len(y), sleep[-1] + wake_margin_epochs + 1)
    span = np.zeros_like(scored)
    span[lo:hi] = True
    return span & scored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=pathlib.Path, required=True, help="sleep-cassette dir")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--max-subject", type=int, default=20, help="20 -> SleepEDF-20, 83 -> -78")
    ap.add_argument("--wake-margin-epochs", type=int, default=60)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    psgs = sorted(p for p in args.src.glob("SC4*-PSG.edf") if int(p.name[3:5]) < args.max_subject)
    print(f"{len(psgs)} recordings")

    for psg in psgs:
        hyps = list(args.src.glob(psg.name[:6] + "*-Hypnogram.edf"))
        assert len(hyps) == 1, f"{psg.name}: {len(hyps)} hypnograms"
        x, y = read_recording(psg, hyps[0])
        mask = select_epochs(y, args.wake_margin_epochs)
        x, y = x[mask], y[mask]
        counts = np.bincount(y, minlength=5)
        np.savez_compressed(args.out / f"{psg.name[:6]}.npz", x=x, y=y)
        print(f"{psg.name[:6]}  {len(y):5d} epochs  W/N1/N2/N3/REM {counts}")


if __name__ == "__main__":
    main()
