"""Schema-faithful SYNTHETIC data for the OSF (arXiv 2603.00190) pilot. No real data are used.

Schema (taken from the repository's loaders/configs and the paper):
  * 12 channels in the order of train_config.TRAIN_EDF_COLS_UNI_ENC:
      ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN,
      EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1
  * 64 Hz x 30 s = 1920 samples per epoch, float32, clamped to [-6, 6] (the loader clamps).
  * 4-class sleep stage label (0 Wake, 1 Light, 2 Deep, 3 REM) and binary labels
    Arousal / Hypopnea / Oxygen Desaturation (osf.datasets.pretrain_dataset.SleepEpochDataset).
  * repo file layout: <csv_dir>/patient_pretrain_{train,valid,test}.csv,
    <csv_dir>/epoch_pretrain_{train,valid,test}.csv (cols nsrrid, seg_id, dataset_name,
    epoch_id, path_head, labels) and <path_head>/epoch-{seg_id:05d}_all.parquet with one
    float column per channel and a float time index in seconds.

Generative model (invented here; it is NOT a model of real PSG):
  Stage s ~ Cat(.15,.45,.15,.25). Three channel groups as in paper Sec 3.1: brain (EOG+EEG, 4 ch),
  respiratory (ABD, THX, NP, SN, 4 ch), other (ECG + 3 EMG, 4 ch). Each group has ONE shared
  narrowband latent oscillation per epoch whose centre frequency depends on (group, stage)
  (+-12% jitter) -> the class signal is redundantly present in every group (cross-channel
  redundancy) with group gains (brain 0.45, resp 0.28, other 0.20; calibrated so that hand-crafted band-power features give macro-AUC well below 1). Every channel additionally has
  channel-specific broadband noise and a channel-specific class-independent narrowband
  "texture" at a random frequency (nuisance that only identifies the epoch within ONE channel).
  Channels are z-scored per epoch.
Arousal/Hypopnea/Oxygen Desaturation labels are independent Bernoulli noise (schema only).
"""
import argparse
import os

import numpy as np

CHANNELS = ["ECG", "EMG_Chin", "EMG_LLeg", "EMG_RLeg", "ABD", "THX", "NP", "SN",
            "EOG_E1_A2", "EOG_E2_A1", "EEG_C3_A2", "EEG_C4_A1"]
FS = 64
T = 30 * FS  # 1920
GROUPS = {"other": [0, 1, 2, 3], "resp": [4, 5, 6, 7], "brain": [8, 9, 10, 11]}
GAIN = {"brain": 0.45, "resp": 0.28, "other": 0.2}
# class-dependent centre frequency (Hz) per group; arbitrary fixed values
CLASS_FREQ = {"brain": [10.0, 6.0, 1.5, 4.0],
              "resp": [0.35, 0.22, 0.15, 0.28],
              "other": [1.2, 0.8, 0.5, 1.0]}
CLASS_P = np.array([0.15, 0.45, 0.15, 0.25])
FREQS = np.fft.rfftfreq(T, 1.0 / FS)


def narrowband(rng, f0, bw):
    """White noise filtered with a Gaussian band around f0 (shape [...,1]); returns [..., T], unit std."""
    shp = f0.shape[:-1]
    w = rng.standard_normal(shp + (T,)).astype(np.float32)
    W = np.fft.rfft(w, axis=-1)
    filt = np.exp(-0.5 * ((FREQS - f0) / bw) ** 2)
    x = np.fft.irfft(W * filt, n=T, axis=-1).astype(np.float32)
    return x / (x.std(axis=-1, keepdims=True) + 1e-8)


def make_epochs(n, rng, signal_scale=1.0):
    y = rng.choice(4, size=n, p=CLASS_P)
    X = np.zeros((n, 12, T), dtype=np.float32)
    for g, idx in GROUPS.items():
        base = np.array(CLASS_FREQ[g], dtype=np.float32)[y]            # [n]
        f0 = (base * rng.uniform(0.88, 1.12, size=n).astype(np.float32))[:, None]  # [n,1]
        bw = np.maximum(0.12 * f0, 0.04)
        z = narrowband(rng, f0, bw)                                      # shared latent [n,T]
        for c in idx:
            gain_c = signal_scale * GAIN[g] * rng.uniform(0.8, 1.2)
            X[:, c] += gain_c * z
    # channel-specific nuisance: broadband noise + class-independent narrowband texture
    for c in range(12):
        X[:, c] += 1.0 * rng.standard_normal((n, T)).astype(np.float32)
        ft = rng.uniform(1.0, 20.0, size=(n, 1)).astype(np.float32)
        X[:, c] += 0.9 * narrowband(rng, ft, np.maximum(0.1 * ft, 0.3))
    X -= X.mean(axis=-1, keepdims=True)
    X /= X.std(axis=-1, keepdims=True) + 1e-8
    np.clip(X, -6, 6, out=X)
    ev = {
        "Arousal": (rng.random(n) < 0.15).astype(np.int64),
        "Hypopnea": (rng.random(n) < 0.20).astype(np.int64),
        "Oxygen Desaturation": (rng.random(n) < 0.15).astype(np.int64),
    }
    return X, y.astype(np.int64), ev


def gen_chunked(n, seed, chunk=500):
    rng = np.random.default_rng(seed)
    Xs, ys, evs = [], [], {k: [] for k in ["Arousal", "Hypopnea", "Oxygen Desaturation"]}
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        X, y, ev = make_epochs(m, rng)
        Xs.append(X)
        ys.append(y)
        for k in evs:
            evs[k].append(ev[k])
    return np.concatenate(Xs), np.concatenate(ys), {k: np.concatenate(v) for k, v in evs.items()}


def write_schema_layout(out_dir, seed, n_patients=24, epochs_per_patient=6):
    """Tiny dataset in the repo's CSV + parquet layout, to exercise SleepEpochDataset."""
    import pandas as pd
    rng = np.random.default_rng(seed + 999)
    csv_dir = os.path.join(out_dir, "schema_layout")
    os.makedirs(csv_dir, exist_ok=True)
    split_of = {"train": range(0, 16), "valid": range(16, 20), "test": range(20, 24)}
    for split, rows in split_of.items():
        pat, ep = [], []
        for p in rows:
            nsrr = f"synth-{p:04d}"
            # relative path (cwd is the mounted /work folder when the check runs)
            head = os.path.join(csv_dir, "psg", nsrr).replace("\\", "/")
            os.makedirs(os.path.join(csv_dir, "psg", nsrr), exist_ok=True)
            X, y, ev = make_epochs(epochs_per_patient, rng)
            for e in range(epochs_per_patient):
                df = pd.DataFrame(X[e].T.astype(np.float32), columns=CHANNELS)
                df.index = np.arange(T) / float(FS)          # float seconds index
                df.to_parquet(os.path.join(csv_dir, "psg", nsrr, f"epoch-{e:05d}_all.parquet"))
                ep.append(dict(nsrrid=nsrr, seg_id=e, dataset_name="synth", epoch_id=e, path_head=head,
                               Stage=int(y[e]), Arousal=int(ev["Arousal"][e]), Hypopnea=int(ev["Hypopnea"][e]),
                               **{"Oxygen Desaturation": int(ev["Oxygen Desaturation"][e])}))
            pat.append(dict(nsrrid=nsrr, dataset_name="synth"))
        pd.DataFrame(pat).to_csv(os.path.join(csv_dir, f"patient_pretrain_{split}.csv"), index=False)
        pd.DataFrame(ep).to_csv(os.path.join(csv_dir, f"epoch_pretrain_{split}.csv"), index=False)
    return csv_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/data")
    ap.add_argument("--n_pre", type=int, default=4000)
    ap.add_argument("--n_probe_train", type=int, default=1200)
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, n, sd in [("pretrain", a.n_pre, a.seed + 1), ("probe_train", a.n_probe_train, a.seed + 2),
                        ("test", a.n_test, a.seed + 3)]:
        X, y, ev = gen_chunked(n, sd)
        np.savez(os.path.join(a.out, f"{name}.npz"), X=X, y=y, **{k.replace(" ", "_"): v for k, v in ev.items()})
        print(f"{name}: X{X.shape} {X.dtype} y{np.bincount(y).tolist()} min={X.min():.2f} max={X.max():.2f}")
    d = write_schema_layout(a.out, a.seed)
    print("schema layout written to", d)


if __name__ == "__main__":
    main()
