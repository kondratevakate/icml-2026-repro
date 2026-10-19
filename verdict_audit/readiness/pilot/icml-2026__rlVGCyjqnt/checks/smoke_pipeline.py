"""Pipeline smoke test + generator sanity (no claim outcome depends on this file).

1. The repo's SleepEpochDataset reads the synthetic parquet/CSV layout -> [12,1920].
2. The repo's DINOCLSModel (OSF) runs shared_step (forward + loss + backward) on a synthetic batch.
3. Hand-crafted band-power features + logistic regression: how learnable is the synthetic stage label
   from all channels / each channel group (generator sanity, done BEFORE any pretraining run).
"""
import sys, time, os
import numpy as np
import torch

sys.path.insert(0, "/work/work/repo")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

t0 = time.time()
print("torch", torch.__version__, "threads", torch.get_num_threads())

# ---- 1. repo dataset loader on the synthetic layout
from osf.datasets.pretrain_dataset import SleepEpochDataset
from train_config import TRAIN_EDF_COLS_UNI_ENC
ds = SleepEpochDataset(csv_dir="work/data/schema_layout", split="pretrain", train_edf_cols=TRAIN_EDF_COLS_UNI_ENC,
                       sample_rate=64, window_size=30, data_source="pretrain")
item = ds[0]
print("loader: len", len(ds), "psg", tuple(item["psg"].shape), item["psg"].dtype, "keys", sorted(item.keys()))
assert tuple(item["psg"].shape) == (12, 1920)

# ---- 2. repo DINOCLSModel smoke test
from osf.models.dino_model_cls import DINOCLSModel
torch.manual_seed(0)
model = DINOCLSModel(psg_encoder_name="vit_nano", num_leads=12, patch_size_time=128, patch_size_ch=4, lead_wise=1,
                     simclr_augmentation="chan_then_pcspan", lr=1e-4, max_epochs=1)
n_enc = sum(p.numel() for p in model.encoders["all"].parameters())
print("DINOCLSModel built; student encoder params (vit_nano):", n_enc)
d = np.load("work/data/pretrain.npz")
x = torch.from_numpy(d["X"][:16])
model.train()
ts = time.time()
loss_dict, metrics = model.shared_step({"psg": x}, 0)
loss_dict["loss"].backward()
print("DINO shared_step ok: loss=%.4f  dino=%.4f  (%.2fs for B=16)" % (loss_dict["loss"].item(), metrics["loss/dino"].item(), time.time() - ts))

# ---- 3. generator sanity with band-power features
t = np.load("work/data/probe_train.npz"); s = np.load("work/data/test.npz")
freqs = np.fft.rfftfreq(1920, 1 / 64)
edges = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2, 3, 4.5, 6, 8, 11, 16, 24, 32]


def bandpower(X):
    P = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    feats = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        feats.append(np.log(P[..., m].mean(-1) + 1e-6))
    return np.stack(feats, -1)  # [N,12,B]


Ftr, Fte = bandpower(t["X"]), bandpower(s["X"])
GR = {"all": list(range(12)), "brain": [8, 9, 10, 11], "resp": [4, 5, 6, 7], "other": [0, 1, 2, 3]}
for name, idx in GR.items():
    A, B = Ftr[:, idx].reshape(len(Ftr), -1), Fte[:, idx].reshape(len(Fte), -1)
    sc = StandardScaler().fit(A)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(A), t["y"])
    auc = roc_auc_score(s["y"], clf.predict_proba(sc.transform(B)), multi_class="ovr", average="macro")
    print("bandpower-logreg macro-AUC [%s channels]: %.3f" % (name, auc))
print("smoke total %.1fs" % (time.time() - t0))
