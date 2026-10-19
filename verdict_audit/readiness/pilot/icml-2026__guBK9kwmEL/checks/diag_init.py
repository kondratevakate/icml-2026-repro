"""Diagnostic (bring-up, no claim metric): are the two towers input-sensitive at initialisation / after the AUG attempt-1 run?
Mean pairwise cosine between L2-normalised projections of different samples (1.0 = constant output)."""
import sys, torch
import rovtl_common as rc
rc.init(reduced=True)
from rovtl_common import *
D = "/work/work/data/easy34x8"
prep = make_pretrain_prepper(D)
torch.manual_seed(5)
model = ImageTabularPretraining(dim=3)
for tag, sd in (("init", None), ("after attempt-1 AUG", "/work/work/ckpt/AUG.pth")):
    if sd:
        model.load_state_dict(torch.load(sd))
    model.train()  # BN with batch statistics, as in training
    img, tab = encode_test(model, D, prep, [0.5, 1.0])
    for name, P in (("image", img), ("table f=1.0", tab[1.0]), ("table f=0.5", tab[0.5])):
        vals = []
        for h in P:
            s = h @ h.T
            n = s.shape[0]
            vals.append(float((s.sum() - n) / (n * (n - 1))))
        print(f"{tag}: {name}: mean pairwise cosine between samples per head {[round(v, 4) for v in vals]}", flush=True)
    c = float(np.mean([(a * b).sum(1).mean().item() for a, b in zip(tab[0.5], tab[1.0])]))
    print(f"{tag}: cos(table f=0.5, same-sample table f=1.0) = {c:.5f}", flush=True)
