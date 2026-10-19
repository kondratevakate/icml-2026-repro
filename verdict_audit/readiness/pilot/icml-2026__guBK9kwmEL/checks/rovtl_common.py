"""Shared helpers for the RoVTL checks. Runs INSIDE the container only (paths are container paths).

Repository classes are imported unchanged from /work/work/repo/RoVTL.  The loops below mirror the repository's
pretraining.py (CLIP loss summed over Matryoshka heads, AdamW wd 1e-4, temperature 0.1) and train_multilabel.py
(unimodal head losses + L_more + L_less + lambda * max(L_more - L_less, 0), AdamW wd 1e-4, grad-clip 40)."""
import json
import os
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')  # CPU is shared with other jobs; avoid spin-waiting
import random
import sys
import time

sys.path.insert(0, "/work/checks")
sys.path.insert(0, "/work/work/repo/RoVTL")
import fasttext_stub  # noqa: E402

fasttext_stub.install()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tarte_ai import TARTE_TablePreprocessor  # noqa: E402

REDUCED = False


def init(reduced=False):
    """Import the repository modules.  reduced=True keeps every 3rd numerical / every 2nd lifestyle categorical
    attribute of the repository's attribute lists (CPU budget: TARTE cost scales with the number of attribute tokens)
    by patching the attribute lists BEFORE the repo dataset modules are imported; reduced=False uses the full lists."""
    global REDUCED, NUMERICAL_ATTRIBUTES, CATEGORICAL_ATTRIBUTES, CLASSIFICATION_CATEGORICAL_ATTRIBUTES
    global CLASSIFICATION_NUMERICAL_ATTRIBUTES, LABEL_MAPPING, DISEASE_MAPPING
    global MRContrastiveDatasetH5, DataCollator, MRDownstreamDatasetH5, DownstreamCollator
    global ImageTabularPretraining, MutlimodalFusionClassifier, CLIPLoss, map_categorical_values, ResNetTokens
    import dataset.attributes as A
    if reduced:
        A.NUMERICAL_ATTRIBUTES = A.NUMERICAL_ATTRIBUTES[::3]
        A.CATEGORICAL_ATTRIBUTES = A.CATEGORICAL_ATTRIBUTES[0:36:2]
        A.CLASSIFICATION_CATEGORICAL_ATTRIBUTES = A.CLASSIFICATION_CATEGORICAL_ATTRIBUTES[::2]
        A.CLASSIFICATION_NUMERICAL_ATTRIBUTES = A.CLASSIFICATION_NUMERICAL_ATTRIBUTES[::3]
    REDUCED = reduced
    NUMERICAL_ATTRIBUTES, CATEGORICAL_ATTRIBUTES = A.NUMERICAL_ATTRIBUTES, A.CATEGORICAL_ATTRIBUTES
    CLASSIFICATION_CATEGORICAL_ATTRIBUTES = A.CLASSIFICATION_CATEGORICAL_ATTRIBUTES
    CLASSIFICATION_NUMERICAL_ATTRIBUTES = A.CLASSIFICATION_NUMERICAL_ATTRIBUTES
    LABEL_MAPPING, DISEASE_MAPPING = A.LABEL_MAPPING, A.DISEASE_MAPPING
    from dataset.MRContrastiveDatasetH5 import MRContrastiveDatasetH5 as _a, DataCollator as _b
    from dataset.MRDownstreamDataset import MRDownstreamDatasetH5 as _c, DownstreamCollator as _d
    from models.imgtab_pretraining import ImageTabularPretraining as _e
    from models.fusion import MutlimodalFusionClassifier as _f
    from utils.clip_loss import CLIPLoss as _g
    from utils.utils import map_categorical_values as _h
    import models.resnet_tokens as rt
    MRContrastiveDatasetH5, DataCollator, MRDownstreamDatasetH5, DownstreamCollator = _a, _b, _c, _d
    ImageTabularPretraining, MutlimodalFusionClassifier, CLIPLoss, map_categorical_values = _e, _f, _g, _h
    ResNetTokens = rt.ResNetTokens
    return A


def patch_resnet_tokens_for_small_inputs():
    """Repo ResNetTokens.forward uses out.squeeze().flatten(start_dim=2), which fails when the last feature map has
    no spatial extent left (inputs smaller than 64x64).  Same computation without squeeze(): identical output for
    the paper's 128x128 scans (checked numerically in check_C00657.py), also valid for 16x16 scans."""
    import torch as _t

    def forward(self, x):
        out = self.resnet(x)[-1] if self.dim == 3 else self.resnet(x)  # (B, 2048, D, H, W)
        tokens = out.flatten(start_dim=2).permute(0, 2, 1)
        feature = self.gap(out).flatten(start_dim=1)
        return feature, tokens
    ResNetTokens.forward = forward


torch.set_num_threads(4)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def now():
    return time.strftime("%H:%M:%S")


def log(*a):
    print(f"[{now()}]", *a, flush=True)


# ----------------------------------------------------------------------------------------- pre-training pieces
def make_pretrain_prepper(data_dir):
    attributes = NUMERICAL_ATTRIBUTES + CATEGORICAL_ATTRIBUTES
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))[attributes]
    df = map_categorical_values(df, CATEGORICAL_ATTRIBUTES, LABEL_MAPPING)
    df = df.rename(columns=DISEASE_MAPPING)
    prep = TARTE_TablePreprocessor()
    prep.fit(df)
    return prep


def pretrain(data_dir, missing_tabular, epochs, batch_size, lr, seed, ckpt_path, prep=None, eval_cb=None, max_batches=None, freeze_image_trunk=False):
    """Mirror of RoVTL/pretraining.py main() for dataset == 'UKBB' (image augmentation off: scans are 16x16)."""
    seed_all(seed)
    prep = prep or make_pretrain_prepper(data_dir)
    collator = DataCollator(prep)
    train = MRContrastiveDatasetH5(os.path.join(data_dir, "train.h5"), augmentation_rate=-1,
                                   simulate_missing=missing_tabular)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=collator)
    model = ImageTabularPretraining(dim=3)
    if freeze_image_trunk:  # CPU budget: random-init 3D ResNet-50 trunk kept fixed (its projection heads and the whole table tower train)
        for p in model.image_encoder.resnet.parameters():
            p.requires_grad = False
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    crit = CLIPLoss(temperature=0.1)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        losses = []
        for bi, data in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            out1, out2, _ = model(data["x"], data["edge_attr"], data["mask"], data["image"].float())
            loss = 0
            for i, o in enumerate(out1):
                loss = loss + crit(o, out2[i])[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        hist.append(float(np.mean(losses)))
        log(f"pretrain missing={missing_tabular} epoch {ep + 1}/{epochs} loss {hist[-1]:.4f} elapsed {time.time() - t0:.0f}s")
        if eval_cb is not None:
            eval_cb(model, ep + 1)
            model.train()
        torch.save(model.state_dict(), ckpt_path)  # keep the latest epoch on disk
    return model, hist


class FracView(Dataset):
    """Deterministic per-sample random subset (fraction f) of the attributes of a fully-observed contrastive item."""

    def __init__(self, base, frac, seed):
        self.base, self.frac, self.seed = base, frac, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        rng = random.Random(self.seed * 100003 + i)
        if self.frac < 1.0:
            for key in ("categorical", "continuous"):
                names = list(item[key].keys())
                k = int(round(self.frac * len(names)))
                keep = set(rng.sample(names, k))
                item[key] = {n: v for n, v in item[key].items() if n in keep}
        return item


@torch.no_grad()
def encode_test(model, data_dir, prep, fracs, split="test", seed=1234, batch_size=64):
    """Return image projections (list per head) and, for each fraction, table projections (list per head)."""
    model.eval()
    collator = DataCollator(prep)
    base = MRContrastiveDatasetH5(os.path.join(data_dir, f"{split}.h5"), augmentation_rate=-1, simulate_missing=False)
    img_proj = None
    tab_proj = {}
    for f in fracs:
        ds = FracView(base, f, seed)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=collator)
        t_list, i_list = None, None
        for data in loader:
            out_tab, out_img, _ = model(data["x"], data["edge_attr"], data["mask"], data["image"].float())
            if t_list is None:
                t_list = [[] for _ in out_tab]
                i_list = [[] for _ in out_img]
            for h, o in enumerate(out_tab):
                t_list[h].append(torch.nn.functional.normalize(o, dim=1))
            for h, o in enumerate(out_img):
                i_list[h].append(torch.nn.functional.normalize(o, dim=1))
        tab_proj[f] = [torch.cat(x) for x in t_list]
        if img_proj is None:
            img_proj = [torch.cat(x) for x in i_list]
    return img_proj, tab_proj


def retrieval_metrics(img_proj, tab_proj_f, tab_proj_full):
    n = img_proj[0].shape[0]
    sim = sum(i @ t.T for i, t in zip(img_proj, tab_proj_f)) / len(img_proj)  # (img, tab)
    ranks_i2t = (sim.argsort(dim=1, descending=True) == torch.arange(n)[:, None]).float().argmax(dim=1)
    ranks_t2i = (sim.T.argsort(dim=1, descending=True) == torch.arange(n)[:, None]).float().argmax(dim=1)
    cos_full = float(np.mean([(a * b).sum(1).mean().item() for a, b in zip(tab_proj_f, tab_proj_full)]))
    return {
        "i2t_top1": float((ranks_i2t == 0).float().mean()), "i2t_top5": float((ranks_i2t < 5).float().mean()),
        "t2i_top1": float((ranks_t2i == 0).float().mean()), "t2i_top5": float((ranks_t2i < 5).float().mean()),
        "cos_to_full_table": cos_full,
    }


# ----------------------------------------------------------------------------------------- down-stream pieces
def make_downstream_prepper(data_dir):
    attributes = NUMERICAL_ATTRIBUTES + CLASSIFICATION_CATEGORICAL_ATTRIBUTES  # as train_multilabel.py
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))[attributes]
    df = map_categorical_values(df, CLASSIFICATION_CATEGORICAL_ATTRIBUTES, LABEL_MAPPING)
    prep = TARTE_TablePreprocessor()
    prep.fit(df)
    return prep


class FirstHalfCollator:
    """DownstreamCollator stacks [set-1 rows; set-2 rows]; keep only the first B rows (used for evaluation)."""

    def __init__(self, base):
        self.base = base

    def __call__(self, batch):
        out = self.base(batch)
        b = len(batch)
        return {k: (v[:b] if torch.is_tensor(v) else v[:b]) for k, v in out.items()}


class SameFullTwiceCollator:
    """ARM_FULL training: both halves carry the complete table (no missingness anywhere in tuning)."""

    def __init__(self, base):
        self.base = base

    def __call__(self, batch):
        b2 = []
        for b in batch:
            c = dict(b)
            c["categorical2"], c["continuous2"] = dict(b["categorical1"]), dict(b["continuous1"])
            c["tab_missing2"] = b["tab_missing1"]
            b2.append(c)
        return self.base(b2)


def load_classifier(ckpt, frozen=True):
    return MutlimodalFusionClassifier(num_classes=4, image_checkpoint=ckpt, tabular_checkpoint=ckpt,
                                      freeze_image=frozen, freeze_tabular=frozen, dim=3)


def finetune(data_dir, ckpt, mode, epochs, batch_size, lr, seed, prep, tabmofe_weight, frozen=True, max_batches=None):
    """Mirror of RoVTL/train_multilabel.py. mode 'rovtl': random subsets + nested subset (simulate_missing=True,
    sample_more=0.3) ; mode 'full': complete table in both halves."""
    seed_all(seed)
    if mode == "rovtl":
        ds = MRDownstreamDatasetH5(h5_path=os.path.join(data_dir, "train.h5"), filter_csv=os.path.join(data_dir, "train.csv"),
                                   simulate_missing=True, sample_more=0.3, augmentation_rate=-1)
        collator = DownstreamCollator(prep)
    else:
        ds = MRDownstreamDatasetH5(h5_path=os.path.join(data_dir, "train.h5"), filter_csv=os.path.join(data_dir, "train.csv"),
                                   simulate_missing=False, sample_more=0.3, augmentation_rate=-1)
        collator = SameFullTwiceCollator(DownstreamCollator(prep))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=collator)
    clf = load_classifier(ckpt, frozen)
    clf.train()
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        ls = []
        for bi, data in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            opt.zero_grad()
            logits, img_logits, tab_logits = clf(data["x"], data["edge_attr"], data["mask"], data["image"].float())
            lab = data["label"]
            uni = crit(img_logits.squeeze()[:batch_size], lab[:batch_size]) + crit(tab_logits[:batch_size], lab[:batch_size])
            uni.backward(retain_graph=True)
            loss_more = crit(logits[:batch_size], lab[:batch_size])
            loss_less = crit(logits[batch_size:], lab[batch_size:])
            tab = torch.maximum(loss_more - loss_less, torch.tensor(0.0))
            loss = loss_more + loss_less + tabmofe_weight * tab
            loss.backward()
            nn.utils.clip_grad_norm_(clf.parameters(), max_norm=40, norm_type=2)
            opt.step()
            ls.append(float(loss.detach()))
        hist.append(float(np.mean(ls)))
        log(f"finetune mode={mode} epoch {ep + 1}/{epochs} loss {hist[-1]:.4f} elapsed {time.time() - t0:.0f}s")
    return clf, hist


@torch.no_grad()
def eval_auc(clf, data_dir, prep, frac, draw_seed, split="test", batch_size=64):
    """Macro AUROC (4 labels) with a fraction `frac` of the tabular attributes available (repo num_present switch)."""
    from torchmetrics import AUROC
    clf.eval()
    random.seed(draw_seed)
    ds = MRDownstreamDatasetH5(h5_path=os.path.join(data_dir, f"{split}.h5"), filter_csv=os.path.join(data_dir, f"{split}.csv"),
                               simulate_missing=False, augmentation_rate=-1, num_present=frac)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
                        collate_fn=FirstHalfCollator(DownstreamCollator(prep)))
    preds, labs = [], []
    for data in loader:
        logits, _, _ = clf(data["x"], data["edge_attr"], data["mask"], data["image"].float())
        preds.append(torch.sigmoid(logits))
        labs.append(data["label"].long())
    auc = AUROC(num_labels=4, average="macro", task="multilabel")
    return float(auc(torch.cat(preds), torch.cat(labs)))
