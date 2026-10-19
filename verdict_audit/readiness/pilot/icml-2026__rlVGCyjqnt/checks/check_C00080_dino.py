"""C00080 secondary information (no outcome weight): the repo's own DINOCLSModel (OSF) pre-trained with the three
augmentation presets on synthetic epochs via pytorch_lightning.Trainer (CPU), then the same linear probe as the SimCLR check.
One seed, one variant per process (run in parallel by run_C00080_dino.sh).
"""
import argparse, json, math, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/work/work/repo"); sys.path.insert(0, "/work/checks")
import pytorch_lightning as pl
import check_C00080 as c
from osf.models.dino_model_cls import DINOCLSModel


class DictDS(Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return {"psg": torch.from_numpy(self.X[i])}


class Wrap(nn.Module):
    def __init__(self, bb): super().__init__(); self.backbone = bb
    def embed(self, x): return self.backbone(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    pl.seed_everything(a.seed)
    # side observation: EMA momentum schedule of the repo when trainer.max_steps == -1 (as in main_pretrain.py, which sets max_epochs only)
    import types
    probe = types.SimpleNamespace(base_momentum=0.996)   # default base_momentum of DINOCLSModel
    print("repo _momentum(step, max_steps=1) for steps 0..5 (what on_train_batch_end computes when trainer.max_steps=-1):",
          [round(DINOCLSModel._momentum(probe, s, 1), 4) for s in range(6)])
    pre = np.load("work/data/pretrain.npz"); trn = np.load("work/data/probe_train.npz"); tst = np.load("work/data/test.npz")
    per_epoch = 100
    epochs = a.steps // per_epoch
    X = pre["X"][: per_epoch * a.bs]
    dl = DataLoader(DictDS(X), batch_size=a.bs, shuffle=True, num_workers=0, drop_last=True)
    model = DINOCLSModel(psg_encoder_name="vit_nano", num_leads=12, patch_size_time=c.PATCH_T, patch_size_ch=4, lead_wise=1,
                         simclr_augmentation=c.PRESETS[a.variant], lr=a.lr, max_epochs=epochs, training_steps_per_epoch=per_epoch)
    model.teacher_temp_warmup_iters = int(0.1 * a.steps)
    trainer = pl.Trainer(max_epochs=epochs, max_steps=a.steps, accelerator="cpu", devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, gradient_clip_val=3.0,
                         gradient_clip_algorithm="norm", num_sanity_val_steps=0, limit_val_batches=0, log_every_n_steps=50)
    t0 = time.time()
    trainer.fit(model, train_dataloaders=dl)
    secs = time.time() - t0
    print(f"[DINO {a.variant}] trained {trainer.global_step} steps in {secs:.0f}s")
    r = c.probe_eval(Wrap(model.encoders["all"].backbone), trn["X"], trn["y"], tst["X"], tst["y"])
    r.update(train_seconds=secs, steps=int(trainer.global_step))
    print(f"[DINO {a.variant}] full AUC {r['full']:.4f} | " + " ".join(f"{k} {r[k]:.4f}" for k in c.SCENARIOS) +
          f" | missing_mean {r['missing_mean']:.4f}")
    json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
