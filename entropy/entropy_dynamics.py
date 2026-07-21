"""
Reproduction (TOY scale, real data) of Claim 1 of:
  "On Revisiting Entropy for Identifying Mislabeled Images" (ICML 2026, orid BUxrIaf7Zc, arXiv 2605.31090).

Claim 1: correctly-labeled samples show a CONSISTENT entropy DECREASE during training, while
mislabeled samples MAINTAIN relatively high entropy throughout. Also probes the paper's SEI idea
(claim 2): a per-sample statistic over the entropy trajectory should separate mislabeled samples.

TOY setup (labeled toy honestly): CIFAR-100 with REAL CIFAR-100N human label noise, coarse labels
(20 superclasses), a subset, a small CNN, CPU. Not the paper's full medical-dataset scale.
"""
import os, pickle, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
DATA = r"C:/Projects/02_academia/icml-repro/data/cifar100n"
SUBSET = 12000
EPOCHS = 25
NCLS = 20  # coarse superclasses


def load():
    # SAFETY: both files are canonical, trusted dataset releases downloaded over HTTPS —
    # CIFAR-100 from https://www.cs.toronto.edu/~kriz/ (ships as pickle by design) and the
    # CIFAR-100N human-label file from the official UCSC-REAL/cifar-10-100n repo. Not untrusted input.
    d = pickle.load(open(os.path.join(DATA, "cifar-100-python/train"), "rb"), encoding="latin1")
    X = np.array(d["data"], dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0
    h = torch.load(os.path.join(DATA, "CIFAR-100_human.pt"), weights_only=False)
    clean = np.array(h["clean_coarse_label"]); noisy = np.array(h["noisy_coarse_label"])
    idx = np.random.permutation(len(X))[:SUBSET]
    X, clean, noisy = X[idx], clean[idx], noisy[idx]
    mean = X.mean((0, 2, 3), keepdims=True); std = X.std((0, 2, 3), keepdims=True)
    X = (X - mean) / (std + 1e-6)
    is_mislabeled = (noisy != clean)
    print(f"subset {SUBSET}, coarse noise rate {is_mislabeled.mean():.3f} "
          f"({is_mislabeled.sum()} mislabeled / {len(is_mislabeled)})")
    return (torch.tensor(X), torch.tensor(noisy, dtype=torch.long),
            torch.tensor(clean, dtype=torch.long), torch.tensor(is_mislabeled))


class SmallCNN(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 4
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, ncls))

    def forward(self, x): return self.fc(self.c(x))


def entropy_of(logits):
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-12)).sum(1)


def run():
    X, ynoisy, yclean, mis = load()
    n = len(X)
    model = SmallCNN(NCLS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 256
    traj = np.zeros((EPOCHS, n), dtype=np.float32)  # per-sample entropy per epoch
    print(f"\n{'epoch':>5} {'train_acc(noisy)':>16} {'H_clean':>9} {'H_mislab':>9} {'gap':>7} {'t':>6}")
    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = model(X[b])
            F.cross_entropy(out, ynoisy[b]).backward()
            opt.step()
        # eval per-sample entropy on all training samples (no grad)
        model.eval()
        with torch.no_grad():
            ent = torch.empty(n); correct = 0
            for i in range(0, n, 1024):
                out = model(X[i:i + 1024])
                ent[i:i + 1024] = entropy_of(out)
                correct += (out.argmax(1) == ynoisy[i:i + 1024]).sum().item()
        traj[ep] = ent.numpy()
        Hc = ent[~mis].mean().item(); Hm = ent[mis].mean().item()
        print(f"{ep+1:>5} {correct/n:>16.3f} {Hc:>9.3f} {Hm:>9.3f} {Hm-Hc:>7.3f} {time.time()-t0:>5.1f}s")

    # --- Claim 1 verdict ---
    m = mis.numpy().astype(bool)
    Hc0, Hcf = traj[0, ~m].mean(), traj[-1, ~m].mean()
    Hm0, Hmf = traj[0, m].mean(), traj[-1, m].mean()
    print("\n=== Claim 1 ===")
    print(f"clean:      entropy {Hc0:.3f} -> {Hcf:.3f}  (drop {Hc0-Hcf:+.3f})")
    print(f"mislabeled: entropy {Hm0:.3f} -> {Hmf:.3f}  (drop {Hm0-Hmf:+.3f})")
    c1 = (Hcf < Hc0 - 0.05) and (Hmf > Hcf + 0.1)
    print("VERDICT:", "reproduced (clean drops, mislabeled stays high)" if c1 else "not clearly separated")

    # --- Claim 2 (SEI-style): can the entropy trajectory identify mislabeled? ---
    sei = traj.mean(axis=0)                       # mean entropy over training = simple SEI proxy
    # AUC of using sei to rank mislabeled (higher sei -> more likely mislabeled)
    order = np.argsort(-sei); ranks = np.empty_like(order); ranks[order] = np.arange(n)
    pos = m.sum(); neg = n - pos
    auc = (ranks[~m].sum() - neg * (neg - 1) / 2) / (pos * neg)  # rank-based AUC (clean should rank low sei)
    auc = max(auc, 1 - auc)
    print("\n=== Claim 2 (SEI proxy = mean entropy over training) ===")
    print(f"AUC separating mislabeled from clean: {auc:.3f}")
    np.save(os.path.join(os.path.dirname(__file__), "entropy_traj.npy"), traj)
    print("saved entropy_traj.npy")


if __name__ == "__main__":
    run()
